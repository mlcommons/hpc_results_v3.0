"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import datetime
import os
import random
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import torch
from apex.optimizers import FusedAdam
from ocpmodels.common import distutils
from ocpmodels.common.data_parallel import BalancedBatchSampler
from ocpmodels.common.fp16_optimizer import FP16_Optimizer
from ocpmodels.common.meter import Meter
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import Prefetcher, save_checkpoint
from ocpmodels.datasets.trajectory_lmdb import get_num_samples
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.scheduler import LRScheduler
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from torch_geometric.data import Batch


def collater(data_list):
    return Batch.from_data_list(data_list)


@registry.register_trainer("base")
class BaseTrainer(ABC):
    def __init__(
        self,
        config,
        run_dir=None,
        print_every=100,
        seed=None,
        logger="tensorboard",
        local_rank=0,
        amp=False,
        cpu=False,
        name="base_trainer",
    ):
        self.name = name
        self.cpu = cpu
        self.start_step = 0
        self.num_targets = 1

        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = "cpu"
            self.cpu = True  # handle case when `--cpu` isn't specified
            # but there are no gpu devices available

        if run_dir is None:
            run_dir = os.getcwd()

        timestamp = torch.tensor(datetime.datetime.now().timestamp(), device=self.device)
        # create directories from master rank only
        distutils.broadcast(timestamp, 0)
        timestamp = datetime.datetime.fromtimestamp(timestamp.int()).strftime("%Y-%m-%d-%H-%M-%S")

        identifier = config["identifier"]
        if identifier:
            timestamp += "-{}".format(identifier)

        self.config = config
        self.loss_c = torch.tensor(
            self.config.get("force_coefficient", 30) * distutils.get_world_size(), device=self.device
        )
        self.config["cmd"] = {
            "identifier": identifier,
            "print_every": print_every,
            "seed": seed,
            "timestamp": timestamp,
            "results_dir": run_dir,
        }
        self.config["gpus"] = distutils.get_world_size() if not self.cpu else 0

        # AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler() if amp else None

        if distutils.is_master():
            os.makedirs(self.config["cmd"]["results_dir"], exist_ok=True)
            print(self.config)

        self.load()
        self.evaluator = Evaluator(task=name)

    def load(self):
        self.load_sizes()
        self.load_seed_from_config()
        self.load_logger()
        self.load_model()
        self.load_optimizer()
        self.load_extras()

    def load_sizes(self):
        n_train, n_val = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)
        if distutils.is_master():
            n_train = torch.tensor(get_num_samples(self.config, "train"), device=self.device)
            n_val = torch.tensor(get_num_samples(self.config, "val"), device=self.device)
        distutils.global_barrier(self.config)
        torch.distributed.broadcast(n_train, 0)
        torch.distributed.broadcast(n_val, 0)
        self.config["train_size"] = n_train.item()
        self.config["val_size"] = n_val.item()

    def load_seed_from_config(self):
        # https://pytorch.org/docs/stable/notes/randomness.html
        seed = self.config["cmd"]["seed"]
        if seed is None:
            seed = torch.tensor(random.randint(0, 65536), device=self.device)
            torch.distributed.broadcast(seed, 0)
            seed = seed.item()
        self.config["cmd"]["seed"] = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_sampler(self, dataset, batch_size, shuffle):
        if "load_balancing" in self.config:
            balancing_mode = self.config["load_balancing"]
            force_balancing = True
        else:
            balancing_mode = "atoms"
            force_balancing = False

        sampler = BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            device=self.device,
            mode=balancing_mode,
            shuffle=shuffle,
            force_balancing=force_balancing,
        )
        return sampler

    def get_dataloader(self, dataset, sampler):
        return DataLoader(
            dataset,
            collate_fn=collater,
            num_workers=self.config["num_workers"],
            prefetch_factor=self.config["prefetch_factor"],
            pin_memory=True,
            batch_sampler=sampler,
        )

    def load_logger(self):
        self.logger = None

    @abstractmethod
    def load_task(self):
        """Derived classes should implement this function."""

    def load_model(self):
        # Build model
        if distutils.is_master():
            print("### Loading model: {}".format(self.config["name"]))

        # TODO(abhshkdz): Eventually move towards computing features on-the-fly
        # and remove dependence from `.edge_attr`.
        bond_feat_dim = None
        if self.config["dataset"] in [
            "trajectory_lmdb",
            "single_point_lmdb",
        ]:
            bond_feat_dim = self.config.get("num_gaussians", 50)
        else:
            raise NotImplementedError

        model_atributes = {
            "use_pbc": self.config["use_pbc"],
            "regress_forces": self.config["regress_forces"],
            "hidden_channels": self.config["hidden_channels"],
            "num_blocks": self.config["num_blocks"],
            "int_emb_size": self.config.get("int_emb_size", 64),
            "basis_emb_size": self.config.get("basis_emb_size", 8),
            "out_emb_channels": self.config["out_emb_channels"],
            "num_spherical": self.config["num_spherical"],
            "num_radial": self.config["num_radial"],
            "otf_graph": self.config.get("otf_graph", False),
            "cutoff": self.config["cutoff"],
            "envelope_exponent": self.config.get("envelope_exponent", 5),
            "num_before_skip": self.config["num_before_skip"],
            "num_after_skip": self.config["num_after_skip"],
            "num_output_layers": self.config["num_output_layers"],
        }

        self.model = registry.get_model_class(self.config["name"])(
            None,
            bond_feat_dim,
            self.num_targets,
            O2=self.config["O2"],
            device=self.device,
            **model_atributes,
        ).to(self.device)

        if distutils.is_master():
            print("### Loaded {} with {} parameters.".format(self.model.__class__.__name__, self.model.num_params))

        if distutils.initialized():
            self.model = DistributedDataParallel(
                self.model,
                static_graph=True,
                gradient_as_bucket_view=True,
                bucket_cap_mb=self.config["bucket_cap_mb"],
                device_ids=[self.device],
            )

    def load_pretrained(self, checkpoint_path=None, ddp_to_dp=False):
        if checkpoint_path is None or os.path.isfile(checkpoint_path) is False:
            print(f"Checkpoint: {checkpoint_path} not found!")
            return False

        print("### Loading checkpoint from: {}".format(checkpoint_path))

        checkpoint = torch.load(
            checkpoint_path,
            map_location=(torch.device("cpu") if self.cpu else None),
        )

        self.start_step = checkpoint.get("step", 0)

        # Load model, optimizer, normalizer state dict.
        # if trained with ddp and want to load in non-ddp, modify keys from
        # module.module.. -> module..
        if ddp_to_dp:
            new_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                name = k[7:]
                new_dict[name] = v
            self.model.load_state_dict(new_dict)
        else:
            self.model.load_state_dict(checkpoint["state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            self.scheduler.scheduler.load_state_dict(checkpoint["scheduler"])

        for key in checkpoint["normalizers"]:
            if key in self.normalizers:
                self.normalizers[key].load_state_dict(checkpoint["normalizers"][key])
            if self.scaler and checkpoint["amp"]:
                self.scaler.load_state_dict(checkpoint["amp"])
        return True

    def load_optimizer(self):
        self.optimizer = FusedAdam(
            params=self.model.parameters(), lr=self.config["lr_initial"], weight_decay=self.config["weight_decay"]
        )

        if self.config["O2"]:
            self.optimizer = FP16_Optimizer(self.optimizer)

    def load_extras(self):
        self.scheduler = LRScheduler(self.optimizer, self.config)
        self.meter = Meter(split="train")

    def save_state(self, epoch, step, metrics):
        state = {
            "epoch": epoch,
            "step": step,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.scheduler.state_dict() if self.scheduler.scheduler_type != "Null" else None,
            "normalizers": {key: value.state_dict() for key, value in self.normalizers.items()},
            "val_metrics": metrics,
            "amp": self.scaler.state_dict() if self.scaler else None,
        }
        return state

    def save(self, epoch, step, metrics):
        if distutils.is_master():
            save_checkpoint(
                self.save_state(epoch, step, metrics),
                self.config["cmd"]["results_dir"],
            )

    @abstractmethod
    def train(self):
        """Derived classes should implement this function."""

    @torch.no_grad()
    def validate(self, split="val", epoch=None):
        if distutils.is_master():
            print("### Evaluating on {}.".format(split))

        self.model.eval()
        evaluator, metrics = Evaluator(task=self.name), {}
        loader = self.val_loader if split == "val" else self.test_loader

        prefetcher = Prefetcher(loader, self.device)
        batch = prefetcher.next()
        while batch is not None:
            out = self.model(batch)
            metrics = self._compute_metrics({"forces": out}, batch, evaluator, metrics)
            batch = prefetcher.next()

        aggregated_metrics = {}
        for k in metrics:
            aggregated_metrics[k] = {
                "total": distutils.all_reduce(metrics[k]["total"], average=False, device=self.device),
                "numel": distutils.all_reduce(metrics[k]["numel"], average=False, device=self.device),
            }
            aggregated_metrics[k]["metric"] = aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
        metrics = aggregated_metrics

        log_dict = {k: metrics[k]["metric"] for k in metrics}
        log_dict.update({"epoch": epoch + 1})
        if distutils.is_master():
            log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
            print(", ".join(log_str))

        return metrics

    @abstractmethod
    def _forward(self, batch_list):
        """Derived classes should implement this function."""

    @abstractmethod
    def _compute_loss(self, out, batch_list):
        """Derived classes should implement this function."""

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
