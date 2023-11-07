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


import gc
import os
import shutil
import time
from functools import partial
from glob import glob
from multiprocessing.pool import ThreadPool

import torch
import torch.distributed as dist
from mlperf_common.frameworks.pyt import PyTCommunicationHandler
from mlperf_common.logging import MLLoggerWrapper
from mlperf_logging import mllog
from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import Prefetcher
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.trainers.base_trainer import BaseTrainer


def copy_dataset(config):
    src, dst, jobs = config["data"], config["data_target"], config["jobs"]
    TRAIN_SUFFIX, VAL_SUFFIX = "oc20_data/s2ef/2M/train", "oc20_data/s2ef/all/val_id"
    SRC_TRAIN, SRC_VAL = f"{src}/{TRAIN_SUFFIX}", f"{src}/{VAL_SUFFIX}"
    DST_TRAIN, DST_VAL = f"{dst}/{TRAIN_SUFFIX}", f"{dst}/{VAL_SUFFIX}"

    os.makedirs(DST_TRAIN, exist_ok=True)
    os.makedirs(DST_VAL, exist_ok=True)
    local_rank = config["local_rank"]
    local_size = config["nproc_per_node"]

    train, val = sorted(glob(f"{SRC_TRAIN}/*")), sorted(glob(f"{SRC_VAL}/*"))
    train, val = train[local_rank::local_size], val[local_rank::local_size]

    copy_train = partial(shutil.copy, dst=DST_TRAIN)
    with ThreadPool(jobs) as p:
        p.map(copy_train, train)

    copy_val = partial(shutil.copy, dst=DST_VAL)
    with ThreadPool(jobs) as p:
        p.map(copy_val, val)

    distutils.local_barrier(config)


def rank_in_group(config, ranks):
    return config["nodes_for_eval"] == 0 or (config["nodes_for_eval"] > 0 and config["global_rank"] in config[ranks])


@registry.register_trainer("mlperf_forces")
class MLPerfForcesTrainer(BaseTrainer):
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
    ):
        super().__init__(
            config,
            run_dir=run_dir,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            name="s2ef",
        )

    def load_task(self):
        if distutils.is_master():
            print("### Loading dataset: {}".format(self.config["dataset"]))

        # Data staging
        copy_dataset(self.config)

        if self.config["dataset"] == "trajectory_lmdb":
            self.train_dataset = registry.get_dataset_class(self.config["dataset"])(self.config, mode="train")

            self.train_sampler = self.get_sampler(
                self.train_dataset,
                self.config["batch_size"],
                shuffle=True,
            )
            self.train_loader = self.get_dataloader(
                self.train_dataset,
                self.train_sampler,
            )

            self.val_loader = self.test_loader = None
            self.val_sampler = self.test_sampler = None
            if "val_dataset" in self.config:
                self.val_dataset = registry.get_dataset_class(self.config["dataset"])(self.config, mode="val")
                self.val_sampler = self.get_sampler(
                    self.val_dataset,
                    self.config.get("eval_batch_size", self.config["batch_size"]),
                    shuffle=False,
                )
                self.val_loader = self.get_dataloader(
                    self.val_dataset,
                    self.val_sampler,
                )

            if "test_dataset" in self.config:
                self.test_dataset = registry.get_dataset_class(self.config["dataset"])(self.config, mode="test")
                self.test_sampler = self.get_sampler(
                    self.test_dataset,
                    self.config.get("eval_batch_size", self.config["batch_size"]),
                    shuffle=False,
                )
                self.test_loader = self.get_dataloader(
                    self.test_dataset,
                    self.test_sampler,
                )
        else:
            raise ValueError("Only trajectory_lmdb dataset supported")

        # Normalizer for the dataset.
        # Compute mean, std of training set labels.
        self.normalizers = {}
        if self.config.get("normalize_labels", False):
            if "target_mean" in self.config:
                self.normalizers["target"] = Normalizer(
                    mean=self.config["target_mean"],
                    std=self.config["target_std"],
                    device=self.device,
                )
            else:
                self.normalizers["target"] = Normalizer(
                    tensor=self.train_loader.dataset.data.y[self.train_loader.dataset.__indices__],
                    device=self.device,
                )

        # If we're computing gradients wrt input, set mean of normalizer to 0 --
        # since it is lost when compute dy / dx -- and std to forward target std
        if self.config.get("regress_forces", True):
            if self.config.get("normalize_labels", False):
                if "grad_target_mean" in self.config:
                    self.normalizers["grad_target"] = Normalizer(
                        mean=self.config["grad_target_mean"],
                        std=self.config["grad_target_std"],
                        device=self.device,
                    )
                else:
                    self.normalizers["grad_target"] = Normalizer(
                        tensor=self.train_loader.dataset.data.y[self.train_loader.dataset.__indices__],
                        device=self.device,
                    )
                    self.normalizers["grad_target"].mean.fill_(0)

    def train(self):
        # Configure mlperf logging
        num_instances = int(self.config["instances"])
        instance = int(os.getenv("EXP_ID", "0"))
        if int(os.getenv("THROUGPUT_RUN", "0")):
            prefix = os.getenv("DATESTAMP", "0")
            mlperf_logfile = os.path.join(self.config["cmd"]["results_dir"], f"{prefix}_{instance}.log")
            mllog.config(filename=mlperf_logfile)
        distutils.global_barrier(self.config)
        mllogger = MLLoggerWrapper(PyTCommunicationHandler(), value=None)
        if rank_in_group(self.config, "train_ranks"):
            mllogger.event(key=mllogger.constants.CACHE_CLEAR, value=True)
            mllogger.start(key=mllogger.constants.INIT_START)
            num_nodes = os.getenv("DGXNNODES", max(1, self.config["world_size"] // self.config["nproc_per_node"]))
            mllogger.mlperf_submission_log(benchmark="oc20", num_nodes=num_nodes)
            mllogger.event(key=mllogger.constants.SEED, value=self.config["cmd"]["seed"])
            mllogger.event(key="number_of_ranks", value=distutils.get_world_size())
            mllogger.event(key="number_of_nodes", value=int(os.environ.get("SLURM_NNODES", 1)) // num_instances)
            mllogger.event(key="accelerators_per_node", value=self.config["nproc_per_node"])
            mllogger.event(
                key=mllogger.constants.GLOBAL_BATCH_SIZE, value=self.config["batch_size"] * self.config["gpus"]
            )
            mllogger.event(key=mllogger.constants.TRAIN_SAMPLES, value=self.config["train_size"])
            mllogger.event(key=mllogger.constants.EVAL_SAMPLES, value=self.config["val_size"])
            mllogger.event(key=mllogger.constants.OPT_NAME, value="AdamW")
            mllogger.event(key=mllogger.constants.OPT_BASE_LR, value=self.config["lr_initial"])
            mllogger.event(key=mllogger.constants.OPT_LR_WARMUP_STEPS, value=self.config["warmup_steps"])
            mllogger.event(key=mllogger.constants.OPT_LR_WARMUP_FACTOR, value=self.config["warmup_factor"])
            mllogger.event(key=mllogger.constants.OPT_LR_DECAY_BOUNDARY_STEPS, value=self.config["lr_milestones"])
            mllogger.event(key=mllogger.constants.OPT_LR_DECAY_FACTOR, value=self.config["lr_gamma"])
            mllogger.event(key="staging_start")
            staging_timer = time.time()
        self.load_task()
        if rank_in_group(self.config, "train_ranks"):
            staging_duration = time.time() - staging_timer
            mllogger.event(
                key="staging_stop", sync=False, metadata={"staging_duration": staging_duration, "instance": instance}
            )
            mllogger.event(
                key="tracked_stats",
                sync=False,
                value={"staging_duration": staging_duration},
                metadata={"step": 0, "instance": instance},
            )
            mllogger.log_init_stop_run_start()

        primary_metric = self.config.get("primary_metric", self.evaluator.task_primary_metric[self.name])
        self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        self.metrics = {}
        stop_training = False
        batch_size = self.config["batch_size"] * self.config["gpus"]

        start_epoch, last_epoch = self.start_step // len(self.train_loader), 0
        if distutils.is_master():
            print(
                f"Starting epoch {start_epoch} "
                + f"train batches {len(self.train_loader)} "
                + f"samples {len(self.train_loader.sampler)} "
                + f"valid batches {len(self.val_loader)} "
                + f"samples {len(self.val_loader.sampler)}"
            )

        if self.config["iterations"] > 0:
            iters = 0
            for batch in self.train_loader:
                batch = batch.to(self.device)
                batch.mask = batch.fixed == 0
                num_samples = torch.sum(batch.mask)
                out = self.model(batch)
                force_target = self.normalizers["grad_target"].norm(batch.force)
                loss = get_loss(out, force_target, batch.mask, num_samples, self.loss_c)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                iters += 1
                if iters == self.config["iterations"]:
                    return
        gc.disable()
        for epoch in range(start_epoch, self.config["max_epochs"]):
            last_epoch = epoch
            if rank_in_group(self.config, "train_ranks"):
                mllogger.start(
                    key=mllogger.constants.EPOCH_START,
                    sync=False,
                    metadata={"epoch_num": epoch + 1, "instance": instance},
                )
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)

                self.model.train()
                times, start, start_epoch = [], time.time(), time.time()
                prefetcher = Prefetcher(self.train_loader, self.device)
                batch = prefetcher.next()
                i = 0
                while batch is not None:
                    batch.mask = batch.fixed == 0
                    num_samples = torch.sum(batch.mask)
                    handle = dist.all_reduce(num_samples, group=dist.group.WORLD, async_op=True)
                    current_step = epoch * len(self.train_loader) + (i + 1)
                    # Forward
                    out = self.model(batch)
                    handle.wait()
                    # Loss
                    force_target = self.normalizers["grad_target"].norm(batch.force)
                    loss = get_loss(out, force_target, batch.mask, num_samples, self.loss_c)

                    # Backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    batch = prefetcher.next()
                    i += 1
                    times.append(time.time() - start)
                    start = time.time()
                    if current_step % self.config["cmd"]["print_every"] == 0 and distutils.is_master():
                        log_dict = {
                            "loss": loss.item(),
                            "lr": self.scheduler.get_lr(),
                            "step": current_step,
                            "time": sum(times) / len(times) * 1000,
                        }
                        times = []
                        log_str = ["{}: {:.2e}".format(k, v) for k, v in log_dict.items()]
                        print(", ".join(log_str), flush=True)

                mllogger.event(
                    key="tracked_stats",
                    sync=False,
                    value={
                        "throughput": (batch_size * i * distutils.get_world_size())
                        / ((time.time() - start_epoch) * 1000)
                    },
                    metadata={"step": epoch + 1, "instance": instance},
                )

            if self.config["nodes_for_eval"] > 0:
                stop_training, self.model = distutils.sync_training_and_evaluation(
                    self.config, self.model, stop_training
                )
                if stop_training:
                    break

            if rank_in_group(self.config, "eval_ranks"):
                mllogger.start(
                    key=mllogger.constants.EVAL_START,
                    sync=False,
                    metadata={"epoch_num": epoch + 1, "instance": instance},
                )
                val_metrics = self.validate(
                    split="val",
                    epoch=epoch + 1,
                )
                mllogger.event(
                    key="eval_error",
                    sync=False,
                    value=val_metrics["forces_mae"]["metric"],
                    metadata={"epoch_num": epoch + 1, "instance": instance},
                )
                mllogger.end(
                    key=mllogger.constants.EVAL_STOP,
                    sync=False,
                    metadata={"epoch_num": epoch + 1, "instance": instance},
                )
                if ("mae" in primary_metric and val_metrics[primary_metric]["metric"] < self.best_val_metric) or (
                    val_metrics[primary_metric]["metric"] > self.best_val_metric
                ):
                    self.best_val_metric = val_metrics[primary_metric]["metric"]
                    # self.save(current_epoch, current_step, val_metrics)

                # Check convergence target stopping criteria
                if (
                    "target_forces_mae" in self.config
                    and val_metrics["forces_mae"]["metric"] < self.config["target_forces_mae"]
                ):
                    print("Target quality met. Stopping training")
                    stop_training = True

                mllogger.end(
                    key=mllogger.constants.EPOCH_STOP,
                    sync=False,
                    metadata={"epoch_num": epoch + 1, "instance": instance},
                )

            if self.config["nodes_for_eval"] == 0 and stop_training:
                break

        if rank_in_group(self.config, "train_ranks"):
            mllogger.log_run_stop(status=mllogger.constants.SUCCESS, epoch=last_epoch + 1, instance=instance)
        self.train_dataset.close_db()
        if "val_dataset" in self.config:
            self.val_dataset.close_db()
        if "test_dataset" in self.config:
            self.test_dataset.close_db()
        distutils.global_barrier(self.config)

    def _forward(self, batch):
        out_energy, out_forces = self.model(batch)
        if out_energy.shape[-1] == 1:
            out_energy = out_energy.view(-1)
        out = {"energy": out_energy, "forces": out_forces}
        return out

    def _compute_loss(self, out, batch):
        force_target = self.normalizers["grad_target"].norm(batch.force)
        loss = torch.nn.functional.l1_loss(out[batch.mask], force_target[batch.mask], reduction="sum")
        loss = loss * self.loss_c / batch.num_samples
        return loss

    def _compute_metrics(self, out, batch, evaluator, metrics={}):
        target = {"energy": batch.y, "forces": batch.force, "natoms": batch.natoms}
        mask = batch.fixed == 0
        out["forces"] = out["forces"][mask]
        target["forces"] = target["forces"][mask]
        out["forces"] = self.normalizers["grad_target"].denorm(out["forces"])
        metrics = evaluator.eval(out, target, prev_metrics=metrics)
        return metrics


@torch.jit.script
def get_loss(out, force, mask, num_samples, loss_c):
    loss = torch.nn.functional.l1_loss(out[mask], force[mask], reduction="sum")
    loss = loss * loss_c / num_samples
    return loss
