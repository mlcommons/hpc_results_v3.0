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


import glob
import importlib
import json
import os
import time
from bisect import bisect
from itertools import product

import torch


def save_checkpoint(state, checkpoint_dir="checkpoints/"):
    filename = os.path.join(checkpoint_dir, "checkpoint.pt")
    torch.save(state, filename)


def warmup_lr_lambda(current_step, optim_config):
    """Returns a learning rate multiplier.
    Till `warmup_steps`, learning rate linearly increases to `initial_lr`,
    and then gets multiplied by `lr_gamma` every time a milestone is crossed.
    """

    # keep this block for older configs that have warmup_epochs instead of warmup_steps
    # and lr_milestones are defined in epochs
    if any(x < 100 for x in optim_config["lr_milestones"]) or "warmup_epochs" in optim_config:
        raise Exception(
            "ConfigError: please define lr_milestones in steps not epochs and define warmup_steps instead of warmup_epochs"
        )

    if optim_config["warmup_steps"] > 0 and current_step <= optim_config["warmup_steps"]:
        alpha = current_step / float(optim_config["warmup_steps"])
        return optim_config["warmup_factor"] * (1.0 - alpha) + alpha
    idx = bisect(optim_config["lr_milestones"], current_step)
    return pow(optim_config["lr_gamma"], idx)


def conditional_grad(dec):
    "Decorator to enable/disable grad depending on whether force/energy predictions are being made"
    # Adapted from https://stackoverflow.com/questions/60907323/accessing-class-property-as-decorator-argument
    def decorator(func):
        def cls_method(self, *args, **kwargs):
            f = func
            if self.regress_forces:
                f = dec(func)
            return f(self, *args, **kwargs)

        return cls_method

    return decorator


# Override the collation method in `pytorch_geometric.data.InMemoryDataset`
def collate(data_list):
    keys = data_list[0].keys
    data = data_list[0].__class__()

    for key in keys:
        data[key] = []
    slices = {key: [0] for key in keys}

    for item, key in product(data_list, keys):
        data[key].append(item[key])
        if torch.is_tensor(item[key]):
            s = slices[key][-1] + item[key].size(item.__cat_dim__(key, item[key]))
        elif isinstance(item[key], int) or isinstance(item[key], float):
            s = slices[key][-1] + 1
        else:
            raise ValueError("Unsupported attribute type")
        slices[key].append(s)

    if hasattr(data_list[0], "__num_nodes__"):
        data.__num_nodes__ = []
        for item in data_list:
            data.__num_nodes__.append(item.num_nodes)

    for key in keys:
        if torch.is_tensor(data_list[0][key]):
            data[key] = torch.cat(data[key], dim=data.__cat_dim__(key, data_list[0][key]))
        else:
            data[key] = torch.tensor(data[key])
        slices[key] = torch.tensor(slices[key], dtype=torch.long)

    return data, slices


# Copied from https://github.com/facebookresearch/mmf/blob/master/mmf/utils/env.py#L89.
def setup_imports():
    from ocpmodels.common.registry import registry

    # First, check if imports are already setup
    has_already_setup = registry.get("imports_setup", no_warning=True)
    if has_already_setup:
        return
    # Automatically load all of the modules, so that
    # they register with registry
    root_folder = registry.get("ocpmodels_root", no_warning=True)

    if root_folder is None:
        root_folder = os.path.dirname(os.path.abspath(__file__))
        root_folder = os.path.join(root_folder, "..")

    trainer_folder = os.path.join(root_folder, "trainers")
    trainer_pattern = os.path.join(trainer_folder, "**", "*.py")
    datasets_folder = os.path.join(root_folder, "datasets")
    datasets_pattern = os.path.join(datasets_folder, "*.py")
    model_folder = os.path.join(root_folder, "models")
    model_pattern = os.path.join(model_folder, "*.py")

    importlib.import_module("ocpmodels.common.meter")

    files = (
        glob.glob(datasets_pattern, recursive=True)
        + glob.glob(model_pattern, recursive=True)
        + glob.glob(trainer_pattern, recursive=True)
    )

    for f in files:
        for key in ["/trainers", "/datasets", "/models"]:
            if f.find(key) != -1:
                splits = f.split(os.sep)
                file_name = splits[-1]
                module_name = file_name[: file_name.find(".py")]
                importlib.import_module("ocpmodels.%s.%s" % (key[1:], module_name))

    registry.register("imports_setup", True)


def create_config_dict(args):
    overrides = {}
    for arg in args:
        arg = arg.strip("--")
        key, val = arg.split("=")
        overrides[key] = val
    return overrides


def save_experiment_log(args, jobs, configs):
    log_file = args.logdir / "exp" / time.strftime("%Y-%m-%d-%I-%M-%S%p.log")
    log_file.parent.mkdir(exist_ok=True, parents=True)
    with open(log_file, "w") as f:
        for job, config in zip(jobs, configs):
            print(
                json.dumps(
                    {
                        "config": config,
                        "slurm_id": job.job_id,
                        "timestamp": time.strftime("%I:%M:%S%p %Z %b %d, %Y"),
                    }
                ),
                file=f,
            )
    return log_file


class Prefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = self.next_batch.to(device=self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_batch = self.next_batch
        self.preload()
        return next_batch


def label_metric_dict(metric_dict, split):
    new_dict = {}
    for key in metric_dict:
        new_dict["{}_{}".format(split, key)] = metric_dict[key]
    metric_dict = new_dict
    return metric_dict


@torch.no_grad()
@torch.jit.script
def get_edge_offsets(edge_index, num_nodes):
    src_counts_full, dst_counts_full, src_off, dst_off = (
        torch.zeros(num_nodes, device="cuda", dtype=torch.long),
        torch.zeros(num_nodes, device="cuda", dtype=torch.long),
        torch.zeros(num_nodes + 1, device="cuda", dtype=torch.long),
        torch.zeros(num_nodes + 1, device="cuda", dtype=torch.long),
    )

    src, dst = edge_index[0].float(), edge_index[1].float()
    dst_nodes, dst_counts = torch.unique(dst, return_counts=True)
    dst_nodes = dst_nodes.to(torch.long, non_blocking=True)
    _, dst_e_idx = torch.sort(dst, stable=False)
    src_nodes, src_counts = torch.unique_consecutive(src, return_counts=True)
    src_nodes = src_nodes.to(torch.long, non_blocking=True)
    src_counts_full[src_nodes], dst_counts_full[dst_nodes] = src_counts, dst_counts
    src_off[1:], dst_off[1:] = src_counts_full.cumsum(dim=0), dst_counts_full.cumsum(dim=0)
    return src_off, dst_off, dst_e_idx
