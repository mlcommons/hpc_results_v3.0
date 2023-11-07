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


import argparse
from pathlib import Path


class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Graph Networks for Electrocatalyst Design")
        self.add_core_args()

    def get_parser(self):
        return self.parser

    def add_core_args(self):
        self.parser.add_argument_group("Core Arguments")
        self.parser.add_argument(
            "--mode",
            choices=["train", "predict", "run-relaxations"],
            default="train",
            help="Whether to train the model, make predictions, or to run relaxations",
        )
        self.parser.add_argument(
            "--identifier",
            default="",
            type=str,
            help="Experiment identifier to append to checkpoint/log/result directory",
        )
        self.parser.add_argument(
            "--debug",
            action="store_true",
            help="Whether this is a debugging run or not",
        )
        self.parser.add_argument(
            "--run-dir",
            default="/results",
            type=str,
            help="Directory to store checkpoint/log/result directory",
        )
        self.parser.add_argument(
            "--print-every",
            default=100,
            type=int,
            help="Log every N iterations",
        )
        self.parser.add_argument("--bucket_cap_mb", type=int, default=25)
        self.parser.add_argument("--seed", default=None, type=int, help="Seed for torch, cuda, numpy")
        self.parser.add_argument("--amp", action="store_true", help="Use mixed-precision training")
        self.parser.add_argument("--checkpoint", type=str, help="Model checkpoint to load")
        self.parser.add_argument("--logdir", default="logs", type=Path, help="Where to store logs")
        self.parser.add_argument("--local_rank", default=0, type=int, help="Local rank")
        self.parser.add_argument("--jobs", type=int, default=12, help="Number of jobs for copying dataset")
        self.parser.add_argument("--instances", type=int, default=1, help="Number of instances for weak scaling")

        self.parser.add_argument(
            "--mlperf_accelerators_per_node", type=int, default=1, help="MLPerf Accelerators per Node"
        )

        self.parser.add_argument("--dataset", type=str, default="trajectory_lmdb")
        self.parser.add_argument(
            "--description", type=str, default="Regressing to energies and forces for DFT trajectories from OCP"
        )
        self.parser.add_argument("--type", type=str, default="regression")
        self.parser.add_argument("--metric", type=str, default="mae")
        self.parser.add_argument("--primary_metric", type=str, default="forces_mae")
        self.parser.add_argument("--target_forces_mae", type=float, default=0.036)
        self.parser.add_argument("--labels", nargs="+", default=["potential energy"])
        self.parser.add_argument("--grad_input", type=str, default="atomic forces")
        self.parser.add_argument("--train_on_free_atoms", action="store_true", default=True)
        self.parser.add_argument("--eval_on_free_atoms", action="store_true", default=True)

        self.parser.add_argument("--name", type=str, default="dimenetplusplus")
        self.parser.add_argument("--hidden_channels", type=int, default=192)
        self.parser.add_argument("--out_emb_channels", type=int, default=192)
        self.parser.add_argument("--num_blocks", type=int, default=3)
        self.parser.add_argument("--cutoff", type=float, default=6.0)
        self.parser.add_argument("--num_radial", type=int, default=6)
        self.parser.add_argument("--num_spherical", type=int, default=7)
        self.parser.add_argument("--num_before_skip", type=int, default=1)
        self.parser.add_argument("--num_after_skip", type=int, default=2)
        self.parser.add_argument("--num_output_layers", type=int, default=3)
        self.parser.add_argument("--regress_forces", action="store_true", default=True)
        self.parser.add_argument("--use_pbc", action="store_true", default=True)
        self.parser.add_argument("--O2", type=int, default=1)

        self.parser.add_argument("--batch_size", type=int, default=4)
        self.parser.add_argument("--prefetch_factor", type=int, default=6)
        self.parser.add_argument("--eval_batch_size", type=int, default=128)
        self.parser.add_argument("--load_balancing", type=str, default="triplets")
        self.parser.add_argument("--nodes_for_eval", type=int, default=0)
        self.parser.add_argument("--num_workers", type=int, default=1)
        self.parser.add_argument("--lr_initial", type=float, default=0.0016)
        self.parser.add_argument("--weight_decay", type=float, default=0.01)
        self.parser.add_argument("--warmup_steps", type=int, default=3908)
        self.parser.add_argument("--warmup_factor", type=float, default=0.2)
        self.parser.add_argument("--lr_gamma", type=float, default=0.1)
        self.parser.add_argument("--lr_milestones", nargs="+", type=int, default=[23448, 31264])
        self.parser.add_argument("--max_epochs", type=int, default=50)
        self.parser.add_argument("--iterations", type=int, default=0)
        self.parser.add_argument("--energy_coefficient", type=int, default=0)
        self.parser.add_argument("--force_coefficient", type=int, default=50)

        self.parser.add_argument("--trainer", type=str, default="mlperf_forces")
        self.parser.add_argument("--logger", type=str, default="tensorboard")

        self.parser.add_argument("--data", type=str, default="/data")
        self.parser.add_argument("--data_target", type=str, default="/dev/shm")
        self.parser.add_argument("--train_dataset", type=str, default="oc20_data/s2ef/2M/train")
        self.parser.add_argument("--val_dataset", type=str, default="oc20_data/s2ef/all/val_id")
        self.parser.add_argument("--target_mean", type=float, default=-0.7554450631141663)
        self.parser.add_argument("--target_std", type=float, default=2.887317180633545)
        self.parser.add_argument("--grad_target_mean", type=float, default=0.0)
        self.parser.add_argument("--grad_target_std", type=float, default=2.887317180633545)
        self.parser.add_argument("--normalize_labels", action="store_true", default=True)


flags = Flags()
