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


import os
import time

import torch
from ocpmodels.common import distutils
from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import setup_imports

if __name__ == "__main__":
    torch.set_num_threads(16)
    parser = flags.get_parser()
    args = parser.parse_args()
    config = vars(args)

    if config["nodes_for_eval"]:
        os.environ["NCCL_SHARP_GROUP_SIZE_THRESH"] = str(config["nodes_for_eval"] + 1)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_override_can_fuse_on_gpu(True)
    distutils.setup(config)

    try:
        setup_imports()
        trainer = registry.get_trainer_class(config.get("trainer", "simple"))(
            config,
            run_dir=config.get("run_dir", "/results"),
            print_every=config.get("print_every", 100),
            seed=config.get("seed", 0),
            logger=config.get("logger", "tensorboard"),
            local_rank=config["local_rank"],
            amp=config.get("amp", False),
            cpu=config.get("cpu", False),
        )
        if config["checkpoint"] is not None:
            trainer.load_pretrained(config["checkpoint"])

        start_time = time.time()
        trainer.train()
        if distutils.is_master():
            print("Total time taken = ", time.time() - start_time)
    finally:
        distutils.cleanup()
