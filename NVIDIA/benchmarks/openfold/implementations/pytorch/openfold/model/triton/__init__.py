# Copyright 2023 NVIDIA CORPORATION
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

from openfold.helpers import is_ampere_arch, is_hopper_arch
from apex.contrib.openfold_triton._layer_norm_config_ampere import _auto_tuned_config_ampere
from apex.contrib.openfold_triton._layer_norm_config_hopper import _auto_tuned_config_hopper
from apex.contrib.openfold_triton import _tuneable_triton_kernels


def load_triton_auto_tuned_cache(
    dap_size: int, arch_type: str = None, verbose: bool = True
) -> None:
    if arch_type is None:
        if is_hopper_arch():
            arch_type = "hopper"
        elif is_ampere_arch():
            arch_type = "ampere"
        else:
            arch_type = "hopper"
    if arch_type not in ("hopper", "ampere"):
        raise ValueError(f"Unknown arch type {repr(arch_type)}")
    if verbose:
        print(
            f"Loading auto-tuned Triton configs for DAP={dap_size} and arch_type={arch_type}..."
        )
    auto_tuned_config = {
        "hopper": _auto_tuned_config_hopper,
        "ampere": _auto_tuned_config_ampere,
    }[arch_type]
    config_for_current_dap = auto_tuned_config[dap_size]
    for func_name, cache in config_for_current_dap.items():
        _tuneable_triton_kernels[func_name].cache = cache
    if verbose:
        print("Loaded auto-tuned Triton configs successfully!")
