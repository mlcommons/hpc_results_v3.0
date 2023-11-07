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

import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch

from openfold.model.alphafold import AlphaFold
from openfold.swa import AlphaFoldSWA

RESUMABLE_CHECKPOINT_FILENAME = "resumable_checkpoint.pt"
SWA_CHECKPOINT_FILENAME = "swa_checkpoint.pt"


def resume_checkpoint(
    alphafold: AlphaFold,
    optimizer: Optional[torch.optim.Optimizer],
    swa_alphafold: Optional[AlphaFoldSWA],
    checkpoint_dirpath: Path,
    device: torch.device,
) -> int:
    # Load the resumable checkpoint:
    resumable_checkpoint_filepath = checkpoint_dirpath / RESUMABLE_CHECKPOINT_FILENAME
    resumable_checkpoint = torch.load(
        resumable_checkpoint_filepath, map_location=device
    )
    iteration = resumable_checkpoint["iteration"]
    alphafold_state_dict = resumable_checkpoint["alphafold_state_dict"]
    alphafold.load_state_dict(alphafold_state_dict, strict=True)
    if optimizer is not None:
        optimizer_state_dict = resumable_checkpoint["optimizer_state_dict"]
        optimizer.load_state_dict(optimizer_state_dict)
    # Load SWA state dict:
    if swa_alphafold is not None and swa_alphafold.enabled:
        swa_checkpoint_filepath = checkpoint_dirpath / SWA_CHECKPOINT_FILENAME
        swa_state_dict = torch.load(swa_checkpoint_filepath, map_location=device)
        swa_alphafold.load_state_dict(swa_state_dict, strict=True)
    return iteration


def resume_from_latest_checkpoint(
    alphafold: AlphaFold,
    optimizer: Optional[torch.optim.Optimizer],
    swa_alphafold: Optional[AlphaFoldSWA],
    checkpoints_dirpath: Path,
    device: torch.device,
    verbose: bool,
) -> int:
    last_checkpoints_dirpath = checkpoints_dirpath / "last"
    last_checkpoint_dirpaths = _get_sorted_last_checkpoint_dirpaths(
        last_checkpoints_dirpath=last_checkpoints_dirpath,
    )
    if len(last_checkpoint_dirpaths) == 0:
        return 0
    last_checkpoint_dirpath = last_checkpoint_dirpaths[0]
    if verbose:
        print(f"Resuming checkpoint from {repr(last_checkpoint_dirpath)}...")
    iteration = resume_checkpoint(
        alphafold=alphafold,
        optimizer=optimizer,
        swa_alphafold=swa_alphafold,
        checkpoint_dirpath=last_checkpoint_dirpath,
        device=device,
    )
    if verbose:
        print(f"Checkpoint resumed from {repr(last_checkpoint_dirpath)} successfully!")
    return iteration


def save_checkpoint(
    alphafold: AlphaFold,
    optimizer: torch.optim.Optimizer,
    swa_alphafold: AlphaFoldSWA,
    iteration: int,
    checkpoint_dirpath: Path,
) -> None:
    checkpoint_dirpath.mkdir(parents=True, exist_ok=True)
    # Save SWA state dict:
    if swa_alphafold.enabled:
        swa_state_dict = swa_alphafold.state_dict()
        swa_checkpoint_filepath = checkpoint_dirpath / SWA_CHECKPOINT_FILENAME
        torch.save(swa_state_dict, swa_checkpoint_filepath)
    # Save the resumable checkpoint:
    if isinstance(alphafold, torch.nn.parallel.DistributedDataParallel):
        alphafold_state_dict = alphafold.module.state_dict()
    else:
        alphafold_state_dict = alphafold.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    resumable_checkpoint = {
        "iteration": iteration,
        "alphafold_state_dict": alphafold_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
    }
    resumable_checkpoint_filepath = checkpoint_dirpath / RESUMABLE_CHECKPOINT_FILENAME
    torch.save(resumable_checkpoint, resumable_checkpoint_filepath)


def save_last_checkpoint(
    alphafold: AlphaFold,
    optimizer: torch.optim.Optimizer,
    swa_alphafold: AlphaFoldSWA,
    iteration: int,
    checkpoints_dirpath: Path,
    keep_last_checkpoints: int,
) -> None:
    if keep_last_checkpoints == 0:
        return
    print("Saving last checkpoint...")
    perf = -time.perf_counter()
    iteration_str = f"{iteration:06}"
    # Save tmp checkpoint:
    tmp_checkpoint_dirpath = checkpoints_dirpath / ".tmplast"
    save_checkpoint(
        alphafold=alphafold,
        optimizer=optimizer,
        swa_alphafold=swa_alphafold,
        iteration=iteration,
        checkpoint_dirpath=tmp_checkpoint_dirpath,
    )
    # Move tmp to last checkpoints:
    last_checkpoint_dirpath = checkpoints_dirpath / "last" / iteration_str
    _move_checkpoint_dirpath(
        source_dirpath=tmp_checkpoint_dirpath,
        target_dirpath=last_checkpoint_dirpath,
        force=True,
    )
    # Delete expendable checkpoints:
    _delete_last_checkpoints(
        last_checkpoints_dirpath=(checkpoints_dirpath / "last"),
        keep_last_checkpoints=keep_last_checkpoints,
    )
    perf += time.perf_counter()
    print(
        f"Last checkpoint saved to {repr(last_checkpoint_dirpath)} successfully! "
        f"({perf:.3f}s)"
    )


def save_val_checkpoint(
    alphafold: Union[AlphaFold, AlphaFoldSWA],
    iteration: int,
    checkpoints_dirpath: Path,
    keep_val_checkpoints: bool,
    val_avg_lddt_ca: float,
) -> None:
    if keep_val_checkpoints == 0:
        return
    print("Saving val checkpoint...")
    perf = -time.perf_counter()
    iteration_str = f"{iteration:06}"
    assert 0.0 <= val_avg_lddt_ca <= 1.0
    val_avg_lddt_ca_str = f"{val_avg_lddt_ca:.6f}".replace(".", "")
    val_checkpoints_dirpath = checkpoints_dirpath / "val"
    val_checkpoint_dirname = f"{val_avg_lddt_ca_str}_{iteration_str}"
    val_checkpoint_dirpath = val_checkpoints_dirpath / val_checkpoint_dirname
    # Check if save val checkpoint based on metric:
    if not _is_val_checkpoint_to_save(
        val_checkpoint_dirpath=val_checkpoint_dirpath,
        keep_val_checkpoints=keep_val_checkpoints,
    ):
        print("Val checkpoint not saved!")
        return
    # Save tmp checkpoint:
    tmp_checkpoint_dirpath = checkpoints_dirpath / ".tmpval"
    tmp_checkpoint_dirpath.mkdir(parents=True, exist_ok=True)
    torch.save(alphafold.state_dict(), tmp_checkpoint_dirpath / "checkpoint.pt")
    # Move tmp to val checkpoints:
    val_checkpoints_dirpath.mkdir(parents=True, exist_ok=True)
    _move_checkpoint_dirpath(
        source_dirpath=tmp_checkpoint_dirpath,
        target_dirpath=val_checkpoint_dirpath,
        force=True,
    )
    # Delete expendable checkpoints:
    _delete_val_checkpoints(
        val_checkpoints_dirpath=(checkpoints_dirpath / "val"),
        keep_val_checkpoints=keep_val_checkpoints,
    )
    perf += time.perf_counter()
    print(
        f"Val checkpoint saved to {repr(val_checkpoint_dirpath)} successfully! "
        f"({perf:.3f}s)"
    )


def _copy_checkpoint_dirpath(
    source_dirpath: Path,
    target_dirpath: Path,
    force: bool,
) -> None:
    assert source_dirpath != target_dirpath
    if target_dirpath.exists() and force:
        shutil.rmtree(target_dirpath)
    assert not target_dirpath.exists()
    shutil.copytree(src=source_dirpath, dst=target_dirpath)


def _move_checkpoint_dirpath(
    source_dirpath: Path,
    target_dirpath: Path,
    force: bool,
) -> None:
    assert source_dirpath != target_dirpath
    if target_dirpath.exists() and force:
        shutil.rmtree(target_dirpath)
    assert not target_dirpath.exists()
    shutil.move(src=source_dirpath, dst=target_dirpath)


def _get_sorted_val_checkpoint_dirpaths(val_checkpoints_dirpath: Path) -> List[Path]:
    assert val_checkpoints_dirpath.name == "val"
    val_checkpoint_dirpaths = list(val_checkpoints_dirpath.glob("[0-9_]*"))
    return sorted(val_checkpoint_dirpaths, reverse=True)


def _get_sorted_last_checkpoint_dirpaths(last_checkpoints_dirpath: Path) -> List[Path]:
    assert last_checkpoints_dirpath.name == "last"
    last_checkpoint_dirpaths = list(last_checkpoints_dirpath.glob("[0-9]*"))
    return sorted(last_checkpoint_dirpaths, reverse=True)


def _delete_val_checkpoints(
    val_checkpoints_dirpath: Path,
    keep_val_checkpoints: int,
) -> None:
    sorted_val_checkpoints = _get_sorted_val_checkpoint_dirpaths(
        val_checkpoints_dirpath=val_checkpoints_dirpath,
    )
    surplus_val_checkpoints = sorted_val_checkpoints[keep_val_checkpoints:]
    for surplus_val_checkpoint in surplus_val_checkpoints:
        shutil.rmtree(surplus_val_checkpoint)


def _delete_last_checkpoints(
    last_checkpoints_dirpath: Path,
    keep_last_checkpoints: int,
) -> None:
    sorted_last_checkpoints = _get_sorted_last_checkpoint_dirpaths(
        last_checkpoints_dirpath=last_checkpoints_dirpath,
    )
    surplus_last_checkpoints = sorted_last_checkpoints[keep_last_checkpoints:]
    for surplus_last_checkpoint in surplus_last_checkpoints:
        shutil.rmtree(surplus_last_checkpoint)


def _is_val_checkpoint_to_save(
    val_checkpoint_dirpath: Path,
    keep_val_checkpoints: int,
) -> bool:
    if keep_val_checkpoints == 0:
        return False
    sorted_val_checkpoints = _get_sorted_val_checkpoint_dirpaths(
        val_checkpoints_dirpath=val_checkpoint_dirpath.parent,
    )
    if keep_val_checkpoints > len(sorted_val_checkpoints):
        return True
    for checkpoint_dirpath in sorted_val_checkpoints[:keep_val_checkpoints]:
        if val_checkpoint_dirpath >= checkpoint_dirpath:
            return True
    return False


def map_init_state_dicts(
    alphafold_state_dict_keys: List[str],
    init_alphafold_state_dict: dict,
    init_optimizer_state_dict: dict,
) -> Tuple[dict, dict]:
    new_alphafold_state_dict = {}
    new_optimizer_state_dict = {}
    new_optimizer_state_dict["state"] = {}
    init_alphafold_state_dict_keys = list(init_alphafold_state_dict.keys())
    for index, key in enumerate(alphafold_state_dict_keys):
        # Rename gate bias:
        if ".mha.linear_g_bias" in key:
            init_key = key.replace("linear_g_bias", "linear_g.bias")
            init_index = init_alphafold_state_dict_keys.index(init_key)
            init_param = init_alphafold_state_dict[init_key]
            init_optim = init_optimizer_state_dict["state"][init_index]
            new_alphafold_state_dict[key] = init_param
            new_optimizer_state_dict["state"][index] = init_optim
        # Fuse MHA linear weights:
        elif ".mha.linear_qkvg.weight" in key:
            init_key_q = key.replace(".linear_qkvg.", ".linear_q.")
            init_key_k = key.replace(".linear_qkvg.", ".linear_k.")
            init_key_v = key.replace(".linear_qkvg.", ".linear_v.")
            init_key_g = key.replace(".linear_qkvg.", ".linear_g.")
            init_index_q = init_alphafold_state_dict_keys.index(init_key_q)
            init_index_k = init_alphafold_state_dict_keys.index(init_key_k)
            init_index_v = init_alphafold_state_dict_keys.index(init_key_v)
            init_index_g = init_alphafold_state_dict_keys.index(init_key_g)
            init_param_q = init_alphafold_state_dict[init_key_q]
            init_param_k = init_alphafold_state_dict[init_key_k]
            init_param_v = init_alphafold_state_dict[init_key_v]
            init_param_g = init_alphafold_state_dict[init_key_g]
            init_optim_q = init_optimizer_state_dict["state"][init_index_q]
            init_optim_k = init_optimizer_state_dict["state"][init_index_k]
            init_optim_v = init_optimizer_state_dict["state"][init_index_v]
            init_optim_g = init_optimizer_state_dict["state"][init_index_g]
            init_param = torch.cat(
                [
                    init_param_q,
                    init_param_k,
                    init_param_v,
                    init_param_g,
                ],
                dim=0,
            )
            init_optim = {}
            init_optim["step"] = init_optim_q["step"]
            init_optim["exp_avg"] = torch.cat(
                [
                    init_optim_q["exp_avg"],
                    init_optim_k["exp_avg"],
                    init_optim_v["exp_avg"],
                    init_optim_g["exp_avg"],
                ],
                dim=0,
            )
            init_optim["exp_avg_sq"] = torch.cat(
                [
                    init_optim_q["exp_avg_sq"],
                    init_optim_k["exp_avg_sq"],
                    init_optim_v["exp_avg_sq"],
                    init_optim_g["exp_avg_sq"],
                ],
                dim=0,
            )
            new_alphafold_state_dict[key] = init_param
            new_optimizer_state_dict["state"][index] = init_optim
        elif ".mha.linear_kv.weight" in key:
            init_key_k = key.replace(".linear_kv.", ".linear_k.")
            init_key_v = key.replace(".linear_kv.", ".linear_v.")
            init_index_k = init_alphafold_state_dict_keys.index(init_key_k)
            init_index_v = init_alphafold_state_dict_keys.index(init_key_v)
            init_param_k = init_alphafold_state_dict[init_key_k]
            init_param_v = init_alphafold_state_dict[init_key_v]
            init_optim_k = init_optimizer_state_dict["state"][init_index_k]
            init_optim_v = init_optimizer_state_dict["state"][init_index_v]
            init_param = torch.cat(
                [
                    init_param_k,
                    init_param_v,
                ],
                dim=0,
            )
            init_optim = {}
            init_optim["step"] = init_optim_k["step"]
            init_optim["exp_avg"] = torch.cat(
                [
                    init_optim_k["exp_avg"],
                    init_optim_v["exp_avg"],
                ],
                dim=0,
            )
            init_optim["exp_avg_sq"] = torch.cat(
                [
                    init_optim_k["exp_avg_sq"],
                    init_optim_v["exp_avg_sq"],
                ],
                dim=0,
            )
            new_alphafold_state_dict[key] = init_param
            new_optimizer_state_dict["state"][index] = init_optim
        # Identity:
        else:
            init_key = key
            init_index = init_alphafold_state_dict_keys.index(init_key)
            init_param = init_alphafold_state_dict[init_key]
            init_optim = init_optimizer_state_dict["state"][init_index]
            new_alphafold_state_dict[key] = init_param
            new_optimizer_state_dict["state"][index] = init_optim

    assert len(new_alphafold_state_dict) == len(new_optimizer_state_dict["state"])
    new_param_groups = deepcopy(init_optimizer_state_dict["param_groups"])
    new_param_groups[0]["params"] = list(range(len(new_alphafold_state_dict)))
    new_optimizer_state_dict["param_groups"] = new_param_groups

    for index, key in enumerate(new_alphafold_state_dict.keys()):
        assert (
            new_alphafold_state_dict[key].size()
            == new_optimizer_state_dict["state"][index]["exp_avg"].size()
        )
        assert (
            new_alphafold_state_dict[key].size()
            == new_optimizer_state_dict["state"][index]["exp_avg_sq"].size()
        )

    new_init_state_dicts = (
        new_alphafold_state_dict,
        new_optimizer_state_dict,
    )

    return new_init_state_dicts
