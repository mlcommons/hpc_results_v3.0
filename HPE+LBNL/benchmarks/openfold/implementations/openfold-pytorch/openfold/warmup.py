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

import time
from typing import Dict

import torch

import openfold.distributed as dist
import openfold.dynamic_axial_parallelism as dap
from openfold.dataloaders import _random_num_recycling_iters_generator
from openfold.helpers import get_seed_from_string
from openfold.torch_utils import map_tensor_tree


def run_training_warmup(
    args,
    device,
    alphafold_config,
    alphafold_fp32,
    alphafold_bf16,
    alphafold_parameters_bf16,
    alphafold_loss,
    optimizer,
    side_stream,
    to_input_dtype,
    num_warmup_iters=20,
    seed=1234567890,
) -> None:
    perf = -time.perf_counter()

    if dist.is_main_process():
        print("training warmup...")

    random_num_recycling_iters_iterator = _random_num_recycling_iters_generator(
        uniform_recycling_iters=list(
            range(0, alphafold_config.num_recycling_iters + 1)
        ),
        seed=seed,
    )

    assert num_warmup_iters >= 20
    for iteration in range(1, num_warmup_iters + 1):
        # Reconfigure cuda stream:
        if alphafold_config.cuda_graphs and iteration == 20:
            torch.cuda.set_stream(torch.cuda.default_stream(device=device))
            torch.cuda.current_stream(device=device).wait_stream(side_stream)

        # Deterministic forward pass during training (dropout etc.):
        if dap.is_enabled():
            forward_seed_string = f"forward_{args.seed}_{dap.group_rank()}_{iteration}"
        elif args.distributed:
            forward_seed_string = f"forward_{args.seed}_{dist.rank()}_{iteration}"
        else:
            forward_seed_string = f"forward_{args.seed}_0_{iteration}"
        torch.manual_seed(get_seed_from_string(forward_seed_string))

        # Next synthetic train batch:
        num_recycling_iters = next(random_num_recycling_iters_iterator)
        synthetic_train_batch = _get_synthetic_train_batch(num_recycling_iters, seed)
        synthetic_train_batch = map_tensor_tree(
            fn=lambda t: t.to(device=device),
            tree=synthetic_train_batch,
        )

        # Forward pass:
        train_inputs = map_tensor_tree(fn=to_input_dtype, tree=synthetic_train_batch)
        if args.precision == "bf16":
            train_outputs = alphafold_bf16(train_inputs)
        else:
            train_outputs = alphafold_fp32(train_inputs)
        loss, losses = alphafold_loss(
            outputs=train_outputs,
            batch=map_tensor_tree(fn=lambda t: t[..., -1], tree=synthetic_train_batch),
        )
        loss = loss / args.gradient_accumulation_iters

        # Backward pass:
        if (iteration - 1) % args.gradient_accumulation_iters == 0:
            if args.precision == "bf16":
                for param_bf16 in alphafold_parameters_bf16:
                    param_bf16.grad = None
            else:
                optimizer.zero_grad()
        loss.backward()

    perf += time.perf_counter()

    if dist.is_main_process():
        print(f"training warmup completed succesfully! ({perf:.3f}s)")


def _get_synthetic_train_batch(
    num_recycling_iters: int,
    seed: int,
    b_s: int = 1,
    N_res: int = 256,
    N_clust: int = 124,
    N_templ: int = 4,
    N_extra_seq: int = 1024,
) -> Dict[str, torch.Tensor]:
    train_batch = {}

    assert num_recycling_iters in {0, 1, 2, 3}

    g = torch.Generator(device=torch.device("cpu"))
    g.manual_seed(seed)

    train_batch["aatype"] = torch.randint(
        20,
        size=(b_s, N_res),
        generator=g,
        dtype=torch.int64,
    )

    train_batch["residue_index"] = torch.arange(
        start=0, end=N_res, dtype=torch.int64
    ).unsqueeze(0)

    train_batch["seq_length"] = torch.tensor([N_res], dtype=torch.int64)

    train_batch["all_atom_positions"] = torch.randn(
        size=(b_s, N_res, 37, 3),
        generator=g,
        dtype=torch.float32,
    )

    train_batch["all_atom_mask"] = torch.randint(
        2,
        size=(b_s, N_res, 37),
        generator=g,
        dtype=torch.float32,
    )

    train_batch["resolution"] = torch.tensor([1.34], dtype=torch.float32)

    train_batch["is_distillation"] = torch.tensor([0.0], dtype=torch.float32)

    train_batch["template_aatype"] = torch.randint(
        20,
        size=(b_s, N_templ, N_res),
        generator=g,
        dtype=torch.int64,
    )

    train_batch["template_all_atom_positions"] = torch.randn(
        size=(b_s, N_templ, N_res, 37, 3),
        generator=g,
        dtype=torch.float32,
    )

    train_batch["template_all_atom_mask"] = torch.randint(
        2,
        size=(b_s, N_templ, N_res, 37),
        generator=g,
        dtype=torch.float32,
    )

    train_batch["template_sum_probs"] = torch.rand(
        size=(b_s, N_templ, 1),
        generator=g,
        dtype=torch.float32,
    ).mul(N_res)

    train_batch["seq_mask"] = torch.ones(
        size=(b_s, N_res),
        dtype=torch.float32,
    )

    train_batch["msa_mask"] = torch.ones(
        size=(b_s, N_clust, N_res),
        dtype=torch.float32,
    )

    train_batch["msa_row_mask"] = torch.ones(
        size=(b_s, N_clust),
        dtype=torch.float32,
    )

    train_batch["template_mask"] = torch.ones(
        size=(b_s, N_templ),
        dtype=torch.float32,
    )

    train_batch["template_pseudo_beta"] = torch.randn(
        size=(b_s, N_templ, N_res, 3),
        generator=g,
        dtype=torch.float32,
    )

    train_batch["template_pseudo_beta_mask"] = torch.ones(
        size=(b_s, N_templ, N_res),
        dtype=torch.float32,
    )

    train_batch["template_torsion_angles_sin_cos"] = (
        torch.rand(
            size=(b_s, N_templ, N_res, 7, 2),
            generator=g,
            dtype=torch.float32,
        )
        .mul(2)
        .add(-1)
    )

    train_batch["template_alt_torsion_angles_sin_cos"] = (
        torch.rand(
            size=(b_s, N_templ, N_res, 7, 2),
            generator=g,
            dtype=torch.float32,
        )
        .mul(2)
        .add(-1)
    )

    train_batch["template_torsion_angles_mask"] = torch.ones(
        size=(b_s, N_templ, N_res, 7),
        dtype=torch.float32,
    )

    train_batch["atom14_atom_exists"] = torch.ones(
        size=(b_s, N_res, 14),
        dtype=torch.float32,
    )

    train_batch["residx_atom14_to_atom37"] = torch.randint(
        37,
        size=(b_s, N_res, 14),
        generator=g,
        dtype=torch.int64,
    )

    train_batch["residx_atom37_to_atom14"] = torch.randint(
        14,
        size=(b_s, N_res, 37),
        generator=g,
        dtype=torch.int64,
    )

    train_batch["atom37_atom_exists"] = (
        torch.randint(
            2,
            size=(b_s, N_res),
            generator=g,
            dtype=torch.float32,
        )
        .unsqueeze(-1)
        .expand(b_s, N_res, 37)
        .contiguous()
    )

    train_batch["atom14_gt_exists"] = (
        torch.randint(
            2,
            size=(b_s, N_res),
            generator=g,
            dtype=torch.float32,
        )
        .unsqueeze(-1)
        .expand(b_s, N_res, 14)
        .contiguous()
    )

    train_batch["atom14_gt_positions"] = torch.randn(
        size=(b_s, N_res, 14, 3),
        generator=g,
        dtype=torch.float32,
    )

    train_batch["atom14_alt_gt_positions"] = torch.randn(
        size=(b_s, N_res, 14, 3),
        generator=g,
        dtype=torch.float32,
    )

    train_batch["atom14_alt_gt_exists"] = (
        torch.randint(
            2,
            size=(b_s, N_res),
            generator=g,
            dtype=torch.float32,
        )
        .unsqueeze(-1)
        .expand(b_s, N_res, 14)
        .contiguous()
    )

    train_batch["atom14_atom_is_ambiguous"] = torch.zeros(
        size=(b_s, N_res, 14),
        dtype=torch.float32,
    )

    train_batch["rigidgroups_gt_frames"] = torch.randn(
        size=(b_s, N_res, 8, 4, 4),
        generator=g,
        dtype=torch.float32,
    )

    train_batch["rigidgroups_gt_exists"] = torch.randint(
        2,
        size=(b_s, N_res, 8),
        generator=g,
        dtype=torch.float32,
    )

    train_batch["rigidgroups_group_exists"] = torch.randint(
        2,
        size=(b_s, N_res, 8),
        generator=g,
        dtype=torch.float32,
    )

    train_batch["rigidgroups_group_is_ambiguous"] = torch.randint(
        2,
        size=(b_s, N_res, 8),
        generator=g,
        dtype=torch.float32,
    )

    train_batch["rigidgroups_alt_gt_frames"] = torch.randn(
        size=(b_s, N_res, 8, 4, 4),
        generator=g,
        dtype=torch.float32,
    )

    train_batch["pseudo_beta"] = torch.randn(
        size=(b_s, N_res, 3),
        generator=g,
        dtype=torch.float32,
    )

    train_batch["pseudo_beta_mask"] = torch.ones(
        size=(b_s, N_res),
        dtype=torch.float32,
    )

    train_batch["backbone_rigid_tensor"] = torch.randn(
        size=(b_s, N_res, 4, 4),
        generator=g,
        dtype=torch.float32,
    )

    train_batch["backbone_rigid_mask"] = torch.ones(
        size=(b_s, N_res),
        dtype=torch.float32,
    )

    train_batch["chi_angles_sin_cos"] = (
        torch.rand(
            size=(b_s, N_res, 4, 2),
            generator=g,
            dtype=torch.float32,
        )
        .mul(2)
        .add(-1)
    )

    train_batch["chi_mask"] = torch.randint(
        2,
        size=(b_s, N_res, 4),
        generator=g,
        dtype=torch.float32,
    )

    train_batch["extra_msa"] = torch.randint(
        22,
        size=(b_s, N_extra_seq, N_res),
        generator=g,
        dtype=torch.int64,
    )

    train_batch["extra_msa_mask"] = torch.ones(
        size=(b_s, N_extra_seq, N_res),
        dtype=torch.float32,
    )

    train_batch["extra_msa_row_mask"] = torch.ones(
        size=(b_s, N_extra_seq),
        dtype=torch.float32,
    )

    train_batch["bert_mask"] = torch.randint(
        2,
        size=(b_s, N_clust, N_res),
        generator=g,
        dtype=torch.float32,
    )

    train_batch["true_msa"] = torch.randint(
        22,
        size=(b_s, N_clust, N_res),
        generator=g,
        dtype=torch.int64,
    )

    train_batch["extra_has_deletion"] = torch.zeros(
        size=(b_s, N_extra_seq, N_res),
        dtype=torch.float32,
    )

    train_batch["extra_deletion_value"] = torch.zeros(
        size=(b_s, N_extra_seq, N_res),
        dtype=torch.float32,
    )

    train_batch["msa_feat"] = torch.rand(
        size=(b_s, N_clust, N_res, 49),
        generator=g,
        dtype=torch.float32,
    )

    train_batch["target_feat"] = torch.randint(
        2,
        size=(b_s, N_res, 22),
        generator=g,
        dtype=torch.float32,
    )

    train_batch["use_clamped_fape"] = torch.tensor([1.0], dtype=torch.float32)

    def _create_recycling_dim(tensor):
        recycling_dim = (num_recycling_iters + 1,)
        output_shape = tensor.shape + recycling_dim
        tensor = tensor.unsqueeze(-1)
        tensor = tensor.expand(output_shape)
        tensor = tensor.contiguous()
        return tensor

    train_batch = {
        key: _create_recycling_dim(tensor) for key, tensor in train_batch.items()
    }

    return train_batch
