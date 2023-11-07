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

import math
import queue
from typing import Dict, Iterator, List, Tuple

import torch
from torch.utils.data import Sampler

import openfold.distributed as dist
from openfold.datasets import InitialTrainingDataset, ValidationDataset
from openfold.helpers import get_seed_from_string


class InitialTrainingSampler(Sampler[Tuple[int, int]]):
    """Sampler for initial training dataset."""

    def __init__(
        self,
        dataset: InitialTrainingDataset,
        local_batch_size: int,
        global_batch_size: int,
        num_train_iters: int,
        num_prev_iters: int,
        seed: int,
        is_distributed: bool = False,
        dap_size: int = 0,
    ) -> None:
        assert num_prev_iters <= num_train_iters
        if is_distributed:
            assert dist.is_initialized()
            rank = dist.rank()
            num_train_ranks = dist.num_train_ranks()
            assert rank is not None
            assert num_train_ranks is not None
            if dap_size > 0:
                assert num_train_ranks % dap_size == 0
                assert global_batch_size % (num_train_ranks // dap_size) == 0
            else:
                assert global_batch_size % num_train_ranks == 0
        weights = dataset.get_sampler_weights()
        num_samples_in_device_epoch = num_train_iters * local_batch_size
        num_samples_in_global_epoch = num_train_iters * global_batch_size
        # Sample indices:
        index_generator = torch.Generator()
        index_generator.manual_seed(seed)
        random_indices = torch.multinomial(
            input=weights,
            num_samples=num_samples_in_global_epoch,
            replacement=True,
            generator=index_generator,
        )
        # Sample seeds:
        seed_generator = torch.Generator()
        seed_generator.manual_seed(seed)
        random_seeds = torch.randint(
            low=0,
            high=2**63 - 1,
            size=[num_samples_in_global_epoch],
            generator=seed_generator,
        )
        # Create (index, seed) pairs:
        assert random_indices.size() == random_seeds.size()
        indices = random_indices.tolist()
        seeds = random_seeds.tolist()
        assert len(indices) == len(seeds)
        index_seed_pairs = list(zip(indices, seeds))
        if is_distributed:
            if dap_size > 0:
                index_seed_pairs = index_seed_pairs[
                    (rank // dap_size) :: (num_train_ranks // dap_size)
                ]
            else:
                index_seed_pairs = index_seed_pairs[rank::num_train_ranks]
        assert len(index_seed_pairs) == num_samples_in_device_epoch
        # Move forward by skipping previous iterations:
        offset = num_prev_iters * local_batch_size
        assert offset <= len(index_seed_pairs)
        self.index_seed_pairs = index_seed_pairs[offset:]

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        assert hasattr(self, "index_seed_pairs")
        yield from self.index_seed_pairs
        del self.index_seed_pairs

    def __len__(self) -> int:
        assert hasattr(self, "index_seed_pairs")
        return len(self.index_seed_pairs)


class ValidationSampler(Sampler[Tuple[int, int]]):
    """Sampler for validation dataset."""

    def __init__(
        self,
        dataset: ValidationDataset,
        is_distributed: bool = False,
    ) -> None:
        indices = list(range(len(dataset)))
        seeds = [
            get_seed_from_string(mmcif_chain["pdb_chain_id"])
            for mmcif_chain in dataset.mmcif_chains
        ]
        assert len(indices) == len(seeds)
        index_seed_pairs = list(zip(indices, seeds))
        if is_distributed:
            assert dist.is_initialized()
            num_val_ranks = dist.num_val_ranks()
            val_rank_assignments = _assign_samples_to_val_ranks(
                dataset=dataset,
                num_val_ranks=num_val_ranks,
            )
            if dist.is_async_val_enabled():
                rank = dist.rank() - dist.num_train_ranks()
            else:
                rank = dist.rank()
            val_rank_indices = val_rank_assignments[rank]
            index_seed_pairs = [index_seed_pairs[index] for index in val_rank_indices]
        self.index_seed_pairs = index_seed_pairs

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        yield from self.index_seed_pairs

    def __len__(self) -> int:
        return len(self.index_seed_pairs)


def _assign_samples_to_val_ranks(
    dataset: ValidationDataset,
    num_val_ranks: int,
) -> Dict[int, List[int]]:
    val_rank_assignments = {}
    pqueue = queue.PriorityQueue()
    for rank in range(num_val_ranks):
        val_rank_assignments[rank] = []
        pqueue.put((0, rank))
    sequence_lengths = [sample["sequence_length"] for sample in dataset.mmcif_chains]
    enumerated_sequence_lengths = list(enumerate(sequence_lengths))
    sorted_enumerated_sequence_lengths = sorted(
        enumerated_sequence_lengths,
        key=lambda x: (-x[1], x[0]),
    )
    for index, sequence_length in sorted_enumerated_sequence_lengths:
        priority, rank = pqueue.get()
        val_rank_assignments[rank].append(index)
        priority += math.pow(sequence_length, 2)
        pqueue.put((priority, rank))
    return val_rank_assignments
