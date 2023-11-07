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

import queue
import random
import threading
from copy import deepcopy
from multiprocessing.managers import SyncManager
from typing import Iterable, Iterator, List

import torch
from torch.utils.data import DataLoader

import openfold.dynamic_axial_parallelism as dap
from openfold.datasets import InitialTrainingDataset, ValidationDataset
from openfold.samplers import InitialTrainingSampler, ValidationSampler
from openfold.torch_utils import collate, map_tensor_tree


class InitialTrainingDataloaderPT(DataLoader):
    """Dataloader for the initial training stage."""

    def __init__(
        self,
        dataset: InitialTrainingDataset,
        sampler: InitialTrainingSampler,
        local_batch_size: int,
        num_workers: int,
        prefetch_factor: int,
        seed: int,
        uniform_recycling_iters: List[int],
        gradient_accumulation_iters: int,
        num_prev_iters: int,
        use_threading: bool,
    ) -> None:
        super(InitialTrainingDataloaderPT, self).__init__(
            dataset=dataset,
            collate_fn=collate,
            sampler=sampler,
            batch_size=local_batch_size,
            num_workers=num_workers,
            drop_last=True,
            prefetch_factor=(prefetch_factor if num_workers > 0 else None),
            persistent_workers=bool(num_workers > 0),
        )
        self._set_train_batch_properties_fn = TrainBatchProperties(
            seed=seed,
            uniform_recycling_iters=uniform_recycling_iters,
            gradient_accumulation_iters=gradient_accumulation_iters,
            num_prev_iters=num_prev_iters,
        )
        if use_threading:
            print(f"threading is not supported in {InitialTrainingDataloaderPT}")

    def __iter__(self) -> Iterator[dict]:
        iterator = super().__iter__()
        for batch in iterator:
            yield self._set_train_batch_properties_fn(batch)


class InitialTrainingDataloaderPQ:
    """Dataloader for the initial training stage with non-blocking priority queue."""

    def __init__(
        self,
        dataset: InitialTrainingDataset,
        sampler: InitialTrainingSampler,
        local_batch_size: int,
        num_workers: int,
        prefetch_factor: int,
        seed: int,
        uniform_recycling_iters: List[int],
        gradient_accumulation_iters: int,
        num_prev_iters: int,
        use_threading: bool,
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.local_batch_size = local_batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self._set_train_batch_properties_fn = TrainBatchProperties(
            seed=seed,
            uniform_recycling_iters=uniform_recycling_iters,
            gradient_accumulation_iters=gradient_accumulation_iters,
            num_prev_iters=num_prev_iters,
        )
        self.queue_maxsize = self.num_workers * self.prefetch_factor
        self.producer_queue_maxsize = 2
        self.consumer_queue_maxsize = 2
        if torch.distributed.is_initialized():
            self.dist_rank = int(torch.distributed.get_rank())
            self.dist_world_size = int(torch.distributed.get_world_size())
        else:
            self.dist_rank = None
            self.dist_world_size = None
        if dap.is_initialized():
            assert self.dist_rank == dap.group_rank() * dap.size() + dap.rank()
            self.dist_group_size = dap.size()
            self.dist_rank_in_group = dap.rank()
            self.dist_producer_rank = (
                self.dist_rank - self.dist_rank % self.dist_group_size
            )
        else:
            self.dist_group_size = None
            self.dist_rank_in_group = None
            self.dist_producer_rank = None
        self.dist_group = None
        self.threading_enabled = use_threading

    def _start_manager(self) -> None:
        self._manager = ManagerPQ()
        self._manager.start()

    def _start_multiprocessing(self) -> None:
        # create queues:
        self._index_queue = torch.multiprocessing.Queue(maxsize=self.queue_maxsize)
        self._sample_pqueue = self._manager.PriorityQueue(maxsize=self.queue_maxsize)
        self._batch_pqueue = self._manager.PriorityQueue(maxsize=self.queue_maxsize)

        # create index process:
        self._index_process = torch.multiprocessing.Process(
            target=_index_worker,
            args=(
                self.sampler,
                self._index_queue,
            ),
        )
        self._index_process.daemon = True
        self._index_process.start()

        # create sample processes:
        self._sample_processes = []
        for w in range(self.num_workers):
            sample_process = torch.multiprocessing.Process(
                target=_sample_worker,
                args=(
                    self.dataset,
                    self._index_queue,
                    self._sample_pqueue,
                ),
            )
            sample_process.daemon = True
            sample_process.start()
            self._sample_processes.append(sample_process)

        # create batch process:
        self._batch_process = torch.multiprocessing.Process(
            target=_batch_worker,
            args=(
                self._sample_pqueue,
                self._batch_pqueue,
                self.local_batch_size,
            ),
        )
        self._batch_process.daemon = True
        self._batch_process.start()

    def _start_producer(self) -> None:
        self._producer_queue = self._manager.Queue(maxsize=self.producer_queue_maxsize)
        self._producer_process = torch.multiprocessing.Process(
            target=_producer_worker,
            args=(
                self._batch_pqueue,
                self._producer_queue,
                self.dist_rank,
                self.dist_world_size,
                self.dist_group_size,
                self.dist_rank_in_group,
                self.dist_producer_rank,
            ),
        )
        self._producer_process.daemon = True
        self._producer_process.start()

    def _start_consumer(self) -> None:
        self._consumer_queue = self._manager.Queue(maxsize=self.consumer_queue_maxsize)
        self._consumer_process = torch.multiprocessing.Process(
            target=_consumer_worker,
            args=(
                self._consumer_queue,
                self.dist_rank,
                self.dist_world_size,
                self.dist_group_size,
                self.dist_rank_in_group,
                self.dist_producer_rank,
            ),
        )
        self._consumer_process.daemon = True
        self._consumer_process.start()

    def _start_threading(self, input_queue) -> None:
        if self.threading_enabled:
            self._batch_tqueue = queue.Queue(maxsize=1)
            self._batch_thread = threading.Thread(
                target=_batch_thread,
                args=(
                    input_queue,
                    self._batch_tqueue,
                ),
            )
            self._batch_thread.daemon = True
            self._batch_thread.start()

    def _close_manager(self) -> None:
        if hasattr(self, "_manager"):
            del self._manager

    def _close_multiprocessing(self) -> None:
        if hasattr(self, "_batch_process"):
            self._batch_process.terminate()
            del self._batch_process

        if hasattr(self, "_sample_processes"):
            for sample_process in reversed(self._sample_processes):
                sample_process.terminate()
            del self._sample_processes

        if hasattr(self, "_index_process"):
            self._index_process.terminate()
            del self._index_process

        if hasattr(self, "_batch_pqueue"):
            del self._batch_pqueue

        if hasattr(self, "_sample_pqueue"):
            del self._sample_pqueue

        if hasattr(self, "_index_queue"):
            del self._index_queue

    def _close_producer(self) -> None:
        if hasattr(self, "_producer_process"):
            self._producer_process.terminate()
            del self._producer_process

        if hasattr(self, "_producer_queue"):
            del self._producer_queue

    def _close_consumer(self) -> None:
        if hasattr(self, "_consumer_process"):
            self._consumer_process.terminate()
            del self._consumer_process

        if hasattr(self, "_consumer_queue"):
            del self._consumer_queue

    def _close_threading(self) -> None:
        if hasattr(self, "_batch_thread"):
            del self._batch_thread

        if hasattr(self, "_batch_tqueue"):
            del self._batch_tqueue

    def _multiprocessing_iter(self) -> Iterator[dict]:
        self._start_manager()
        self._start_multiprocessing()
        self._start_threading(self._batch_pqueue)
        sampler_length = len(self.sampler)
        num_dataloader_iters = sampler_length // self.local_batch_size
        for i in range(num_dataloader_iters):
            if self.threading_enabled:
                priority, batch = self._batch_tqueue.get()
            else:
                priority, batch = self._batch_pqueue.get()
            batch["__priority__"] = priority
            yield batch
        self._close_threading()
        self._close_multiprocessing()
        self._close_manager()

    def _synchronous_iter(self) -> Iterator[dict]:
        sampler_iterator = iter(self.sampler)
        sampler_length = len(self.sampler)
        num_dataloader_iters = sampler_length // self.local_batch_size
        for i in range(num_dataloader_iters):
            samples = []
            for j in range(self.local_batch_size):
                index = next(sampler_iterator)
                sample = self.dataset[index]
                assert isinstance(sample, dict)
                samples.append(sample)
            batch = collate(samples)
            batch["__priority__"] = i
            yield batch

    def _multiprocessing_iter_producer(self) -> Iterator[dict]:
        self._start_manager()
        self._start_multiprocessing()
        self._start_producer()
        self._start_threading(self._producer_queue)
        sampler_length = len(self.sampler)
        num_dataloader_iters = sampler_length // self.local_batch_size
        for i in range(num_dataloader_iters):
            if self.threading_enabled:
                batch = self._batch_tqueue.get()
            else:
                batch = self._producer_queue.get()
            yield batch
        self._close_threading()
        self._close_producer()
        self._close_multiprocessing()
        self._close_manager()

    def _multiprocessing_iter_consumer(self) -> Iterator[dict]:
        self._start_manager()
        self._start_consumer()
        self._start_threading(self._consumer_queue)
        sampler_length = len(self.sampler)
        num_dataloader_iters = sampler_length // self.local_batch_size
        for i in range(num_dataloader_iters):
            if self.threading_enabled:
                batch = self._batch_tqueue.get()
            else:
                batch = self._consumer_queue.get()
            yield batch
        self._close_threading()
        self._close_consumer()
        self._close_manager()

    def _synchronous_iter_producer(self) -> Iterator[dict]:
        sampler_iterator = iter(self.sampler)
        sampler_length = len(self.sampler)
        num_dataloader_iters = sampler_length // self.local_batch_size
        for i in range(num_dataloader_iters):
            samples = []
            for j in range(self.local_batch_size):
                index = next(sampler_iterator)
                sample = self.dataset[index]
                assert isinstance(sample, dict)
                samples.append(sample)
            batch = collate(samples)
            batch["__priority__"] = i
            object_list = [batch]
            torch.distributed.broadcast_object_list(
                object_list=object_list,
                src=self.dist_producer_rank,
                group=self.dist_group,
            )
            assert isinstance(batch, dict)
            yield batch

    def _synchronous_iter_consumer(self) -> Iterator[dict]:
        sampler_length = len(self.sampler)
        num_dataloader_iters = sampler_length // self.local_batch_size
        for i in range(num_dataloader_iters):
            object_list = [None]
            torch.distributed.broadcast_object_list(
                object_list=object_list,
                src=self.dist_producer_rank,
                group=self.dist_group,
            )
            batch = object_list[0]
            assert isinstance(batch, dict)
            yield batch

    def _get_multiprocessing_iterator(self) -> Iterable[dict]:
        if not self.dist_group_size:
            return self._multiprocessing_iter()
        if self.dist_rank_in_group == 0:
            return self._multiprocessing_iter_producer()
        else:
            return self._multiprocessing_iter_consumer()

    def _get_synchronous_iterator(self) -> Iterable[dict]:
        if not self.dist_group_size:
            return self._synchronous_iter()
        assert self.dist_group is None
        self.dist_group = _init_dist_groups(
            dist_rank=self.dist_rank,
            dist_world_size=self.dist_world_size,
            dist_group_size=self.dist_group_size,
            dist_rank_in_group=self.dist_rank_in_group,
            dist_producer_rank=self.dist_producer_rank,
            backend="nccl",
        )
        assert self.dist_group is not None
        if self.dist_rank_in_group == 0:
            return self._synchronous_iter_producer()
        else:
            return self._synchronous_iter_consumer()

    def __iter__(self) -> Iterator[dict]:
        if self.num_workers > 0:
            iterator = self._get_multiprocessing_iterator()
        elif self.num_workers == 0:
            iterator = self._get_synchronous_iterator()
        for i, batch in enumerate(iterator):
            yield self._set_train_batch_properties_fn(batch)

    def __del__(self) -> None:
        self._close_threading()
        self._close_consumer()
        self._close_producer()
        self._close_multiprocessing()
        self._close_manager()


class ValidationDataloader(DataLoader):
    """Validation dataloader."""

    def __init__(
        self,
        dataset: ValidationDataset,
        sampler: ValidationSampler,
        num_workers: int,
        use_cache: bool,
    ) -> None:
        super(ValidationDataloader, self).__init__(
            dataset=dataset,
            collate_fn=collate,
            sampler=sampler,
            batch_size=1,
            num_workers=num_workers,
            drop_last=True,
            prefetch_factor=(4 if num_workers > 0 else None),
            persistent_workers=True,
            multiprocessing_context=torch.multiprocessing.get_context("spawn"),
        )
        self._is_cache_enabled = use_cache
        self._is_cache_filled = False
        self._cache = {}

    def __iter__(self) -> Iterator[dict]:
        if not self._is_cache_enabled:
            # run without cache:
            iterator = super().__iter__()
            for batch in iterator:
                yield batch

        elif self._is_cache_enabled and not self._is_cache_filled:
            # fill cache:
            self._is_cache_filled = True
            iterator = super().__iter__()
            for i, batch in enumerate(iterator):
                self._cache[i] = _batch_deepcopy(batch)
                yield batch

        elif self._is_cache_enabled and self._is_cache_filled:
            # yield from cache:
            for i, batch in self._cache.items():
                yield _batch_deepcopy(batch)


class TrainBatchProperties:
    """Assigns randomized global train batch properties."""

    def __init__(
        self,
        seed: int,
        uniform_recycling_iters: List[int],
        gradient_accumulation_iters: int,
        num_prev_iters: int,
    ) -> None:
        self._random_num_recycling_iters_iterator = (
            _random_num_recycling_iters_generator(
                uniform_recycling_iters=uniform_recycling_iters,
                seed=seed,
            )
        )
        assert gradient_accumulation_iters >= 1
        self._gradient_accumulation_iters = gradient_accumulation_iters
        assert num_prev_iters >= 0
        self._iteration = num_prev_iters
        self._num_recycling_iters = None
        # restore rng state by iterating through previous iterations:
        assert num_prev_iters % gradient_accumulation_iters == 0
        for _ in range(num_prev_iters // gradient_accumulation_iters):
            next(self._random_num_recycling_iters_iterator)

    def __call__(self, batch: dict) -> dict:
        self._iteration += 1
        if (self._iteration - 1) % self._gradient_accumulation_iters == 0:
            self._num_recycling_iters = next(self._random_num_recycling_iters_iterator)
        assert self._num_recycling_iters is not None
        batch = map_tensor_tree(
            fn=lambda t: t[..., : self._num_recycling_iters + 1],
            tree=batch,
        )
        return batch


class ManagerPQ(SyncManager):
    pass


ManagerPQ.register("PriorityQueue", queue.PriorityQueue)


def _random_num_recycling_iters_generator(
    uniform_recycling_iters: List[int],
    seed: int,
) -> Iterator[int]:
    assert isinstance(uniform_recycling_iters, list)
    assert len(uniform_recycling_iters) > 0
    rng = random.Random(seed)
    while True:
        num_recycling_iters_values = uniform_recycling_iters.copy()
        rng.shuffle(num_recycling_iters_values)
        for num_recycling_iters in num_recycling_iters_values:
            yield num_recycling_iters


def _index_worker(sampler, index_queue) -> None:
    for priority, index in enumerate(sampler):
        index_queue.put((priority, index))


def _sample_worker(dataset, index_queue, sample_pqueue) -> None:
    while True:
        priority, index = index_queue.get()
        sample = dataset[index]
        sample_pqueue.put((priority, sample))


def _batch_worker(sample_pqueue, batch_pqueue, local_batch_size: int) -> None:
    while True:
        samples = []
        priorities = []
        for j in range(local_batch_size):
            priority, sample = sample_pqueue.get()
            samples.append(sample)
            priorities.append(priority)
        batch = collate(samples)
        priority = min(priorities)
        batch_pqueue.put((priority, batch))


def _producer_worker(
    batch_pqueue,
    producer_queue,
    dist_rank: int,
    dist_world_size: int,
    dist_group_size: int,
    dist_rank_in_group: int,
    dist_producer_rank: int,
) -> None:
    dist_group = _init_dist_groups(
        dist_rank=dist_rank,
        dist_world_size=dist_world_size,
        dist_group_size=dist_group_size,
        dist_rank_in_group=dist_rank_in_group,
        dist_producer_rank=dist_producer_rank,
        backend="gloo",
    )
    while True:
        priority, batch = batch_pqueue.get()
        batch["__priority__"] = priority
        object_list = [batch]
        torch.distributed.broadcast_object_list(
            object_list=object_list,
            src=dist_producer_rank,
            group=dist_group,
        )
        producer_queue.put(batch)


def _consumer_worker(
    consumer_queue,
    dist_rank: int,
    dist_world_size: int,
    dist_group_size: int,
    dist_rank_in_group: int,
    dist_producer_rank: int,
) -> None:
    dist_group = _init_dist_groups(
        dist_rank=dist_rank,
        dist_world_size=dist_world_size,
        dist_group_size=dist_group_size,
        dist_rank_in_group=dist_rank_in_group,
        dist_producer_rank=dist_producer_rank,
        backend="gloo",
    )
    while True:
        object_list = [None]
        torch.distributed.broadcast_object_list(
            object_list=object_list,
            src=dist_producer_rank,
            group=dist_group,
        )
        batch = object_list[0]
        consumer_queue.put(batch)


def _batch_thread(input_queue, batch_tqueue) -> None:
    while True:
        obj = input_queue.get()
        batch_tqueue.put(obj)


def _init_dist_groups(
    dist_rank: int,
    dist_world_size: int,
    dist_group_size: int,
    dist_rank_in_group: int,
    dist_producer_rank: int,
    backend: str,
) -> torch.distributed.ProcessGroup:
    dist_group = None
    assert dist_world_size % dist_group_size == 0
    assert dist_rank_in_group == dist_rank % dist_group_size
    assert dist_producer_rank == dist_rank - dist_rank % dist_group_size
    num_dist_groups = dist_world_size // dist_group_size
    for dist_group_rank in range(num_dist_groups):
        dist_ranks_forming_group = list(
            range(
                dist_group_rank * dist_group_size,
                (dist_group_rank + 1) * dist_group_size,
            )
        )
        new_dist_group = torch.distributed.new_group(
            ranks=dist_ranks_forming_group,
            backend=backend,
        )
        if dist_rank in dist_ranks_forming_group:
            assert dist_group is None
            dist_group = new_dist_group
        del new_dist_group
    assert dist_group is not None
    assert isinstance(dist_group, torch.distributed.ProcessGroup)
    return dist_group


def _batch_deepcopy(batch: dict) -> dict:
    output = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.clone()
        else:
            output[key] = deepcopy(value)
    return output
