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

import torch


class CudaTimer:
    def __init__(self, name, warmup, num_steps, exit=True):
        self.name = name
        self.num_steps = num_steps
        self.exit = exit
        self._start_step = warmup
        self._stop_step = warmup + num_steps
        self._cpu_ticks, self._cpu_tocks = [], []
        self._gpu_ticks, self._gpu_tocks = [], []
        self._step = 0
        self._cpu_time_ms, self._gpu_time_ms = 0.0, 0.0

    def tick(self):
        if self._start_step <= self._step < self._stop_step:
            gpu_tick = torch.cuda.Event(enable_timing=True)
            gpu_tick.record()
            cpu_tick = time.perf_counter()
            self._gpu_ticks.append(gpu_tick)
            self._cpu_ticks.append(cpu_tick)

    def tock(self):
        if self._start_step <= self._step < self._stop_step:
            gpu_tock = torch.cuda.Event(enable_timing=True)
            gpu_tock.record()
            cpu_tock = time.perf_counter()
            self._gpu_tocks.append(gpu_tock)
            self._cpu_tocks.append(cpu_tock)

    def add_ticks(self, cpu_ticks, cpu_tocks, gpu_ticks, gpu_tocks):
        if self._start_step <= self._step < self._stop_step:
            assert len(cpu_ticks) == len(cpu_tocks)
            assert len(gpu_ticks) == len(gpu_tocks)
            self._cpu_ticks += cpu_ticks
            self._cpu_tocks += cpu_tocks
            self._gpu_ticks += gpu_ticks
            self._gpu_tocks += gpu_tocks

    def add_elapsed_time(self, cpu_time_ms, gpu_time_ms):
        if self._start_step <= self._step < self._stop_step:
            self._cpu_time_ms += cpu_time_ms
            self._gpu_time_ms += gpu_time_ms

    def step(self):
        if self._step == self._stop_step - 1:
            torch.cuda.synchronize()
            for (gpu_tick, gpu_tock), (cpu_tick, cpu_tock) in zip(
                zip(self._gpu_ticks, self._gpu_tocks),
                zip(self._cpu_ticks, self._cpu_tocks),
            ):
                self._cpu_time_ms += (cpu_tock - cpu_tick) * 1000
                self._gpu_time_ms += gpu_tick.elapsed_time(gpu_tock)
            self._cpu_time_ms /= self.num_steps
            self._gpu_time_ms /= self.num_steps
            print("# " + "-" * 78)
            print(f"{self.name}:")
            print(f"  Elapsed CPU time per iteration: {self._cpu_time_ms:.4f}ms")
            print(f"  Elapsed GPU time per iteration: {self._gpu_time_ms:.4f}ms")
            if self.exit:
                exit(0)
            self._cpu_time_ms, self._gpu_time_ms = 0.0, 0.0
        self._step += 1
