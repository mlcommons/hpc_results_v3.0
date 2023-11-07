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

import os
import time

_ENABLED = bool(int(os.environ.get("MLPERF_POWER_TRAIN_AFTER_RUN_STOP", "0")))

_START = None

_CONVERGED = False

_REQUIRED_MINUTES = 10


def is_enabled() -> bool:
    return _ENABLED


def start() -> None:
    global _START
    if not _ENABLED:
        raise RuntimeError("power measurement is not enabled")
    if _START is not None:
        raise RuntimeError("power measurement has already started")
    _START = time.perf_counter()


def mark_as_converged() -> bool:
    global _CONVERGED
    log_run_stop = False
    if not _CONVERGED:
        log_run_stop = True
    _CONVERGED = True
    return log_run_stop


def is_completed(verbose: bool = False) -> bool:
    if not _ENABLED:
        raise RuntimeError("power measurement is not enabled")
    if _START is None:
        raise RuntimeError("power measurement has not started")
    now = time.perf_counter()
    elapsed_seconds = now - _START
    elapsed_minutes = elapsed_seconds / 60
    if verbose:
        print(f"power measurement elapsed minutes {elapsed_minutes:.3f}")
    if elapsed_minutes > _REQUIRED_MINUTES:
        return True and _CONVERGED
    else:
        return False
