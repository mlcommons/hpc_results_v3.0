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

import bisect
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
from ocpmodels.common.registry import registry
from torch.utils.data import Dataset


def connect_db(lmdb_path=None):
    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    return env


def get_num_samples(config, mode):
    srcdir = Path(config["data"], config[f"{mode}_dataset"])
    db_paths = sorted(srcdir.glob("*.lmdb"))
    assert len(db_paths) > 0, f"No LMDBs found in {srcdir}"

    _keys, envs = [], []
    for db_path in db_paths:
        envs.append(connect_db(db_path))
        length = pickle.loads(envs[-1].begin().get("length".encode("ascii")))
        _keys.append(list(range(length)))

    keylens = [len(k) for k in _keys]
    num_samples = sum(keylens)
    return num_samples


@registry.register_dataset("trajectory_lmdb")
class TrajectoryLmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation trajectories.
    Useful for Structure to Energy & Force (S2EF) and Initial State to
    Relaxed State (IS2RS) tasks.

    Args:
        config (dict): Dataset configuration
        transform (callable, optional): Data transform function.
            (default: :obj:`None`)
    """

    def __init__(self, config, transform=None, mode="train"):
        super(TrajectoryLmdbDataset, self).__init__()
        self.config = config
        self.mode = mode

        if "src" in config:
            srcdir = Path(config["src"])
        else:
            srcdir = Path(self.config["data_target"], self.config[f"{mode}_dataset"])

        self.metadata_path = srcdir / "metadata.npz"

        db_paths = sorted(srcdir.glob("*.lmdb"))
        assert len(db_paths) > 0, f"No LMDBs found in {srcdir}"

        self._keys, self.envs = [], []
        for db_path in db_paths:
            self.envs.append(self.connect_db(db_path))
            length = pickle.loads(self.envs[-1].begin().get("length".encode("ascii")))
            self._keys.append(list(range(length)))

        keylens = [len(k) for k in self._keys]
        self._keylen_cumulative = np.cumsum(keylens).tolist()
        self.transform = transform
        self.num_samples = sum(keylens)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Figure out which db this should be indexed from.
        db_idx = bisect.bisect(self._keylen_cumulative, idx)
        # Extract index of element within that db.
        el_idx = idx - self._keylen_cumulative[db_idx - 1] if db_idx > 0 else idx

        # Return features.
        datapoint_pickled = self.envs[db_idx].begin().get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
        data_object = pickle.loads(datapoint_pickled)

        # sort edges by source node
        _, idx = data_object.edge_index[0].sort()
        data_object.edge_index = data_object.edge_index[:, idx]

        # DISTANCE
        cell_offsets = data_object.cell_offsets.unsqueeze(1).float()
        cell = data_object.cell.expand(cell_offsets.shape[0], -1, -1)
        data_object.offsets = cell_offsets.bmm(cell).squeeze(1)[idx]

        num_nodes = data_object.pos.shape[0]
        src_counts_full, dst_counts_full = (
            torch.zeros(num_nodes, device="cpu", dtype=torch.long),
            torch.zeros(num_nodes, device="cpu", dtype=torch.long),
        )
        src, dst = data_object.edge_index[0].float(), data_object.edge_index[1].float()
        dst_nodes, dst_counts = torch.unique(dst, return_counts=True)
        dst_nodes = dst_nodes.to(torch.long, non_blocking=True)
        src_nodes, src_counts = torch.unique_consecutive(src, return_counts=True)
        src_nodes = src_nodes.to(torch.long, non_blocking=True)
        src_counts_full[src_nodes], dst_counts_full[dst_nodes] = src_counts, dst_counts
        data_object.src_off = src_counts_full.cumsum(dim=0)
        data_object.dst_off = dst_counts_full.cumsum(dim=0)

        data_object.atomic_numbers = data_object.atomic_numbers.to(torch.int64)

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        return env

    def close_db(self):
        for env in self.envs:
            env.close()
