# The MIT License (MIT)
#
# Modifications Copyright (c) 2020-2023 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import sys
import glob
import h5py as h5
import numpy as np
import torch
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from .common import get_datashapes


class CamDaliNumpyFusedDataloader(object):

    def get_pipeline(self, datalabel_files, num_shards, shard_id):
        self.data_size = 1152*768*16*4
        self.label_size = 1152*768*4
        
        pipeline = Pipeline(batch_size=self.batchsize, 
                            num_threads=self.num_threads, 
                            device_id=self.device.index,
                            seed = self.seed)
                                 
        with pipeline:
            datalabel = fn.readers.numpy(name="datalabel",
                                    device = self.io_device,
                                    file_root = self.root_dir,
                                    files = [os.path.basename(x) for x in datalabel_files],
                                    num_shards = num_shards,
                                    shard_id = shard_id,
                                    stick_to_shard = self.stick_to_shard,
                                    shuffle_after_epoch = self.shuffle,
                                    prefetch_queue_depth = 2,
                                    cache_header_information = True,
                                    register_buffers = True,
                                    pad_last_batch = True,
                                    dont_use_mmap = not self.use_mmap,
                                    lazy_init = self.lazy_init,
                                    bytes_per_sample_hint = (self.data_size + self.label_size),
                                    seed = self.seed).gpu()

            # slice the stuff
            data, label = datalabel[:, :, :-1], datalabel[:, :, -1]
                                       
            # normalize data
            data = fn.normalize(data, 
                                device = "gpu",
                                mean = self.data_mean,
                                stddev = self.data_stddev,
                                scale = 1.,
                                bytes_per_sample_hint = self.data_size)
            
            # cast label to long
            label = fn.cast(label,
                            device = "gpu",
                            dtype = types.DALIDataType.INT64,
                            bytes_per_sample_hint = self.label_size)
                            
            if self.transpose:
                data = fn.transpose(data,
                                    device = "gpu",
                                    perm = [2, 0, 1],
                                    bytes_per_sample_hint = self.data_size)

            pipeline.set_outputs(data, label)
            
        return pipeline
            


    def init_files(self, root_dir, prefix_datalabel, statsfile, file_list_datalabel = None):
        self.root_dir = root_dir
        self.prefix_datalabel = prefix_datalabel

        # get files
        if file_list_datalabel is not None and os.path.isfile(os.path.join(root_dir, file_list_datalabel)):
            with open(os.path.join(root_dir, file_list_datalabel), "r") as f:
                token = f.readlines()
            self.datalabel_files = sorted([os.path.join(root_dir, x.strip()) for x in token])
        else:
            self.datalabel_files = sorted(glob.glob(os.path.join(self.root_dir, self.prefix_datalabel)))

        # get shapes
        self.data_shape, self.label_shape = get_datashapes()
        #self.data_shape = np.load(self.data_files[0]).shape
        #self.label_shape = np.load(self.label_files[0]).shape

        # open statsfile
        with h5.File(statsfile, "r") as f:
            data_mean = f["climate"]["minval"][...]
            data_stddev = (f["climate"]["maxval"][...] - data_mean)
            
        #reshape into broadcastable shape: channels first
        self.data_mean = np.reshape( data_mean, (1, 1, data_mean.shape[0]) ).astype(np.float32)
        self.data_stddev = np.reshape( data_stddev, (1, 1, data_stddev.shape[0]) ).astype(np.float32)

        # clean up old iterator
        if self.iterator is not None:
            del(self.iterator)
            self.iterator = None
        
        # clean up old pipeline
        if self.pipeline is not None:
            del(self.pipeline)
            self.pipeline = None

        # io devices
        self.io_device = "gpu" if self.read_gpu else "cpu"

        # restrict file list depending on shuffle mode:
        datalabel_files = self.datalabel_files
        num_shards = self.num_shards
        shard_id = self.shard_id

        # modify sharding for gpu-local and node-local shuffling
        if (self.shuffle_mode == "gpu") or (self.shuffle_mode == "node"):

            # modifier
            if self.shuffle_mode == "node":
                num_local_ranks = torch.cuda.device_count()
                num_shards = num_shards // num_local_ranks
                local_shard_id = shard_id % num_local_ranks
                shard_id = shard_id // num_local_ranks
            
            # shard the bulk first
            num_files = len(self.data_files)
            num_files_per_shard = num_files // num_shards
            shard_start = shard_id * num_files_per_shard
            shard_end = shard_start + num_files_per_shard
            
            # get the remainder now
            rem_start = num_shards * num_files_per_shard
            rem_end = num_files

            # extract file lists
            # remainder
            datalabel_rem = datalabel_files[rem_start:rem_end]

            # get the bulk
            datalabel_files = datalabel_files[shard_start:shard_end]
            
            # append remainder
            if shard_id < len(datalabel_rem):
                datalabel_files.append(datalabel_rem[shard_id])

        # reset shard ids 
        if self.shuffle_mode == "gpu":
            num_shards = 1
            shard_id = 0
        elif self.shuffle_mode == "node":
            num_shards = num_local_ranks
            shard_id = local_shard_id

        # DEBUG
        #print(f"shard {self.shard_id}->{shard_id} {self.num_shards}->{num_shards} {len(self.data_files)}->{len(data_files)}, {len(self.label_files)}->{len(label_files)}", flush=True)
        #print(data_files, label_files, flush=True)
        # DEBUG
        
        # set up pipeline
        self.pipeline = self.get_pipeline(datalabel_files, num_shards, shard_id)
       
        # build pipes
        self.global_size = len(self.datalabel_files)
        self.pipeline.build()

        # init iterator
        if not self.lazy_init:
            self.init_iterator()

        
    def __init__(self, root_dir, prefix_datalabel, statsfile,
                 batchsize, file_list_datalabel = None, 
                 num_threads = 1, device = torch.device("cpu"),
                 num_shards = 1, shard_id = 0,
                 shuffle_mode = None, oversampling_factor = 1,
                 is_validation = False,
                 lazy_init = False, transpose = True, augmentations = [],
                 use_mmap = True, read_gpu = False, seed = 333):
    
        # read filenames first
        self.batchsize = batchsize
        self.num_threads = num_threads
        self.device = device
        self.io_device = "gpu" if read_gpu else "cpu"
        self.use_mmap = use_mmap
        self.shuffle_mode = shuffle_mode
        self.read_gpu = read_gpu
        self.pipeline = None
        self.iterator = None
        self.lazy_init = lazy_init
        self.transpose = transpose
        self.augmentations = augmentations
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.is_validation = is_validation
        self.seed = seed
        self.epoch_size = 0
        self.oversampling_factor = oversampling_factor

        assert(self.oversampling_factor == 1)

        # shuffle mode:
        if self.shuffle_mode is not None:
            self.shuffle = True
            self.stick_to_shard = False
        else:
            self.shuffle = False
            self.stick_to_shard = True
            
        # init files
        self.init_files(root_dir, prefix_datalabel,
                        statsfile, file_list_datalabel)
        
        self.iterator = DALIGenericIterator([self.pipeline], ['data', 'label'], auto_reset = True,
                                            reader_name = "datalabel",
                                            last_batch_policy = LastBatchPolicy.PARTIAL if self.is_validation else LastBatchPolicy.DROP,
                                            prepare_first_batch = False)

        self.epoch_size = self.pipeline.epoch_size()
        

    @property
    def shapes(self):
        return self.data_shape, self.label_shape

    
    def __iter__(self):
        #self.iterator.reset()
        for token in self.iterator:
            data = token[0]['data']
            label = token[0]['label']
            
            yield data, label, ""



# some small test code
def main(args):

    # create dataloader
    loader = CamDaliNumpyFusedDataloader(args.data_dir_prefix,
                                         'datalabel-*.npy',
                                         args.statsfile,
                                         args.local_batch_size,
                                         file_list_datalabel = None,
                                         num_threads = 1, device = torch.device("cuda:0"),
                                         num_shards = 1, shard_id = 0,
                                         shuffle_mode = None, oversampling_factor = 1,
                                         is_validation = False,
                                         lazy_init = False, transpose = True, augmentations = [],
                                         use_mmap = True, read_gpu = args.ebale_gds, seed = 333)
    
    count = 0
    for data, label, filename in loader:
        count += 1
    

if __name__ == "__main__":
    import argparse as ap
    AP = ap.ArgumentParser()
    AP.add_argument("--data_dir_prefix", type=str, default='/', help="prefix to data dir")
    AP.add_argument("--statsfile", type=str, default='/', help="full path to statsfile") 
    AP.add_argument("--enable_gds", action='store_true')
    AP.add_argument("--local_-batch_size", type=int, default=1, help="Number of samples per local minibatch")
    args = AP.parse_args()

    main(args)
    
