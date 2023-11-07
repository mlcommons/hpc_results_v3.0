import os
import sys
import numpy as np
import torch
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from .common import get_datashapes


class DummyDaliDataloader(object):

    def get_pipeline(self):
        self.data_size = np.prod(self.data_shape) * 4
        self.label_size = np.prod(self.label_shape) * 8
        
        pipeline = Pipeline(batch_size=self.batchsize, 
                            num_threads=self.num_threads, 
                            device_id=self.device.index,
                            seed = self.seed)
                                 
        with pipeline:
            data = fn.constant(device="gpu",
                               fdata=1.,
                               shape=self.data_shape,
                               bytes_per_sample_hint = self.data_size,
                               dtype=types.DALIDataType.FLOAT)

            label = fn.constant(device="gpu",
                                idata=0,
                                shape=self.label_shape,
                                bytes_per_sample_hint = self.label_size,
                                dtype=types.DALIDataType.INT64)
                            
            if self.transpose:
                data = fn.transpose(data,
                                    device = "gpu",
                                    perm = [2, 0, 1],
                                    bytes_per_sample_hint = self.data_size)
                                
            pipeline.set_outputs(data, label)
            
        return pipeline
            
        
    def __init__(self, root_dir, prefix_data, prefix_label, statsfile,
                 batchsize, file_list_data = None, file_list_label = None,
                 num_threads = 1, device = torch.device("cpu"),
                 num_shards = 1, shard_id = 0,
                 shuffle_mode = False, oversampling_factor = 1,
                 is_validation = False,
                 lazy_init = False, transpose = True, augmentations = [],
                 use_mmap = True, read_gpu = False, seed = 333):
    
        # read filenames first
        self.batchsize = batchsize
        self.num_threads = num_threads
        self.device = device
        self.io_device = "gpu" if read_gpu else "cpu"
        self.read_gpu = read_gpu
        self.pipeline = None
        self.iterator = None
        self.transpose = transpose
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.is_validation = is_validation
        self.seed = seed
        self.epoch_size = 0
        self.oversampling_factor = oversampling_factor
        self.data_shape, self.label_shape = get_datashapes()

        # set up pipeline
        self.pipeline = self.get_pipeline()
        
        # build pipes
        self.global_size = 121266 if not self.is_validation else 15158
        self.local_size = self.global_size // self.num_shards
        self.pipeline.build()
        
        # init iterator        
        self.iterator = DALIGenericIterator([self.pipeline], ['data', 'label'], auto_reset = True,
                                            size = self.local_size,
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
