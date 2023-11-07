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


class CamDaliRecordIODataloader(object):

    def get_pipeline(self, data_files, label_files, num_shards, shard_id):
        self.data_size = 1152*768*16*4
        self.label_size = 1152*768*4
        
        pipeline = Pipeline(batch_size=self.batchsize, 
                            num_threads=self.num_threads, 
                            device_id=self.device.index,
                            seed = self.seed)

        data_shape, label_shape = get_datashapes()
                                 
        with pipeline:
            # read data
            data = fn.readers.mxnet(name="data",
                                    device = self.io_device,
                                    path=data_files,
                                    index_path=[x.replace(".rec", ".idx") for x in data_files],
                                    num_shards = num_shards,
                                    shard_id = shard_id,
                                    stick_to_shard = self.stick_to_shard,
                                    shuffle_after_epoch = self.shuffle,
                                    prefetch_queue_depth = 2,
                                    pad_last_batch = True,
                                    dont_use_mmap = not self.use_mmap,
                                    lazy_init = self.lazy_init,
                                    bytes_per_sample_hint = self.data_size,
                                    seed = self.seed).gpu()

            label = fn.readers.mxnet(name="label",
                                     device = self.io_device,
                                     path=label_files,
                                     index_path=[x.replace(".rec", ".idx") for x in label_files],
                                     num_shards = num_shards,
                                     shard_id = shard_id,
                                     stick_to_shard = self.stick_to_shard,
                                     shuffle_after_epoch = self.shuffle,
                                     prefetch_queue_depth = 2,
                                     pad_last_batch = True,
                                     dont_use_mmap = not self.use_mmap,
                                     lazy_init = self.lazy_init,
                                     bytes_per_sample_hint = self.label_size,
                                     seed = self.seed).gpu()

            # get into right format
            data = fn.reinterpret(data,
                                  device = "gpu",
                                  dtype = types.DALIDataType.FLOAT,
                                  layout = "HWC",
                                  shape = data_shape,
                                  bytes_per_sample_hint = self.data_size)

            label = fn.reinterpret(label,
                                   device = "gpu",
                                   dtype = types.DALIDataType.INT32,
                                   layout = "HW",
                                   shape = label_shape,
                                   bytes_per_sample_hint = self.label_size) 
                                       
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
            


    def init_files(self, root_dir, prefix_data, prefix_label, statsfile, file_list_data = None, file_list_label = None):
        self.root_dir = root_dir
        self.prefix_data = prefix_data
        self.prefix_label = prefix_label

        # get files
        # data
        if file_list_data is not None and os.path.isfile(os.path.join(root_dir, file_list_data)):
            with open(os.path.join(root_dir, file_list_data), "r") as f:
                token = f.readlines()
            self.data_files = [os.path.join(root_dir, x.strip()) for x in token]
        else:
            self.data_files = glob.glob(os.path.join(self.root_dir, self.prefix_data))
        # label
        if file_list_label is not None and os.path.isfile(os.path.join(root_dir, file_list_label)):
            with open(os.path.join(root_dir, file_list_label), "r") as f:
                token = f.readlines()
            self.label_files = [os.path.join(root_dir, x.strip()) for x in token]
        else:
            self.label_files = glob.glob(os.path.join(self.root_dir, self.prefix_label))

        # get shapes
        self.data_shape = np.load(self.data_files[0]).shape
        self.label_shape = np.load(self.label_files[0]).shape

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
        data_files = self.data_files
        label_files = self.label_files
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
            num_files = len(data_files)
            num_files_per_shard = num_files // num_shards
            shard_start = shard_id * num_files_per_shard
            shard_end = shard_start + num_files_per_shard
            
            # get the remainder now
            rem_start = num_shards * num_files_per_shard
            rem_end = num_files
            data_rem = data_files[rem_start:rem_end]
            label_rem = label_files[rem_start:rem_end]

            # get the bulk
            data_files = data_files[shard_start:shard_end]
            label_files = label_files[shard_start:shard_end]
            
            # append remainder
            if shard_id < len(data_rem):
                data_files.append(data_rem[shard_id])
                label_files.append(label_rem[shard_id])

        # reset shard ids 
        if self.shuffle_mode == "gpu":
            num_shards = 1
            shard_id = 0
        elif self.shuffle_mode == "node":
            num_shards = num_local_ranks
            shard_id = local_shard_id
        
        # set up pipeline
        self.pipeline = self.get_pipeline(data_files, label_files, num_shards, shard_id)
       
        # build pipes
        self.global_size = len(self.data_files)
        self.pipeline.build()

        # init iterator
        if not self.lazy_init:
            self.init_iterator()

        
    def __init__(self, root_dir, prefix_data, prefix_label, statsfile,
                 batchsize, file_list_data = None, file_list_label = None,
                 num_threads = 1, device = torch.device("cpu"),
                 num_shards = 1, shard_id = 0,
                 shuffle_mode = None, is_validation = False,
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

        # shuffle mode:
        if self.shuffle_mode is not None:
            self.shuffle = True
            self.stick_to_shard = False
        else:
            self.shuffle = False
            self.stick_to_shard = True
        
        # init files
        self.init_files(root_dir, prefix_data, prefix_label,
                        statsfile, file_list_data, file_list_label)
        
        self.iterator = DALIGenericIterator([self.pipeline], ['data', 'label'], auto_reset = True,
                                            reader_name = "data",
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
