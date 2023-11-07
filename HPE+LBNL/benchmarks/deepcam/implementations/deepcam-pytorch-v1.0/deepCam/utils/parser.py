import argparse as ap

#dict helper for argparse
class StoreDictKeyPair(ap.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)

def get_parser():
    
    AP = ap.ArgumentParser()
    AP.add_argument("--wireup_method", type=str, default="nccl-openmpi", choices=["dummy", "nccl-file", "nccl-openmpi", \
                                                                                  "nccl-slurm", "nccl-slurm-pmi", "mpi"], help="Specify what is used for wiring up the ranks")
    AP.add_argument("--wandb_certdir", type=str, default="/opt/certs", help="Directory in which to find the certificate for wandb logging.")
    AP.add_argument("--run_tag", type=str, help="Unique run tag, to allow for better identification")
    AP.add_argument("--experiment_id", type=int, default=1, help="Experiment Number")
    AP.add_argument("--output_dir", type=str, help="Directory used for storing output. Needs to read/writeable from rank 0")
    AP.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to restart training from.")
    AP.add_argument("--data_dir_prefix", type=str, default='/', help="prefix to data dir")
    AP.add_argument("--max_inter_threads", type=int, default=1, help="Maximum number of concurrent readers")
    AP.add_argument("--max_epochs", type=int, default=30, help="Maximum number of epochs to train")
    AP.add_argument("--save_frequency", type=int, default=100, help="Frequency with which the model is saved in number of steps")
    AP.add_argument("--logging_frequency", type=int, default=100, help="Frequency with which the training progress is logged. If not positive, logging will be disabled")
    AP.add_argument("--local_batch_size", type=int, default=1, help="Number of samples per local minibatch")
    AP.add_argument("--local_batch_size_validation", type=int, default=1, help="Number of samples per local minibatch for validation")
    AP.add_argument("--channels", type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], help="Channels used in input")
    AP.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "AdamW", "LAMB", "DistributedLAMB"], help="Optimizer to use (LAMB requires APEX support).")
    AP.add_argument("--start_lr", type=float, default=1e-3, help="Start LR")
    AP.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    AP.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for linear LR warmup")
    AP.add_argument("--lr_warmup_factor", type=float, default=1., help="Multiplier for linear LR warmup")
    AP.add_argument("--lr_schedule", action=StoreDictKeyPair)
    AP.add_argument("--target_iou", type=float, default=0.82, help="Target IoU score.")
    AP.add_argument("--model_prefix", type=str, default="model", help="Prefix for the stored model")
    AP.add_argument("--gradient_accumulation_frequency", type=int, default=1, help="Number of gradient accumulation steps before update")
    AP.add_argument("--batchnorm_group_size", type=int, default=1, help="Process group size for sync batchnorm")
    AP.add_argument("--shuffle_mode", type=str, default="global", choices=["global", "node", "gpu"], help="Specifies how to shuffle the data")
    AP.add_argument("--data_format", type=str, default="dali-numpy", choices=["hdf5", "dali-numpy", "dali-recordio", "dali-es", "dali-es-disk", "dali-dummy"], help="Specify data format")
    AP.add_argument("--data_cache_directory", type=str, default="/tmp", help="Directory to which the data is cached. Only relevant for dali-es-disk dataloader, ignored otherwise")
    AP.add_argument("--data_oversampling_factor", type=int, default=1, help="Determines how many different shard per nodes will be staged")
    AP.add_argument("--precision_mode", type=str, default="amp", choices=["fp32", "amp", "fp16"], help="Specify precision format")
    AP.add_argument("--enable_gds", action='store_true')
    AP.add_argument("--enable_jit", action='store_true')
    AP.add_argument("--enable_nhwc", action='store_true')
    AP.add_argument("--enable_graph", action='store_true', help="Flag for enabling CUDA graph capture.")
    AP.add_argument("--disable_tuning", action='store_true', help="Flag for disabling cuDNN benchmark mode to autotune kernels. Should not be necessary")
    AP.add_argument("--force_groupbn", action='store_true')
    AP.add_argument("--disable_validation", action='store_true')
    AP.add_argument("--disable_comm_overlap", action='store_true')
    AP.add_argument("--data_augmentations", type=str, nargs='+', default=[], help="Data augmentations used. Supported are [roll, flip]")
    AP.add_argument("--enable_wandb", action='store_true')
    AP.add_argument("--resume_logging", action='store_true')
    AP.add_argument("--seed", default=333, type=int)
    
    return AP
