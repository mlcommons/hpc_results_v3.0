export OMPI_MCA_btl="^openib" #To prevent deadlock between Horovd and NCCL at 96 nodes
export DALI_DONT_USE_MMAP=1 # 0 for /raid and 1 for lustre
export MXNET_EXTENDED_NORMCONV_SUPPORT=1 # supports Arch 80 NormConv fusion

## System config params
export DGXNGPU=4
export DGXNSOCKET=1
export DGXSOCKETCORES=64
export DGXHT=2  # HT is on is 2, HT off is 1
export HOROVOD_NUM_NCCL_STREAMS=1
export MXNET_HOROVOD_NUM_GROUPS=1
export HOROVOD_CYCLE_TIME=0.1
export MXNET_OPTIMIZER_AGGREGATION_SIZE=54
export MXNET_ENABLE_CUDA_GRAPHS=1

export OMPI_MCA_coll_hcoll_enable=0 
export HCOLL_ENABLE_MCAST=0  
export NCCL_NET_GDR_LEVEL=PHB

# Enable SHARP
#export NCCL_COLLNET_ENABLE=1
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_IFNAME=hsn

export DTYPE="amp"
export STAGING_DIR="/tmp"
