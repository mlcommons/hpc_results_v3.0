export OMPI_MCA_btl="^openib" #To prevent deadlock between Horovd and NCCL at 96 nodes
export DALI_DONT_USE_MMAP=0 # 0 for /raid and 1 for lustre

## System config params
export DGXNGPU=8
export DGXNSOCKET=2
export DGXSOCKETCORES=64
export DGXHT=2  # HT is on is 2, HT off is 1

export OMPI_MCA_coll_hcoll_enable=0 
export HCOLL_ENABLE_MCAST=0  

source $(dirname ${BASH_SOURCE[0]})/config_data_selene.sh

# Enable SHARP
#export NCCL_COLLNET_ENABLE=1

export DTYPE="amp"
export STAGING_DIR="/tmp"
