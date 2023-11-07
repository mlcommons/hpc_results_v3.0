export OMPI_MCA_btl="^openib" #To prevent deadlock between Horovd and NCCL at 96 nodes
export DALI_DONT_USE_MMAP=1 # 0 for /raid and 1 for lustre

## System config params
export DGXNGPU=4
export DGXNSOCKET=1
export DGXSOCKETCORES=64
export DGXHT=2  # HT is on is 2, HT off is 1

# Network settings
export OMPI_MCA_coll_hcoll_enable=0 
export HCOLL_ENABLE_MCAST=0  
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=hsn

export DTYPE="amp"
#export STAGING_DIR="/tmp"
export STAGING_DIR="/dev/shm"

export DATADIR=/pscratch/sd/s/sfarrell/cosmoflow-benchmark/data/hpc_v2.0_gzip
#export DATADIR=/pscratch/sd/n/namehta4/optimized-hpc/cosmoUniverse_2019_05_4parE_tf_v2_numpy
