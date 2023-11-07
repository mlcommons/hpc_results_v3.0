#!/bin/bash

SRC_DIR=${HOME}/cosmoUniverse_2019_05_4parE_tf_v2_nocomp
DST_DIR=${HOME}/cosmoUniverse_2019_05_4parE_tf_v2_nocomp_8192_txz
COMPRESS="xz"
NPROC=32
N_TAR=8192

for type in train validation ; do
    echo $type
    mkdir -p $DST_DIR/$type
    echo "`date +%s.%N` #packing ${SRC_DIR}/$type to ${DST_DIR}/$type start at `date`"
    mpirun -n $NPROC ./pack_v2_core.sh ${SRC_DIR} ${DST_DIR} $type ./full_${type}_files_v2 $N_TAR $COMPRESS
    echo "`date +%s.%N` #packing $type data end at `date`"
done
