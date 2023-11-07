#!/bin/bash

SRC_DIR=${HOME}/cosmoUniverse_2019_05_4parE_tf_v2
DST_DIR=${HOME}/cosmoUniverse_2019_05_4parE_tf_v2_nocomp
NPROC=512

echo "`date +%s.%N` #decompress start at `date`"

mpirun -n $NPROC \
       -x CUDA_VISIBLE_DEVICES="" \
       python decompress.py -i $SRC_DIR -o $DST_DIR

echo "`date +%s.%N` #decompress end at `date`"

