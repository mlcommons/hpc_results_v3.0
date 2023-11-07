#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=small"
#PJM -L elapse=00:30:00
#PJM -L "node=1"
#PJM --mpi "proc=1"
#PJM -j
#PJM -S

set -ex

. ../setenv

cd ${COSMOFLOW_BASE}/setup/mpi4py

python setup.py install

pip list

echo "#end"
