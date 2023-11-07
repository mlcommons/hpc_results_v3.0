#!/bin/bash

cd ../implementation_fugaku/setup

# Build and Install TensorFlow with OneDNN for aarch64 as described on the following page
# https://github.com/fujitsu/tensorflow/wiki/TensorFlow-oneDNN-build-manual-for-FUJITSU-Software-Compiler-Package-(TensorFlow-v2.2.0)

# install mesh-tensorflow module
11_installMTF.sh

# install MPI4py
13_installMPI4py.sh
