# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG FROM_IMAGE_NAME=nvcr.io/nvdlfwea/pytorch:23.09-py3

FROM ${FROM_IMAGE_NAME}

ENV DEBIAN_FRONTEND=noninteractive


# custom triton installation 
## issues with finding libcuda.so on PM, so we revert the
## commit which replaces `whereis` to `ldconfig -p`
RUN pip uninstall -y triton
WORKDIR /workspace/downloads/triton
RUN git clone --recursive https://github.com/openai/triton.git \
    && cd triton \
    && git reset --hard e621604 \
    && git revert --no-commit c9ab448 \
    && cd python \
    && python setup.py install


# Install pip requirements:
RUN pip install \
    biopython==1.79 \
    Pympler==1.0.1 \
    dacite==1.8.0 \
    "git+https://github.com/mlcommons/logging.git@hpc-3.0.0" \
    "git+https://github.com/NVIDIA/mlperf-common.git"

# Build and install Kalign from source:
RUN wget -q -P /workspace/downloads https://github.com/TimoLassmann/kalign/archive/refs/tags/v3.3.5.tar.gz \
    && tar -xzf /workspace/downloads/v3.3.5.tar.gz --directory /workspace \
    && rm -r /workspace/downloads \
    && ls /workspace \
    && cd /workspace/kalign-3.3.5 \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j \
    && make install \
    && rm -r /workspace/kalign-3.3.5

# Copy OpenFold source code into the docker image:
COPY . /workspace/openfold
WORKDIR /workspace/openfold

# Install OpenFold source code package in editable mode:
RUN pip install -e .

# Set number of threads to use for parallel regions:
ENV OMP_NUM_THREADS=1
