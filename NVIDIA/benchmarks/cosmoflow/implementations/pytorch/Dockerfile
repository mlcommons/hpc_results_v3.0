# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.09-py3

FROM ${FROM_IMAGE_NAME} as build

ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
       && apt-get install -y --no-install-recommends \
       libboost-dev libboost-python-dev libboost-numpy-dev libboost-iostreams-dev libaio-dev \
       && rm -rf /var/lib/apt/lists/*

COPY utils/csrc .
RUN mkdir build; cd build; cmake ..; make -j8


FROM ${FROM_IMAGE_NAME}

# Recommended SHARP ENV settings for NCCL
#ENV SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD=0 \
#    SHARP_COLL_LOCK_ON_COMM_INIT=1
ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive
# Install dependencies for system configuration logger
RUN apt-get update \
       && apt-get install -y --no-install-recommends \
       infiniband-diags \
       pciutils \
       libboost-python-dev libboost-numpy-dev libboost-iostreams-dev libaio-dev \
       && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /workspace/cosmoflow

COPY requirements.txt .
RUN pip install "git+https://github.com/mlcommons/logging.git@hpc-3.0.0" \
       && pip install --no-cache-dir -r requirements.txt

# Copy Cosmoflow code
COPY . .
COPY --from=build /workspace/build/libCosmoflowExt.so utils/libCosmoflowExt.so
