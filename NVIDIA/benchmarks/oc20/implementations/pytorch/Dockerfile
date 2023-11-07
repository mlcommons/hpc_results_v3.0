# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
FROM ${FROM_IMAGE_NAME}

RUN FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0+PTX" \
    CPATH=/usr/local/cuda/include \
    pip install --no-cache-dir torch-geometric==1.7.2

ADD ./requirements.txt .
RUN pip install --disable-pip-version-check -r requirements.txt

RUN mkdir -p /opt/cugraph-ops &&\
    cd /opt/cugraph-ops &&\
    wget https://files.pythonhosted.org/packages/d5/02/604753f90b84696e1c67983afee39b56ec0898c78df573d7ecb388d2ffbf/spdlog-2.0.6.tar.gz &&\
    tar -zxvf spdlog-2.0.6.tar.gz

WORKDIR /workspace/oc20
ADD . /workspace/oc20 

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0+PTX"
ENV CPATH="${CPATH}:/opt/cugraph-ops/spdlog-2.0.6/spdlog/include"

RUN rm -rf /opt/cugraph-ops-binding/build &&\
    mkdir -p /opt/cugraph-ops-binding/build &&\
    cd /opt/cugraph-ops-binding/build &&\
    cmake -DINSTALL_BASE="" /workspace/oc20/cugraph-ops-binding &&\
    make -j2
ENV PYTHONPATH="${PYTHONPATH}:/opt/cugraph-ops-binding/build"

RUN cp pyg_patch/__init__.py /usr/local/lib/python3.10/dist-packages/torch_geometric
RUN cp pyg_patch/data/* /usr/local/lib/python3.10/dist-packages/torch_geometric/data

RUN pip install -e .

ENV OMP_NUM_THREADS=16
ENV CUBLASLT_HEURISTICS_CACHE_CAPACITY=0
