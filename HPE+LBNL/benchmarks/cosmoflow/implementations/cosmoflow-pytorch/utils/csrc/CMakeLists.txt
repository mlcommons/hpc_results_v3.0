# Copyright (c) 2021-2022 NVIDIA CORPORATION. All rights reserved.
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

cmake_minimum_required(VERSION 3.10)
project(CosmoflowExt)

find_package(Python COMPONENTS Development NumPy)
find_package(Boost COMPONENTS python numpy iostreams)

add_library(CosmoflowExt SHARED
    src/main.cc
    src/common.hh
    src/file_direct.hh
    src/file_direct.cc
    src/aio_handler.hh
    src/aio_handler.cc
)

set_property(TARGET CosmoflowExt PROPERTY CXX_STANDARD 20)
target_link_libraries(CosmoflowExt
    Python::Module
    Python::NumPy
    Boost::python
    Boost::numpy
    Boost::iostreams
    aio)

set(CMAKE_CXX_FLAGS "-O2 -Wall -Wextra")
