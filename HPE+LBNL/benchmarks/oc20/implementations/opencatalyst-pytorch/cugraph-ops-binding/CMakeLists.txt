#=============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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
#=============================================================================

set(RAPIDS_VERSION "22.08")

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

include(FetchContent)

FetchContent_Declare(
  rapids-cmake
  GIT_REPOSITORY https://github.com/rapidsai/rapids-cmake.git
  GIT_TAG        origin/branch-${RAPIDS_VERSION}
)
FetchContent_MakeAvailable(rapids-cmake)
include(rapids-cmake)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

project(OCP_CUGRAPH_OPS_BINDING VERSION ${RAPIDS_VERSION} LANGUAGES CXX)

##############################################################################
# - User Options  ------------------------------------------------------------

option(CUDA_STATIC_RUNTIME "Use CUDA static runtime" OFF)
option(INSTALL_BASE "directory where cugraph-ops package is found" /opt/conda)

##############################################################################
# - Dependencies  ------------------------------------------------------------

# default build type
rapids_cmake_build_type(Release)

# CUDA runtime
rapids_cuda_init_runtime(USE_STATIC ${CUDA_STATIC_RUNTIME})

# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
rapids_find_package(CUDAToolkit REQUIRED
    BUILD_EXPORT_SET ocp-exports
    INSTALL_EXPORT_SET ocp-exports
)

##############################################################################
# - Compiler options ---------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(DEFAULT_CXX_FLAGS "")

# reset the default flags if we have DEBUG_CXXFLAGS
if(CMAKE_BUILD_TYPE MATCHES Debug)
  if (DEFINED ENV{DEBUG_CXXFLAGS})
    set(DEFAULT_CXX_FLAGS "$ENV{DEBUG_CXXFLAGS}")
    separate_arguments(DEFAULT_CXX_FLAGS)
  endif()
endif()

##############################################################################
# - Variables ----------------------------------------------------------------

set(CUGRAPH_OPS_INCLUDE_DIR ${INSTALL_BASE}/include CACHE STRING
  "Path to cugraph-ops include directory")

set(CUGRAPH_OPS_LIB ${INSTALL_BASE}/lib/libcugraph-ops++.so CACHE STRING
  "Path to cugraph-ops C++ library")

set(PY_TARGET "cugraph_ops_binding" CACHE STRING "python target name")

##############################################################################
# - pybind11 - (header only) -------------------------------------------------

# we need this to be downloaded at configure time so that we can safely do
# `add_subdirectory()` on it
# Ref: https://stackoverflow.com/questions/47027741/smart-way-to-cmake-a-project-using-pybind11-by-externalproject-add
FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        8e5d3d234ef3bbd9efdbba865d1e606d4c5e97bb)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
  FetchContent_Populate(pybind11)
endif()

add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})

pybind11_add_module(${PY_TARGET} SHARED
  binding.cpp
  torch_extension.cpp
)

find_package(Python COMPONENTS Interpreter REQUIRED)
set(PY_EXE ${Python_EXECUTABLE})
message(STATUS "Using python ${Python_EXECUTABLE}")
execute_process(COMMAND ${PY_EXE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
                OUTPUT_VARIABLE TORCH_CMAKE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
set(Torch_ROOT "${TORCH_CMAKE_PREFIX}/Torch")
find_package(Torch "1.9.0" "REQUIRED")
if (NOT TORCH_FOUND)
    message(FATAL_ERROR "Torch not found.")
    return()
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import torch.utils.cpp_extension; print(torch.utils.cpp_extension.TORCH_LIB_PATH)"
  OUTPUT_VARIABLE TORCH_LIB_PATH_ OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET COMMAND_ECHO STDOUT
)
# remove any error output which we don't care about
STRING(REGEX REPLACE "\n" ";" TORCH_LIB_PATH_ "${TORCH_LIB_PATH_}")
LIST(GET TORCH_LIB_PATH_ -1 TORCH_LIB_PATH)
find_library(Torch_python_lib
  NAMES torch_python
  HINTS "${TORCH_LIB_PATH}"
)
find_library(Torch_lib
  NAMES torch
  HINTS "${TORCH_LIB_PATH}"
)
if (Torch_python_lib)
  message(STATUS "Found libtorch_python")
else()
  message(FATAL_ERROR "libtorch_python library not found (hint: ${TORCH_LIB_PATH})")
endif()
execute_process(COMMAND ${PY_EXE} -c "import torch; print(torch.torch.compiled_with_cxx11_abi())"
        OUTPUT_VARIABLE Torch_CXX11 OUTPUT_STRIP_TRAILING_WHITESPACE)
string(TOUPPER ${Torch_CXX11} Torch_CXX11)
message(STATUS "Torch_CXX11: ${Torch_CXX11}")
set(USE_CXX11_ABI ${Torch_CXX11})

if (${USE_CXX11_ABI})
    message(STATUS "Using CXX ABI = 1")
    target_compile_definitions(${PY_TARGET} PUBLIC -D_GLIBCXX_USE_CXX11_ABI=1)
else()
    message(STATUS "Using CXX ABI = 0")
    target_compile_definitions(${PY_TARGET} PUBLIC -D_GLIBCXX_USE_CXX11_ABI=0)
endif()

message(STATUS "Got Pytorch includes: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Got Pytorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Linking against: ${CUGRAPH_OPS_LIB}, ${Torch_lib}, ${Torch_python_lib}")

target_include_directories(${PY_TARGET}
  PRIVATE
    ${CUGRAPH_OPS_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
  SYSTEM PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(${PY_TARGET}
  PUBLIC
    ${CUGRAPH_OPS_LIB}
    ${Torch_lib}
    ${Torch_python_lib}
)
