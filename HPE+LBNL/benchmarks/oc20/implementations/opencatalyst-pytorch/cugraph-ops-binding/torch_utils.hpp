/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <torch/extension.h>

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>

namespace torch_extension {

// note that the utility functions below do not type check!
template <typename T>
inline T* maybe_ptr(at::optional<at::Tensor>& tensor)
{
  if (tensor.has_value()) return reinterpret_cast<T*>(tensor.value().data_ptr());
  return nullptr;
}

template <typename T>
inline const T* maybe_ptr(const at::optional<at::Tensor>& tensor)
{
  if (tensor.has_value()) return reinterpret_cast<const T*>(tensor.value().data_ptr());
  return nullptr;
}

template <typename T>
inline T* get_ptr(at::Tensor& tensor)
{
  return reinterpret_cast<T*>(tensor.data_ptr());
}

template <typename T>
inline const T* get_ptr(const at::Tensor& tensor)
{
  return reinterpret_cast<const T*>(tensor.data_ptr());
}

}  // namespace torch_extension
