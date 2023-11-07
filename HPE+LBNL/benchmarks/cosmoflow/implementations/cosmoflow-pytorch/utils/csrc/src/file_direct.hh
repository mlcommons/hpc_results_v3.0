/* Copyright (c) 2021-2022 NVIDIA CORPORATION. All rights reserved.
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

#include <boost/python.hpp>

#include <stdint.h>
#include <string>
#include <vector>

std::vector<char> read_file_pure(std::string, std::size_t, std::size_t);
boost::python::object read_file(std::string, std::size_t, std::size_t);

void write_file_single(std::string, const std::string &, std::size_t, bool);
void write_file_batch(boost::python::list &, boost::python::list &, std::size_t,
                      bool);

void read_write_fused(std::string, std::string, std::size_t, std::size_t, bool);
