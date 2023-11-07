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

#include <boost/python.hpp>

#include "aio_handler.hh"
#include "file_direct.hh"

const char *hello() { return "Hello World!"; }

BOOST_PYTHON_MODULE(libCosmoflowExt) {
  PyEval_InitThreads();

  boost::python::def("hello", &hello);
  boost::python::def("read_file", &read_file);
  boost::python::def("write_file", &write_file_single);
  boost::python::def("write_file_batch", &write_file_batch);
  boost::python::def("read_write_fused", &read_write_fused);

  boost::python::class_<AioHandler>("AioHandler",
                                    boost::python::init<std::size_t>())
      .def("submit_save_file", &AioHandler::SubmitSaveFile)
      .def("wait_for_all", &AioHandler::WaitForAll);
}
