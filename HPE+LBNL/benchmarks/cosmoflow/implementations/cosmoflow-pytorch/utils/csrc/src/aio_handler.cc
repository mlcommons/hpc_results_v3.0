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

#include "aio_handler.hh"

#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>

void AioHandler::SubmitSaveFile(std::string file_name, const std::string &data,
                                std::size_t block_size) {
  int fd =
      open(file_name.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_DIRECT, S_IRWXU);
  if (fd == -1)
    throw std::runtime_error("Error, cannot open file " + file_name);

  // std::cout << "Saving file" << file_name << std::endl;

  std::size_t file_size = data.size();
  posix_fadvise(fd, 0, file_size, POSIX_FADV_SEQUENTIAL | POSIX_FADV_DONTNEED);

  // std::cout << "File size" << file_size << std::endl;

  const std::size_t alignment = block_size;
  const std::size_t buffer_size =
      ((file_size + alignment - 1) / alignment) * alignment;

  char *buffer =
      static_cast<char *>(std::aligned_alloc(block_size, buffer_size));
  if (buffer == nullptr)
    throw std::runtime_error("Error allocating IO buffer");

  // std::cout << "Allocated everything, time to copy" << std::endl;

  std::memcpy(buffer, data.c_str(), file_size);
  this->event_queue.emplace(fd, file_name, buffer, buffer_size);

  // std::cout << "Submitting request" << std::endl;
  iocb *cbs[1] = {this->event_queue.back().GetRequestIocb()};
  if (io_submit(this->write_ctx_, 1, cbs) < 0)
    throw std::runtime_error("Error submitting write for file " + file_name);
}

void AioHandler::WaitForAll() {
  for (std::size_t completed = 0; completed < this->event_queue.size();) {
    io_event event_buffer[512];
    int now_completed =
        io_getevents(this->write_ctx_, 1, 512, event_buffer, nullptr);
    if (now_completed < 0)
      break;
    completed += now_completed;
  }
  while (!this->event_queue.empty())
    this->event_queue.pop();
  // io_queue_run(this->write_ctx_);
}
