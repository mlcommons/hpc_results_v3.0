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

#include <libaio.h>

#include <queue>
#include <stdexcept>
#include <unistd.h>

class AioHandlerRequest {
public:
  AioHandlerRequest(int fd, std::string file_name, char *buffer,
                    std::size_t file_size)
      : fd(fd), memory_buffer_(buffer), file_name_(file_name) {
    io_prep_pwrite(&this->request_, fd, this->memory_buffer_, file_size, 0);
    this->request_.data = this->file_name_.data();
  }
  ~AioHandlerRequest() {
    std::free(this->memory_buffer_);
    close(this->fd);
  }

  iocb *GetRequestIocb() { return &this->request_; }

private:
  int fd;
  char *memory_buffer_;
  std::string file_name_;
  iocb request_;
};

class AioHandler {
public:
  AioHandler(std::size_t event_queue_size)
      : event_queue_size_(event_queue_size) {
    memset(&this->write_ctx_, 0, sizeof(this->write_ctx_));
    // memset(&this->read_ctx_, 0, sizeof(this->read_ctx_));

    if (int error = io_setup(this->event_queue_size_, &this->write_ctx_);
        error < 0)
      throw std::runtime_error("Error, failed to initialize write context." +
                               std::to_string(error));

    // if (int error = io_setup(this->event_queue_size_, &this->read_ctx_);
    //     error < 0)
    //   throw std::runtime_error("Error, failed to initialize read context.");
  }
  ~AioHandler() {
    // io_destroy(this->read_ctx_);
    io_destroy(this->write_ctx_);
  }

  void SubmitSaveFile(std::string file_name, const std::string &data,
                      std::size_t block_size);
  void WaitForAll();

private:
  std::size_t event_queue_size_;
  io_context_t write_ctx_;
  // io_context_t read_ctx_;

  std::queue<AioHandlerRequest> event_queue;
};
