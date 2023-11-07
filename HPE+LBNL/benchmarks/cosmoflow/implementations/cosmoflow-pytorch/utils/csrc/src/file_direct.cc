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

#include "file_direct.hh"
#include "common.hh"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>

#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

static inline size_t get_file_size(int fd) {
  struct stat sb;
  fstat(fd, &sb);
  return sb.st_size;
}

static bool read_file_impl(int fd, char *write_buffer, off_t offset,
                           std::size_t size) {
  while (size > 0) {
    ssize_t read_bytes = pread(fd, write_buffer, size, offset);
    if (read_bytes >= 0) {
      size -= read_bytes;
      offset += read_bytes;
      write_buffer += read_bytes;
    } else {
      return false;
    }
  }
  return true;
}

static bool write_file_impl(int fd, const char *read_buffer, off_t offset,
                            std::size_t size) {
  while (size > 0) {
    ssize_t write_bytes = pwrite(fd, read_buffer, size, offset);
    if (write_bytes >= 0) {
      size -= write_bytes;
      offset += write_bytes;
      read_buffer += write_bytes;
    } else {
      return false;
    }
  }
  return true;
}

std::vector<char> read_file_pure(std::string file_name, std::size_t block_size,
                                 std::size_t file_size) {
  int fd = open(file_name.c_str(), O_RDONLY);
  if (fd == -1)
    throw std::runtime_error("Error, cannot open file " + file_name + ".");

  if (file_size == 0)
    file_size = get_file_size(fd);

  posix_fadvise(fd, 0, file_size, POSIX_FADV_SEQUENTIAL | POSIX_FADV_DONTNEED);
  std::size_t buffer_size =
      ((file_size + block_size - 1) / block_size) * block_size;
  std::size_t aligned_size = (file_size / block_size) * block_size;

  char *buffer =
      static_cast<char *>(std::aligned_alloc(block_size, buffer_size));
  // char *buffer = static_cast<char *>(memalign(block_size, buffer_size));
  if (buffer == nullptr)
    throw std::runtime_error("Error allocating IO buffer.");

  if (!read_file_impl(fd, buffer + aligned_size, aligned_size,
                      file_size - aligned_size))
    throw std::runtime_error("Error while reading reminder of file " +
                             file_name + ".");

  if (fcntl(fd, F_SETFL, O_DIRECT) < 0)
    throw std::runtime_error("Error setting direct access to " + file_name +
                             ".");

  if (!read_file_impl(fd, buffer, 0, aligned_size))
    throw std::runtime_error("Error while reading bulk from file " + file_name +
                             ".");

  close(fd);

  std::vector<char> output_buffer;
  output_buffer.reserve(16777287);

  boost::iostreams::filtering_ostream output_stream;
  output_stream.push(boost::iostreams::gzip_decompressor(15));
  output_stream.push(boost::iostreams::back_inserter(output_buffer));

  boost::iostreams::write(output_stream, buffer, file_size);
  output_stream.flush();

  std::free(buffer);

  return output_buffer;
}

void write_file_pure(std::string file_name, std::size_t block_size,
                     std::size_t file_size, const char *file_content) {
  int fd = open(file_name.c_str(), O_CREAT | O_WRONLY | O_TRUNC, S_IRWXU);
  if (fd == -1)
    throw std::runtime_error("Error, cannot open file " + file_name + ".");

  posix_fadvise(fd, 0, file_size, POSIX_FADV_SEQUENTIAL);
  if (!write_file_impl(fd, file_content, 0, file_size))
    throw std::runtime_error("Error while writing file" + file_name);
#if 0
  std::size_t buffer_size =
      ((file_size + block_size - 1) / block_size) * block_size;
  std::size_t aligned_size = (file_size / block_size) * block_size;

  if (aligned_size == 0) {
    if (!write_file_impl(fd, file_content + aligned_size, aligned_size,
                         file_size - aligned_size))
      throw std::runtime_error("Error while writing reminder of file " +
                               file_name + ".");
    return;
  }

  char *buffer =
      static_cast<char *>(std::aligned_alloc(block_size, buffer_size));
  // char *buffer = static_cast<char *>(memalign(block_size, buffer_size));
  if (buffer == nullptr)
    throw std::runtime_error("Error allocating IO buffer.");

  if (!write_file_impl(fd, file_content + aligned_size, aligned_size,
                       file_size - aligned_size))
    throw std::runtime_error("Error while writing reminder of file " +
                             file_name + ".");

  // if (fcntl(fd, F_SETFL, O_DIRECT) < 0)
  //   throw std::runtime_error("Error setting direct access to " + file_name +
  //                            ".");

  std::memcpy(buffer, file_content, aligned_size);
  if (!write_file_impl(fd, buffer, 0, aligned_size))
    throw std::runtime_error("Error while writing bulk to file " + file_name +
                             ".");
  std::free(buffer);
#endif
  close(fd);
}

void write_file(std::string file_name, const std::string &data,
                std::size_t block_size, bool write_idx) {
  write_file_pure(file_name, block_size, data.size(), data.c_str());

  if (write_idx) {
    std::size_t current = 0, file_size = data.size(), ptr = 0;
    std::string result;

    while (current < file_size) {
      std::int64_t proto_len =
          *reinterpret_cast<const std::int64_t *>(data.c_str() + current);
      std::string current_str = std::to_string(current);
      std::string proto_len_str = std::to_string(proto_len + 16);

      result += current_str + ' ' + proto_len_str + '\n';
      current += (16 + proto_len);
    }

    write_file_pure(file_name + ".idx", block_size, result.size(),
                    result.c_str());
  }
}

void write_file_single(std::string file_name, const std::string &file_data,
                       std::size_t block_size, bool write_idx) {
  GILRemoveGuard gil_(true);
  write_file(file_name, file_data, block_size, write_idx);
}

void write_file_batch(boost::python::list &file_names,
                      boost::python::list &file_datas, std::size_t block_size,
                      bool write_idx) {
  if (boost::python::len(file_names) != boost::python::len(file_datas))
    throw std::runtime_error("Argument sizes doesn't match!");

  for (std::size_t i = 0, list_len = boost::python::len(file_names);
       i < list_len; ++i) {
    std::string file_name = boost::python::extract<std::string>(file_names[i]);
    std::string file_data = boost::python::extract<std::string>(file_datas[i]);
    GILRemoveGuard gil_(true);

    write_file(file_name, file_data, block_size, write_idx);
  }
}

boost::python::object read_file(std::string file_name, std::size_t block_size,
                                std::size_t file_size) {
  std::vector<char> output_buffer;
  {
    GILRemoveGuard nogil_;
    output_buffer = read_file_pure(file_name, block_size, file_size);
  }

  boost::python::object result = boost::python::object(boost::python::handle<>(
      PyBytes_FromStringAndSize(output_buffer.data(), output_buffer.size())));
  return result;
}

void read_write_fused(std::string input_file, std::string output_file,
                      std::size_t file_size, std::size_t block_size, bool idx) {
  GILRemoveGuard nogil_(true);
  std::vector<char> output_buffer =
      read_file_pure(input_file, block_size, file_size);
  std::string input_buffer =
      std::string(output_buffer.begin(), output_buffer.end());
  write_file(output_file, input_buffer, block_size, idx);
}
