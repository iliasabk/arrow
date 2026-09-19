// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/python/util.h"

#include <cstdint>
#include <limits>

#include "arrow/array.h"
#include "arrow/python/common.h"

namespace arrow ::py {

Result<std::shared_ptr<Array>> Arange(int64_t start, int64_t stop, int64_t step,
                                      MemoryPool* pool) {
  if (step == 0) {
    return Status::Invalid("Step must not be zero");
  }
  // The element count and the produced values are computed in unsigned
  // arithmetic: |stop - start| and i * step may not fit in an int64_t
  // (e.g. start == INT64_MIN), where signed arithmetic would overflow
  // (GH-51393).
  uint64_t length;
  uint64_t ustep;
  bool descending;
  if (step > 0 && stop > start) {
    const uint64_t diff =
        static_cast<uint64_t>(stop) - static_cast<uint64_t>(start);
    ustep = static_cast<uint64_t>(step);
    // Ceiling division for positive step (diff + step - 1 could overflow)
    length = diff / ustep + (diff % ustep != 0);
    descending = false;
  } else if (step < 0 && stop < start) {
    const uint64_t diff =
        static_cast<uint64_t>(start) - static_cast<uint64_t>(stop);
    // Unsigned negation yields |step| and stays defined for INT64_MIN
    ustep = 0 - static_cast<uint64_t>(step);
    // Ceiling division for negative step
    length = diff / ustep + (diff % ustep != 0);
    descending = true;
  } else {
    return MakeEmptyArray(int64());
  }
  if (length > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
                   sizeof(int64_t)) {
    return Status::Invalid("arange: requested range is too large");
  }
  const int64_t size = static_cast<int64_t>(length);
  std::shared_ptr<Buffer> data_buffer;
  ARROW_ASSIGN_OR_RAISE(data_buffer, AllocateBuffer(size * sizeof(int64_t), pool));
  auto values = reinterpret_cast<int64_t*>(data_buffer->mutable_data());
  const uint64_t ustart = static_cast<uint64_t>(start);
  if (descending) {
    for (int64_t i = 0; i < size; ++i) {
      values[i] =
          static_cast<int64_t>(ustart - static_cast<uint64_t>(i) * ustep);
    }
  } else {
    for (int64_t i = 0; i < size; ++i) {
      values[i] =
          static_cast<int64_t>(ustart + static_cast<uint64_t>(i) * ustep);
    }
  }
  auto data = ArrayData::Make(int64(), size, {nullptr, data_buffer}, 0);
  return MakeArray(data);
}

}  // namespace arrow::py
