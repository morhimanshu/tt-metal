// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flip.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::data_movement::detail {
ttnn::Tensor flip(
    const ttnn::Tensor& a,
    const ttnn::SmallVector<uint32_t>& dims,
    const std::optional<MemoryConfig>& output_mem_config) {
    uint32_t rank = a.logical_shape().rank();
}
}  // namespace ttnn::operations::data_movement::detail
