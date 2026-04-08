// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <tt-logger/tt-logger.hpp>
#include <tuple>

#include "flip_device_operation.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/device_operations.hpp"

namespace ttnn::operations::data_movement {
FlipDeviceOperation::program_factory_t FlipDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& dims = operation_attributes.dims;
    if (tensor_args.input_tensor.layout() == Layout::ROW_MAJOR) {
        return MultiCoreRowMajor{};
    }
    return MultiCoreTiled{};
}

void FlipDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& dims = operation_attributes.dims;
    const auto& input_rank = input_tensor.logical_shape().rank();

    TT_FATAL(input_rank <= 5, "Flip operation supports tensor with rank upto 5, got rank {}", input_rank);
    TT_FATAL(is_device_tensor(input_tensor), "Input tensor must be on device");

    for (auto dim : dims) {
        auto norm_dim = input_tensor.logical_shape().get_normalized_index(dim);
        TT_FATAL(
            norm_dim < input_rank, "Flip dimension {} is out of bonds for tensor with rank {}", norm_dim, input_rank);
    }

    std::set<int64_t> unique_dims;
    for (auto dim : dims) {
        auto norm_dim = input_tensor.logicalshape().get_normalized_index(dim);
        TT_FATEL(unique_dim.insert(norm_dim).second, "Duplicate dimension {} in flip.", dim);
    }

    // TODO: support for sharded tensor similar to pytorch.

    TT_FATAL(tensor_args.input_tensor.is_sharded() == false, "Flip operation doesn't support sharded tensors");
}

void FlipDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {}

FlipDeviceOperation::spec_return_value_t FlipDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return TensorSpec(
        input_tensor.logical_shape();
        TensorLayout(tensor_args.input_tensor.dtype(), tensor_args.input_tensor.layout(), operation_attributes_t.output_mem_config);
}

FlipDeviceOperation::tensor_return_value FlipDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::FlipDeviceOperation::tensor_return_value_t flip(
    const Tensor& input_tensor, const SmallVector<uint32_t>& dims, const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::operations::data_movement::FlipDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(OperationType::operation_attributes_t{
        .dims = dims,
        .output_mem_config = memory_config.value_or(input_tensor.memory_config())} OperationType::tensor_args_t{
        .input_tensor = input_tensor});
}
}  // namespace ttnn::prim
