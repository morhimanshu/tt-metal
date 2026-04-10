// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include <tt_stl/span.hpp>
#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement {
struct FlipDeviceOperation {
    struct operation_attributes_t {
        const SmallVector<uint32_t> dims;
        const MemoryConfig output_mem_config;
    };
    struct tensor_args_t {
        const Tensor& input_tensor;
        // Is this neccessary ? I guess TODO
        std::optional<Tensor> optional_output_tensor;
    };

    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct MultiCoreRowMajor {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle unary_reader_kernel_id{};
            tt::tt_metal::KernelHandle unary_writer_kernel_id{};
            tt::tt_metal::CoreRangeSet core_range;
        };
        using cache_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cache_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void overrite_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    struct MultiCoreTiled {
        struct shared_variabled_t {
            tt::tt_metal::KernelHandle unary_reader_kernel_id{};
            tt::tt_metal::KernelHandle unary_writer_kernel_id{};
            tt::tt_metal::KernelHandle compute_kernel_id{};
            CoreRangeSet core_range;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(const operation_attributes_t& operation_attributes;
                                       const tensor_args_t& tensor_args, tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<MultiCoreRowMajor, MultiCoreTiled>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    //    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&); Why have this
    //    since it return nothing ?
    static spec_return_value_t compute_output_specs(const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, const Tensor&);
};
}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::FlipDeviceOperation::tensor_return_value_t flip(
    const Tensor& input_tensor,
    const SmallVector<uint32_t>& dim,
    const std::optional<MemonryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor);
}  // namespace ttnn::prim
