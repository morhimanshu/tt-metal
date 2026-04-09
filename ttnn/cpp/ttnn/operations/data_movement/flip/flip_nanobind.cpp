// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include "flip.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_permute(nb::module_& mod) {
    const auto* doc = R"doc(
        Flips the dimensions of the input tensor along specified dims.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            dim (tuple): axis to flip on.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.
    )doc";

    ttnn::bind_function<"flip">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttnn::SmallVector<int64_t>&,
                const std::optional<ttnn::MemoryConfig>&>(&ttnn::),
            nb::arg("input_tensor").noconvert(),
            nb::arg("dims"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, const ttnn::SmallVector<int64_t>&, float>(&ttnn::flip),
            nb::arg("input_tensor").noconvert(),
            nb::arg("dims"),
}

}  // namespace ttnn::operations::data_movement::detail
