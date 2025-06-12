// ttnn_cpp_extension/include/ttnn_cpp_extension/ops/linear.hpp
#pragma once
#include <torch/extension.h>
#include <c10/util/Optional.h>

namespace tt_eager::ops::linear {

// Лінійний шар / dense (matmul + optional bias)
at::Tensor ttnn_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt);

}  // namespace tt_eager::ops::linear
