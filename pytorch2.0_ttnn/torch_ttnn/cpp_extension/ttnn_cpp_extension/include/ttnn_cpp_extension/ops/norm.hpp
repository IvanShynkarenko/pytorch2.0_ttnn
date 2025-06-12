// ttnn_cpp_extension/include/ttnn_cpp_extension/ops/norm.hpp
#pragma once
#include <torch/extension.h>
#include <c10/util/Optional.h>

namespace tt_eager::ops::norm {

// LayerNorm
at::Tensor ttnn_layer_norm(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double eps);

}  // namespace tt_eager::ops::norm
