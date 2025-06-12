// ttnn_cpp_extension/include/ttnn_cpp_extension/ops/dropout.hpp
#pragma once
#include <torch/extension.h>

namespace tt_eager::ops::dropout {

// Dropout (тільки inference‐no‐op)
at::Tensor ttnn_dropout(
    const at::Tensor& input,
    double p,
    bool train);

}  // namespace tt_eager::ops::dropout
