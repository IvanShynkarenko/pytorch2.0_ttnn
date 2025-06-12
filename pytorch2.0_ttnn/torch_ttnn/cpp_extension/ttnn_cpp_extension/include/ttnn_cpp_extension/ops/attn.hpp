// ttnn_cpp_extension/include/ttnn_cpp_extension/ops/attn.hpp
#pragma once
#include <torch/extension.h>
#include <c10/util/Optional.h>

namespace tt_eager::ops::attn {

// Реалізація scaled‐dot‐product attention
at::Tensor ttnn_scaled_dot_attn(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor>& mask_opt);

}  // namespace tt_eager::ops::attn
