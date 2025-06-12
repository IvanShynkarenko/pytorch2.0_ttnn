// ttnn_cpp_extension/include/ttnn_cpp_extension/ops/utils.hpp
#pragma once
#include <torch/extension.h>

namespace tt_eager::ops::utils {

// VIEW / RESHAPE
at::Tensor& ttnn_view_out   (const at::Tensor& self, at::IntArrayRef shape, at::Tensor& out);
at::Tensor  ttnn_view       (const at::Tensor& self, at::IntArrayRef shape);

// PERMUTE / TRANSPOSE
at::Tensor& ttnn_permute_out(const at::Tensor& self, at::IntArrayRef dims, at::Tensor& out);
at::Tensor  ttnn_permute    (const at::Tensor& self, at::IntArrayRef dims);

}  // namespace tt_eager::ops::utils
