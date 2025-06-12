// ttnn_cpp_extension/ops/norm.cpp
#include <ttnn/operations/normalized.hpp>      // for layer_norm
#include "ttnn_cpp_extension/ops/norm.hpp"
#include "ttnn_cpp_extension/core/TtnnTensorImpl.hpp"

namespace tt_eager::ops::norm {

at::Tensor ttnn_layer_norm(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double eps) {

    auto inp_impl = static_cast<at::TtnnTensorImpl*>(input.unsafeGetTensorImpl());
    auto x = inp_impl->get_ttnn_tensor();

    // TTNN layer_norm takes shape and epsilon
    auto y = ttnn::layer_norm(x, std::vector<uint32_t>(normalized_shape.begin(), normalized_shape.end()), eps);

    // apply weight and bias if present
    if (weight_opt.has_value()) {
        auto w_impl = static_cast<at::TtnnTensorImpl*>(weight_opt->unsafeGetTensorImpl());
        y = ttnn::mul(y, w_impl->get_ttnn_tensor());
    }
    if (bias_opt.has_value()) {
        auto b_impl = static_cast<at::TtnnTensorImpl*>(bias_opt->unsafeGetTensorImpl());
        y = ttnn::add(y, b_impl->get_ttnn_tensor());
    }

    auto result = at::empty({0}, input.options()).to(input.device());
    auto res_impl = static_cast<at::TtnnTensorImpl*>(result.unsafeGetTensorImpl());
    res_impl->set_sizes_and_strides_as(input);
    res_impl->set_ttnn_tensor(y);
    return result;
}

}  // namespace tt_eager::ops::norm
