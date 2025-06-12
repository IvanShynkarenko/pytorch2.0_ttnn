// ttnn_cpp_extension/ops/linear.cpp
#include <ttnn/operations/linear.hpp>         // for dense or matmul
#include "ttnn_cpp_extension/ops/linear.hpp"
#include "ttnn_cpp_extension/core/TtnnTensorImpl.hpp"

namespace tt_eager::ops::linear {

at::Tensor ttnn_linear(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias_opt) {
    auto inp_impl = static_cast<at::TtnnTensorImpl*>(input.unsafeGetTensorImpl());
    auto w_impl   = static_cast<at::TtnnTensorImpl*>(weight.unsafeGetTensorImpl());
    auto x = inp_impl->get_ttnn_tensor();
    auto w = w_impl->get_ttnn_tensor();

    // matmul or dense
    auto y = ttnn::matmul(x, ttnn::transpose(w, {1,0}));

    // add bias if present
    if (bias_opt.has_value()) {
        auto b_impl = static_cast<at::TtnnTensorImpl*>(bias_opt->unsafeGetTensorImpl());
        auto b = b_impl->get_ttnn_tensor();
        y = ttnn::add_scalar(y, b);  // broadcast-add
    }

    auto result = at::empty({0}, input.options()).to(input.device());
    auto res_impl = static_cast<at::TtnnTensorImpl*>(result.unsafeGetTensorImpl());
    res_impl->set_sizes_and_strides_as(input);
    res_impl->set_ttnn_tensor(y);
    return result;
}

}  // namespace tt_eager::ops::linear
