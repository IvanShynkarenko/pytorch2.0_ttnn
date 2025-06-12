// ttnn_cpp_extension/ops/attn.cpp
#include <cmath>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/linear.hpp>      // for matmul
#include "ttnn_cpp_extension/ops/attn.hpp"
#include "ttnn_cpp_extension/core/TtnnTensorImpl.hpp"

namespace tt_eager::ops::attn {

at::Tensor ttnn_scaled_dot_attn(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor>& mask_opt) {

    // unwrap TTNN tensors
    auto q_impl = static_cast<at::TtnnTensorImpl*>(query.unsafeGetTensorImpl());
    auto k_impl = static_cast<at::TtnnTensorImpl*>(key.unsafeGetTensorImpl());
    auto v_impl = static_cast<at::TtnnTensorImpl*>(value.unsafeGetTensorImpl());
    auto q = q_impl->get_ttnn_tensor();
    auto k = k_impl->get_ttnn_tensor();
    auto v = v_impl->get_ttnn_tensor();

    // QK^T
    auto scores = ttnn::matmul(q, ttnn::transpose(k, {1,0}));

    // scale
    double scale = 1.0 / std::sqrt(static_cast<double>(q.shape()[ -1 ]));
    scores = ttnn::mul_scalar(scores, scale);

    // mask if provided
    if (mask_opt.has_value()) {
        auto mask_impl = static_cast<at::TtnnTensorImpl*>(mask_opt->unsafeGetTensorImpl());
        auto m = mask_impl->get_ttnn_tensor();
        // assume mask is 0/1, expand and subtract large where mask==0
        auto negInf = ttnn::fill(scores.shape(), std::numeric_limits<float>::lowest(), scores.device());
        scores = ttnn::where(m, scores, negInf);
    }

    // softmax
    auto attn = ttnn::softmax(scores, /*axis=*/-1);

    // attn * V
    auto out = ttnn::matmul(attn, v);

    // wrap back into torch
    auto result = query.clone().to(at::kMeta); // placeholder; actual sizes
    auto out_impl = static_cast<at::TtnnTensorImpl*>(result.unsafeGetTensorImpl());
    out_impl->set_sizes_and_strides_as(query);
    out_impl->set_ttnn_tensor(out);
    return result;
}

}  // namespace tt_eager::ops::attn
