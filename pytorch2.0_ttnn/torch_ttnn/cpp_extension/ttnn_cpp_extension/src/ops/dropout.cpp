// ttnn_cpp_extension/ops/dropout.cpp
#include "ttnn_cpp_extension/ops/dropout.hpp"
#include "ttnn_cpp_extension/core/TtnnTensorImpl.hpp"

namespace tt_eager::ops::dropout {

// inference‐only no‐op: just pass through
at::Tensor ttnn_dropout(const at::Tensor& input, double p, bool train) {
    TORCH_CHECK(!train, "Dropout train mode not supported on TT-NN");
    auto inp_impl = static_cast<at::TtnnTensorImpl*>(input.unsafeGetTensorImpl());
    auto t = inp_impl->get_ttnn_tensor();
    // no change
    at::Tensor result = input.clone();  
    auto res_impl = static_cast<at::TtnnTensorImpl*>(result.unsafeGetTensorImpl());
    res_impl->set_ttnn_tensor(t);
    return result;
}

}  // namespace tt_eager::ops::dropout
