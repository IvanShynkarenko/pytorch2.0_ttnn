// ttnn_cpp_extension/ops/utils.cpp
#include <ttnn/operations/transform.hpp>             // for reshape, transpose
#include "ttnn_cpp_extension/ops/utils.hpp"
#include "ttnn_cpp_extension/core/TtnnTensorImpl.hpp"

namespace tt_eager::ops::utils {

// VIEW / RESHAPE

at::Tensor& ttnn_view_out(const at::Tensor& self, at::IntArrayRef shape, at::Tensor& out) {
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1);
    auto impl = static_cast<at::TtnnTensorImpl*>(self.unsafeGetTensorImpl());
    auto t = impl->get_ttnn_tensor();
    auto result = ttnn::reshape(t, ttnn::Shape(std::vector<uint32_t>(shape.begin(), shape.end())));
    auto out_impl = static_cast<at::TtnnTensorImpl*>(out.unsafeGetTensorImpl());
    out_impl->set_sizes_and_strides_as(self);
    out_impl->set_ttnn_tensor(result);
    return out;
}

at::Tensor ttnn_view(const at::Tensor& self, at::IntArrayRef shape) {
    auto out = at::empty(self.sizes(), self.options()).to(self.device());
    return ttnn_view_out(self, shape, out);
}

// PERMUTE / TRANSPOSE

at::Tensor& ttnn_permute_out(const at::Tensor& self, at::IntArrayRef dims, at::Tensor& out) {
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1);
    auto impl = static_cast<at::TtnnTensorImpl*>(self.unsafeGetTensorImpl());
    auto t = impl->get_ttnn_tensor();
    auto result = ttnn::transpose(t, std::vector<uint32_t>(dims.begin(), dims.end()));
    auto out_impl = static_cast<at::TtnnTensorImpl*>(out.unsafeGetTensorImpl());
    out_impl->set_sizes_and_strides_as(self);
    out_impl->set_ttnn_tensor(result);
    return out;
}

at::Tensor ttnn_permute(const at::Tensor& self, at::IntArrayRef dims) {
    auto out = at::empty(self.sizes(), self.options()).to(self.device());
    return ttnn_permute_out(self, dims, out);
}

}  // namespace tt_eager::ops::utils
