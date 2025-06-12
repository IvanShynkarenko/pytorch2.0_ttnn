#include <ttnn/operations/eltwise/unary/unary.hpp>

#include "ttnn_cpp_extension/ops/unary.hpp"

#include "ttnn_cpp_extension/core/TtnnTensorImpl.hpp"

#include "ttnn_cpp_extension/utils/extension_utils.hpp"
#include <torch/extension.h>                         // щоб бачити at::empty, at::TensorOptions, тощо
#include "ttnn_cpp_extension/ops/creation.hpp" 

namespace tt_eager::ops::unary {
at::Tensor& ttnn_abs_out(const at::Tensor& self, at::Tensor& out) {
    LOGGING("");
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1);
    TORCH_CHECK(out.device().type() == c10::DeviceType::PrivateUse1);

    // Get underlying TTNN tensor object from input
    at::TtnnTensorImpl* tensor_impl = static_cast<at::TtnnTensorImpl*>(self.unsafeGetTensorImpl());
    auto ttnn_tensor = tensor_impl->get_ttnn_tensor();

    // Call TTNN operation
    auto result = ttnn::abs(ttnn_tensor);

    // Get underlying TTNN tensor object from output
    at::TtnnTensorImpl* out_tensor_impl = static_cast<at::TtnnTensorImpl*>(out.unsafeGetTensorImpl());
    out_tensor_impl->set_sizes_and_strides_as(self);

    // Set output TTNN tensor to result
    auto out_ttnn_tensor = out_tensor_impl->get_ttnn_tensor();
    out_tensor_impl->set_ttnn_tensor(result);

    return out;
}


at::Tensor& ttnn_gelu_out(const at::Tensor& self, at::Tensor& out) {
    LOGGING("");
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1);
    TORCH_CHECK(out.device().type() == c10::DeviceType::PrivateUse1);

    // Get underlying TTNN tensor object from input
    at::TtnnTensorImpl* self_impl = static_cast<at::TtnnTensorImpl*>(self.unsafeGetTensorImpl());
    auto ttnn_tensor = self_impl->get_ttnn_tensor();

    // Call GELU
    auto result = ttnn::gelu(ttnn_tensor);

    // Set output TTNN tensor to result
    at::TtnnTensorImpl* out_impl = static_cast<at::TtnnTensorImpl*>(out.unsafeGetTensorImpl());
    out_impl->set_sizes_and_strides_as(self);
    out_impl->set_ttnn_tensor(result);

    return out;
}

at::Tensor ttnn_gelu(const at::Tensor& self) {
    LOGGING("Allocating new tensor for GELU");
    auto out = tt_eager::ops::create::custom_empty_memory_format(
        self.sizes(),
        c10::optional<at::ScalarType>(self.scalar_type()),
        c10::nullopt,
        c10::optional<at::Device>(self.device()),
        c10::nullopt
    );
    return ttnn_gelu_out(self, out);
}

at::Tensor& ttnn_tanh_out(const at::Tensor& self, at::Tensor& out) {
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1);
    auto inp_impl = static_cast<at::TtnnTensorImpl*>(self.unsafeGetTensorImpl());
    auto t = inp_impl->get_ttnn_tensor();
    auto y = ttnn::tanh(t);
    auto out_impl = static_cast<at::TtnnTensorImpl*>(out.unsafeGetTensorImpl());
    out_impl->set_sizes_and_strides_as(self);
    out_impl->set_ttnn_tensor(y);
    return out;
}

at::Tensor ttnn_tanh(const at::Tensor& self) {
    auto out = at::empty(self.sizes(), self.options()).to(self.device());
    return ttnn_tanh_out(self, out);
}
}  // namespace tt_eager::ops::unary
