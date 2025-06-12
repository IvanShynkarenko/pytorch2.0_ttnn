#include <ATen/native/DispatchStub.h>
#include <torch/extension.h>               // охоплює і pybind

#include "ttnn_cpp_extension/utils/device.hpp"
#include "ttnn_cpp_extension/core/TtnnCustomAllocator.hpp"
#include "ttnn_cpp_extension/core/copy.hpp"

// ==== ops ====
#include "ttnn_cpp_extension/ops/creation.hpp"
#include "ttnn_cpp_extension/ops/unary.hpp"
#include "ttnn_cpp_extension/ops/binary.hpp"
#include "ttnn_cpp_extension/ops/utils.hpp"
#include "ttnn_cpp_extension/ops/attn.hpp"
#include "ttnn_cpp_extension/ops/linear.hpp"
#include "ttnn_cpp_extension/ops/norm.hpp"
#include "ttnn_cpp_extension/ops/dropout.hpp"


// Register custom allocator. Only used to create dummy Torch tensor object.
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &get_ttnn_custom_allocator());

// Register all kernels for the PrivateUse1 backend
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // creation
    m.impl("aten::empty_strided",             &tt_eager::ops::create::custom_empty_strided);
    m.impl("empty.memory_format",             &tt_eager::ops::create::custom_empty_memory_format);

    // copy
    m.impl("_copy_from",                      &ttnn_copy_from);

    // unary
    m.impl("abs.out",                         &tt_eager::ops::unary::ttnn_abs_out);
    m.impl("gelu",                            &tt_eager::ops::unary::ttnn_gelu);
    m.impl("gelu.out",                        &tt_eager::ops::unary::ttnn_gelu_out);
    m.impl("tanh",                            &tt_eager::ops::unary::ttnn_tanh);
    m.impl("tanh.out",                        &tt_eager::ops::unary::ttnn_tanh_out);

    // binary
    m.impl("add.out",                         &tt_eager::ops::binary::ttnn_add_out);
    m.impl("add.Tensor",                      &tt_eager::ops::binary::ttnn_add_tensor);

    // utils: view/reshape
    m.impl("aten::view",                      &tt_eager::ops::utils::ttnn_view);
    m.impl("aten::view.out",                  &tt_eager::ops::utils::ttnn_view_out);
    m.impl("aten::reshape",                   &tt_eager::ops::utils::ttnn_view);
    m.impl("aten::reshape.out",               &tt_eager::ops::utils::ttnn_view_out);

    // utils: permute/transpose
    m.impl("aten::permute",                   &tt_eager::ops::utils::ttnn_permute);
    m.impl("aten::permute.out",               &tt_eager::ops::utils::ttnn_permute_out);
    m.impl("aten::transpose",                 &tt_eager::ops::utils::ttnn_permute);
    m.impl("aten::transpose.out",             &tt_eager::ops::utils::ttnn_permute_out);

    // attention
    m.impl("aten::scaled_dot_product_attention",
           &tt_eager::ops::attn::ttnn_scaled_dot_attn);

    // linear / dense
    m.impl("aten::linear",                    &tt_eager::ops::linear::ttnn_linear);
    // (if you later implement an out-variant)
    // m.impl("aten::linear.out",             &tt_eager::ops::linear::ttnn_linear_out);

    // normalization
    m.impl("aten::layer_norm",                &tt_eager::ops::norm::ttnn_layer_norm);

    // dropout (inference‐only)
    m.impl("aten::dropout",                   &tt_eager::ops::dropout::ttnn_dropout);
}

// Python bindings for device control
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("as_torch_device",   &as_torch_device,   "get torch device from existing ttnn device");
    m.def("get_ttnn_tensor",   &get_ttnn_tensor,   "open ttnn device and get torch tensor");
    m.def("open_torch_device", &open_torch_device, "alias for as_torch_device");
    m.def("close_torch_device",&close_torch_device,"close torch device and associated ttnn device");
}
