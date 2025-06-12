import torch
# no `import ttnn`

x = torch.randn((4,4), dtype=torch.float32)
# use the built-in name for PrivateUse1:
x_tt = x.to("privateuseone:0")

y_tt = torch.nn.functional.gelu(x_tt)
y = y_tt.to("cpu")

expected = torch.nn.functional.gelu(x)
assert torch.allclose(y, expected, atol=1e-3)
print("✅ GELU TTNN matches PyTorch")
