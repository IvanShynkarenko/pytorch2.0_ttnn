from transformers import BertTokenizer, BertModel
import torch
from torch.fx import symbolic_trace
from collections import defaultdict

# 1. Load tokenizer and inputs
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("hello world", return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 2. Wrap model
class MyWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

# 3. Trace
model = BertModel.from_pretrained("bert-base-uncased").eval()
traced = symbolic_trace(MyWrapper(model), concrete_args={
    "input_ids": input_ids,
    "attention_mask": attention_mask
})

# 4. Collect unique ops
unique_ops = defaultdict(set)

for node in traced.graph.nodes:
    if node.op == "call_function":
        unique_ops["call_function"].add(str(node.target))
    elif node.op == "call_method":
        unique_ops["call_method"].add(str(node.target))
    elif node.op == "call_module":
        mod_type = type(dict(traced.named_modules())[node.target]).__name__
        unique_ops["call_module"].add(f"{node.target} ({mod_type})")

# 5. Print summary
print("\n✅ Unique PyTorch Ops Used:")
for op_type, targets in unique_ops.items():
    print(f"\n🔹 {op_type.upper()}:")
    for t in sorted(targets):
        print(f"  - {t}")
