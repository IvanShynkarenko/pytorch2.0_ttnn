from transformers import BertTokenizer, BertModel
import torch
from torch.fx import symbolic_trace

print("🚀 Starting tracing")

# 1. Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("hello world", return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 2. Model
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# 3. Функція-обгортка
class MyWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

wrapped_model = MyWrapper(model)

# 4. Tracing із конкретними аргументами
traced = symbolic_trace(
    wrapped_model,
    concrete_args={
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
)

# 5. Print ops
print("\n🔍 Operations used in forward():")
for node in traced.graph.nodes:
    print(f"{node.op:10s} {node.name:20s} {node.target}")

print("\n✅ Done")
