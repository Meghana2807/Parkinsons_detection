import torch
from model import CNNTransformer

model = CNNTransformer()
dummy_input = torch.randn(1, 3, 224, 224)

output = model(dummy_input)
print("Output shape:", output.shape)
