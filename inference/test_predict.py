import torch
from inference.predict import predict_with_severity

dummy_input = torch.randn(1, 3, 224, 224)
result = predict_with_severity(dummy_input)
print(result)
