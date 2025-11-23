import torch
from dit import DiT

model = DiT()

print(sum(p.numel() for p in model.parameters() if p.requires_grad))