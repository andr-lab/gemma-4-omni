import torch

print("allow_bf16_reduced_precision_reduction:", getattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", None))
print("allow_tf32:", torch.backends.cuda.matmul.allow_tf32)
