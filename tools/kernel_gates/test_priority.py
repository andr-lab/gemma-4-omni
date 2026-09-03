import torch
s = torch.cuda.Stream(priority=-1)
print("CUDA Stream priority:", s.priority)
