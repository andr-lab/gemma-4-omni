import torch

ck = torch.load('/workspace/cache_ars_full_v2.pt', map_location='cpu', weights_only=False, mmap=True)

print("tr_raw[0] shape:", ck['tr_raw'][0].shape, "norm:", torch.norm(ck['tr_raw'][0], dim=-1).mean().item())
print("tr_htx[0] shape:", ck['tr_htx'][0].shape, "norm:", torch.norm(ck['tr_htx'][0]).item())
print("tr_sub[0] shape:", ck['tr_sub'][0].shape, "norm:", torch.norm(ck['tr_sub'][0], dim=-1).mean().item())
print("tr_c[0] shape:", ck['tr_c'][0].shape, "unique:", len(torch.unique(ck['tr_c'][0])))

# Check if tr_htx varies across windows
norms_htx = [torch.norm(ck['tr_htx'][i]).item() for i in range(10)]
print("First 10 htx norms:", norms_htx)
