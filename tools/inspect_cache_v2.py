import torch

ck = torch.load('/workspace/cache_ars_full_v2.pt', map_location='cpu', weights_only=False, mmap=True)
print("Keys in cache_ars_full_v2.pt:", list(ck.keys()))

for k in ck.keys():
    v = ck[k]
    if isinstance(v, (list, tuple)):
        print(f"Key '{k}': list/tuple of length {len(v)}, first item type: {type(v[0])}")
        if len(v) > 0 and hasattr(v[0], 'shape'):
            print(f"  first item shape: {v[0].shape}, dtype: {v[0].dtype}")
        elif len(v) > 0 and isinstance(v[0], str):
            print(f"  first item sample text: {repr(v[0][:80])}")
    elif hasattr(v, 'shape'):
        print(f"Key '{k}': tensor/array shape {v.shape}, dtype {v.dtype}")
    else:
        print(f"Key '{k}': type {type(v)}")
