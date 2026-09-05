#!/usr/bin/env python3
import json

with open("validation/kernel_gates/P4_NATIVE/GEMMA4_NATIVE_TOPOLOGY.json") as f:
    topo = json.load(f)

for l in topo["layers"]:
    print(f"Layer {l['index']:2d}: gate={l.get('gate_proj_shape')}, k_shared={l.get('is_kv_shared_layer')}, src={l.get('kv_shared_layer_index')}, head_dim={l.get('head_dim')}")
