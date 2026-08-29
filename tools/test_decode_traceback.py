import sys, os, inspect, traceback
import torch
import transformers

print(f"=== Testing Universal check_model_inputs on Python {sys.version} ===")
print(f"Transformers version: {transformers.__version__}")

# Universal check_model_inputs implementation
from transformers.utils.generic import check_model_inputs as _orig_cmi

def universal_check_model_inputs(*args, **kwargs):
    # Case 1: called as @check_model_inputs on a function (e.g. @check_model_inputs def forward(...))
    if len(args) == 1 and callable(args[0]) and not kwargs:
        func = args[0]
        sig = inspect.signature(_orig_cmi)
        if "func" in sig.parameters:
            # 5.x style: _orig_cmi(func)
            return _orig_cmi(func)
        else:
            # 4.x style: _orig_cmi()(func)
            return _orig_cmi()(func)
    # Case 2: called as @check_model_inputs(...) with parens/kwargs
    sig = inspect.signature(_orig_cmi)
    if "func" in sig.parameters:
        # 5.x style: ignoring factory kwargs or returning wrapper
        return lambda f: _orig_cmi(f)
    else:
        # 4.x style: _orig_cmi(*args, **kwargs) returns wrapped_fn
        return _orig_cmi(*args, **kwargs)

# Test universal wrapper on dummy class
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type('Cfg', (), {'use_cache': False, 'return_dict': True, 'num_hidden_layers': 1, 'sub_configs': []})()
        
    @universal_check_model_inputs
    def forward_no_parens(self, inputs_embeds=None, use_cache=None, **kwargs):
        return type('Out', (), {'last_hidden_state': inputs_embeds})()

    @universal_check_model_inputs()
    def forward_with_parens(self, inputs_embeds=None, use_cache=None, **kwargs):
        return type('Out', (), {'last_hidden_state': inputs_embeds})()

dm = DummyModel()
t = torch.randn(1, 4, 16)
out1 = dm.forward_no_parens(inputs_embeds=t)
out2 = dm.forward_with_parens(inputs_embeds=t)
print("Dummy model forward_no_parens OK:", out1.last_hidden_state.shape)
print("Dummy model forward_with_parens OK:", out2.last_hidden_state.shape)
print("✅ Universal decorator test PASSED!")
