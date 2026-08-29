import os, sys

target_path = "/workspace/qwen3tts_src/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"
if not os.path.exists(target_path):
    print(f"Error: {target_path} not found")
    sys.exit(1)

with open(target_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace the direct import with the universal wrapper
target_import = "from transformers.utils.generic import check_model_inputs"
universal_def = """from transformers.utils.generic import check_model_inputs as _orig_check_model_inputs
import inspect

def check_model_inputs(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        func = args[0]
        sig = inspect.signature(_orig_check_model_inputs)
        if "func" in sig.parameters:
            return _orig_check_model_inputs(func)
        else:
            return _orig_check_model_inputs()(func)
    sig = inspect.signature(_orig_check_model_inputs)
    if "func" in sig.parameters:
        return lambda f: _orig_check_model_inputs(f)
    else:
        return _orig_check_model_inputs(*args, **kwargs)
"""

if target_import in content:
    content = content.replace(target_import, universal_def)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Successfully patched {target_path} with universal check_model_inputs wrapper.")
else:
    print(f"Warning: '{target_import}' not found (already patched?)")
