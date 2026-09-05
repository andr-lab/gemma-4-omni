#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import inspect
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

fn = ROPE_INIT_FUNCTIONS["proportional"]
print("ROPE_INIT_FUNCTIONS['proportional'] source:")
print(inspect.getsource(fn))
