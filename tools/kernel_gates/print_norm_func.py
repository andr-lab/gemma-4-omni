#!/usr/bin/env python3
import inspect
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

print(inspect.getsource(Gemma4RMSNorm._norm))
