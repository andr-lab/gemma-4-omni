#!/usr/bin/env python3
import inspect
from transformers import StaticCache

print(inspect.getsource(StaticCache.update))
