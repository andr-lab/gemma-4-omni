#!/usr/bin/env python3
import inspect
from transformers.cache_utils import StaticLayer

print(inspect.getsource(StaticLayer.update))
