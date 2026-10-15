from __future__ import annotations

import ctypes
from pathlib import Path

import pytest
import torch

from duplex_kernel.native_listener_microbatch import (
    BATCH_API_VERSION,
    BATCH_CAPABILITIES_V1,
    _configure_native_api,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the target CUDA runtime")
def test_native_microbatch_dso_loads_and_advertises_batch_v1() -> None:
    root = Path(__file__).resolve().parents[2]
    library = root / "native" / "gemma4_listener_microbatch" / "libgemma4_listener_microbatch.so"
    assert library.is_file(), f"candidate DSO not built: {library}"

    # Importing torch first matches the production Python process, which has
    # already loaded its ATen/C10 shared libraries before listener activation.
    native = ctypes.CDLL(str(library))
    features = _configure_native_api(native)

    assert int(native.gemma4_listener_abi_version()) == 3
    assert int(native.gemma4_listener_batch_api_version()) == BATCH_API_VERSION
    assert int(native.gemma4_listener_batch_capabilities()) & BATCH_CAPABILITIES_V1 == BATCH_CAPABILITIES_V1
    assert features.trace
