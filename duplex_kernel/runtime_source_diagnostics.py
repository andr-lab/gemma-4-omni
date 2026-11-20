"""Read-only diagnostics for the Python sources used by a live launch."""
from __future__ import annotations

from importlib import metadata
from pathlib import Path
import sys
from typing import Mapping


_SOURCE_KEYS = (
    "server",
    "listener",
    "frontend",
    "worker",
    "session",
    "native_binding",
)
_DISTRIBUTIONS = ("torch", "transformers", "tokenizers")


def _resolved_module_path(module: object | None) -> str | None:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, (str, Path)) or not str(module_file):
        return None
    return str(Path(module_file).expanduser().resolve())


def _distribution_version(distribution: str) -> str | None:
    imported_module = sys.modules.get(distribution)
    imported_version = getattr(imported_module, "__version__", None)
    if isinstance(imported_version, str) and imported_version:
        # Distribution metadata may omit local build tags such as ``+cu130``.
        # The imported module is authoritative for the runtime actually serving.
        return imported_version
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def collect_runtime_source_diagnostics(
    modules: Mapping[str, object],
) -> dict[str, object]:
    """Return resolved imported module paths and installed package versions.

    The caller supplies the modules it already imported. This function does
    not import model, framework, or tokenizer packages to discover them.
    """
    diagnostics: dict[str, object] = {
        key: _resolved_module_path(modules.get(key))
        for key in _SOURCE_KEYS
    }
    diagnostics["python_executable"] = str(Path(sys.executable).resolve())
    diagnostics["versions"] = {
        distribution: _distribution_version(distribution)
        for distribution in _DISTRIBUTIONS
    }
    return diagnostics
