import ctypes
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from duplex_kernel.microbatch.probe_native_listener_capacity_parity import (
    _copy_cached_kv_from_native,
    _load_cuda_runtime_for_cache_copy,
)


class _FakeCudaMemcpy:
    def __init__(self, result: int = 0) -> None:
        self.result = result
        self.calls: list[tuple[object, ...]] = []
        self.argtypes = None
        self.restype = None

    def __call__(self, *args: object) -> int:
        self.calls.append(args)
        return self.result


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...], pointer: int) -> None:
        self.shape = shape
        self.pointer = pointer
        self.device = "cuda:0"

    def data_ptr(self) -> int:
        return self.pointer


class _FakeTorch:
    bfloat16 = "bf16"

    def __init__(self) -> None:
        self.allocations: list[_FakeTensor] = []

    def empty(self, shape: tuple[int, ...], *, dtype: object, device: object) -> _FakeTensor:
        assert dtype == self.bfloat16
        assert device == "cuda:0"
        tensor = _FakeTensor(shape, 1000 + len(self.allocations))
        self.allocations.append(tensor)
        return tensor


class _FakeNative:
    def __init__(self, layer_type: str = "sliding") -> None:
        self._ctx = 77
        self.current_position = 3
        self.output_tensor = SimpleNamespace(device="cuda:0")
        self.language_model = SimpleNamespace(
            config=SimpleNamespace(layer_types=[layer_type]),
        )
        self._lib = SimpleNamespace(
            gemma4_listener_get_cached_k=lambda ctx, layer: 0x1000 + layer.value,
            gemma4_listener_get_cached_v=lambda ctx, layer: 0x2000 + layer.value,
        )


def test_versioned_runtime_is_resolved_and_cache_copy_uses_cuda_memcpy(tmp_path: Path) -> None:
    torch_root = tmp_path / "torch"
    runtime_path = torch_root / "lib" / "libcudart.so.13"
    runtime_path.parent.mkdir(parents=True)
    runtime_path.write_bytes(b"test runtime placeholder")
    cuda_memcpy = _FakeCudaMemcpy()
    loaded_names: list[str] = []

    def fake_cdll(name: str) -> SimpleNamespace:
        loaded_names.append(name)
        return SimpleNamespace(cudaMemcpy=cuda_memcpy)

    torch = _FakeTorch()
    native = _FakeNative()
    runtime, selected_path, selected_sha256 = _load_cuda_runtime_for_cache_copy(
        SimpleNamespace(__file__=str(torch_root / "__init__.py")),
        find_library=lambda name: str(runtime_path) if name == "cudart" else None,
        cdll_factory=fake_cdll,
        maps_path=tmp_path / "missing-maps",
    )
    keys, values = _copy_cached_kv_from_native(torch, native, 0, runtime.cudaMemcpy)

    assert selected_path == str(runtime_path.resolve())
    assert selected_sha256 == hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    assert loaded_names == [str(runtime_path.resolve())]
    assert keys.shape == values.shape == (1, 1, 3, 256)
    assert cuda_memcpy.argtypes == [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    assert cuda_memcpy.restype is ctypes.c_int
    assert len(cuda_memcpy.calls) == 2
    assert [call[3].value for call in cuda_memcpy.calls] == [3, 3]
    assert [call[2].value for call in cuda_memcpy.calls] == [3 * 256 * 2, 3 * 256 * 2]


def test_versioned_runtime_is_found_in_torch_cuda_package_path(tmp_path: Path) -> None:
    torch_root = tmp_path / "site-packages" / "torch"
    runtime_path = torch_root / "lib" / "libcudart.so.13"
    runtime_path.parent.mkdir(parents=True)
    runtime_path.write_bytes(b"test runtime placeholder")
    loaded_names: list[str] = []

    def fake_cdll(name: str) -> SimpleNamespace:
        loaded_names.append(name)
        return SimpleNamespace(cudaMemcpy=_FakeCudaMemcpy())

    _runtime, selected_path, _selected_sha256 = _load_cuda_runtime_for_cache_copy(
        SimpleNamespace(__file__=str(torch_root / "__init__.py")),
        find_library=lambda _name: None,
        cdll_factory=fake_cdll,
        maps_path=tmp_path / "missing-maps",
    )
    assert selected_path == str(runtime_path.resolve())
    assert loaded_names == [str(runtime_path.resolve())]


def test_unversioned_global_alias_is_not_loaded_or_synthesized(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="versioned libcudart runtime"):
        _load_cuda_runtime_for_cache_copy(
            SimpleNamespace(__file__=str(tmp_path / "torch" / "__init__.py")),
            find_library=lambda _name: "libcudart.so",
            cdll_factory=lambda _name: pytest.fail("must not load an unversioned alias"),
            maps_path=tmp_path / "missing-maps",
        )


def test_bare_versioned_soname_requires_an_absolute_installed_path(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="versioned libcudart runtime"):
        _load_cuda_runtime_for_cache_copy(
            SimpleNamespace(__file__=str(tmp_path / "torch" / "__init__.py")),
            find_library=lambda _name: "libcudart.so.13",
            cdll_factory=lambda _name: pytest.fail("must not load a bare versioned SONAME"),
            maps_path=tmp_path / "missing-maps",
        )


def test_absolute_find_library_file_outside_mapped_or_packaged_paths_is_rejected(tmp_path: Path) -> None:
    external = tmp_path / "elsewhere" / "libcudart.so.13"
    external.parent.mkdir()
    external.write_bytes(b"existing but untrusted runtime")
    with pytest.raises(OSError, match="outside the mapped/packaged"):
        _load_cuda_runtime_for_cache_copy(
            SimpleNamespace(__file__=str(tmp_path / "torch" / "__init__.py")),
            find_library=lambda _name: str(external),
            cdll_factory=lambda _name: pytest.fail("outside runtime path must not be loaded"),
            maps_path=tmp_path / "missing-maps",
        )


def test_mapped_torch_runtime_is_selected_by_absolute_path_and_version(tmp_path: Path) -> None:
    mapped = tmp_path / "libcudart.so.13.0.48"
    mapped.write_bytes(b"mapped torch CUDA runtime")
    maps = tmp_path / "maps"
    maps.write_text(f"7f000000-7f100000 r-xp 00000000 00:00 0 {mapped}\n", encoding="utf-8")
    loaded: list[str] = []

    def fake_cdll(name: str) -> SimpleNamespace:
        loaded.append(name)
        return SimpleNamespace(cudaMemcpy=_FakeCudaMemcpy())

    runtime, selected_path, _selected_sha256 = _load_cuda_runtime_for_cache_copy(
        SimpleNamespace(),
        find_library=lambda _name: "libcudart.so.13",
        cdll_factory=fake_cdll,
        maps_path=maps,
    )
    assert selected_path == str(mapped.resolve())
    assert loaded == [str(mapped.resolve())]
    assert runtime.cudaMemcpy.argtypes is not None

    wrong = tmp_path / "libcudart.so.12.9.0"
    wrong.write_bytes(b"different CUDA runtime")
    with pytest.raises(OSError, match="does not match"):
        _load_cuda_runtime_for_cache_copy(
            SimpleNamespace(),
            find_library=lambda _name: str(wrong),
            cdll_factory=lambda _name: pytest.fail("must not load a mismatched CUDA runtime"),
            maps_path=maps,
        )


def test_multiple_package_runtime_matches_fail_closed_unless_absolute_path_selects_one(tmp_path: Path) -> None:
    torch_root = tmp_path / "site-packages" / "torch"
    first = torch_root / "lib" / "libcudart.so.13.0.1"
    second = tmp_path / "site-packages" / "nvidia" / "cuda_runtime" / "lib" / "libcudart.so.13.0.2"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"runtime 13.0.1")
    second.write_bytes(b"runtime 13.0.2")

    with pytest.raises(OSError, match="matches multiple installed CUDA runtimes"):
        _load_cuda_runtime_for_cache_copy(
            SimpleNamespace(__file__=str(torch_root / "__init__.py")),
            find_library=lambda _name: "libcudart.so.13",
            cdll_factory=lambda _name: pytest.fail("ambiguous runtime must not be loaded"),
            maps_path=tmp_path / "missing-maps",
        )

    loaded: list[str] = []

    def fake_cdll(name: str) -> SimpleNamespace:
        loaded.append(name)
        return SimpleNamespace(cudaMemcpy=_FakeCudaMemcpy())

    _runtime, selected_path, _selected_sha256 = _load_cuda_runtime_for_cache_copy(
        SimpleNamespace(__file__=str(torch_root / "__init__.py")),
        find_library=lambda _name: str(first),
        cdll_factory=fake_cdll,
        maps_path=tmp_path / "missing-maps",
    )
    assert selected_path == str(first.resolve())
    assert loaded == [str(first.resolve())]


def test_native_cache_copy_fails_closed_on_cuda_memcpy_error() -> None:
    torch = _FakeTorch()
    native = _FakeNative(layer_type="full_attention")
    with pytest.raises(RuntimeError, match="cudaMemcpy K failed with code 11"):
        _copy_cached_kv_from_native(torch, native, 0, _FakeCudaMemcpy(result=11))
    assert torch.allocations[0].shape == torch.allocations[1].shape == (1, 1, 3, 512)
