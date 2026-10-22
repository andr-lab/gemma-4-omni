from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

from duplex_kernel.cpu_audio_archive import CPUAudioArchiveWorker


def test_cpu_archive_worker_is_single_threaded_bounded_and_exhausts_segments(tmp_path) -> None:
    started = threading.Event()
    release = threading.Event()
    factory_calls: list[tuple[str, dict[str, object]]] = []
    active = 0
    max_active = 0
    lock = threading.Lock()

    class Model:
        def transcribe(self, pcm, *, language=None, beam_size=5):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            started.set()
            assert release.wait(timeout=3.0)
            with lock:
                active -= 1
            assert pcm.dtype == np.float32
            assert language is None
            assert beam_size == 5
            return iter([SimpleNamespace(text=" voce"), SimpleNamespace(text=" reale ")]), SimpleNamespace(
                language="it", duration=0.1
            )

    def model_factory(path, **kwargs):
        factory_calls.append((path, kwargs))
        return Model()

    model_dir = tmp_path / "small"
    model_dir.mkdir()
    worker = CPUAudioArchiveWorker(
        model_dir,
        cpu_threads=4,
        max_pending=2,
        model_factory=model_factory,
    )
    try:
        first = worker.submit(np.ones(1600, dtype=np.float32))
        assert first is not None and started.wait(timeout=1.0)
        second = worker.submit(np.ones(1600, dtype=np.float32))
        assert second is not None
        assert worker.submit(np.ones(1600, dtype=np.float32)) is None
        release.set()
        assert first.result(timeout=3.0)["text"] == "voce reale"
        assert second.result(timeout=3.0)["text"] == "voce reale"
        assert len(factory_calls) == 1
        assert factory_calls[0] == (
            str(model_dir),
            {
                "device": "cpu",
                "compute_type": "int8",
                "cpu_threads": 4,
                "num_workers": 1,
                "local_files_only": True,
            },
        )
        assert max_active == 1
    finally:
        release.set()
        worker.shutdown(wait=True)
