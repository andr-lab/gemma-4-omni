"""Sample runtime allocator and branch state for acceptance sidecars."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping
from urllib.request import Request, urlopen

SAMPLE_INTERVAL_SECONDS = 0.5
MIB = 1024.0 * 1024.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def http_json(url: str, timeout_s: float = 5.0) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{url} returned a non-object JSON response")
    return payload


def compact_sample(payload: Mapping[str, Any], sampled_at_utc: str) -> dict[str, Any]:
    allocator = payload.get("cuda_allocator")
    allocator = allocator if isinstance(allocator, Mapping) else {}
    sessions_value = payload.get("sessions")
    sessions: list[dict[str, Any]] = []
    if isinstance(sessions_value, list):
        for session in sessions_value:
            if not isinstance(session, Mapping):
                continue
            branches_value = session.get("branches")
            branches: list[dict[str, Any]] = []
            if isinstance(branches_value, list):
                for branch in branches_value:
                    if not isinstance(branch, Mapping):
                        continue
                    compact_branch = {
                        key: branch.get(key)
                        for key in (
                            "response_id", "generation_id", "speaker_segment_id",
                            "playback_pending", "playback_cancelled", "finalized",
                            "ended_response", "finalizer_task_status",
                            "client_consumed_segments", "spoken_prefix_exclusive_cursor",
                            "kv_cache_position", "kv_cache_source", "playback_segments",
                        )
                    }
                    branch_allocator = branch.get("allocator")
                    if isinstance(branch_allocator, Mapping):
                        compact_branch["allocator"] = {
                            key: branch_allocator.get(key)
                            for key in (
                                "status", "device", "allocated_bytes", "reserved_bytes",
                                "peak_allocated_bytes", "peak_reserved_bytes",
                            )
                        }
                    if "finalized_at_utc" in branch:
                        compact_branch["finalized_at_utc"] = branch.get("finalized_at_utc")
                    branches.append(compact_branch)
            sessions.append({
                "session_id": session.get("session_id"),
                "response_id": session.get("response_id"),
                "generation_id": session.get("generation_id"),
                "speaker_segment_id": session.get("speaker_segment_id"),
                "active_response": session.get("active_response"),
                "kv_cache_position": session.get("kv_cache_position"),
                "kv_cache_source": session.get("kv_cache_source"),
                "branches": branches,
            })
    return {
        "sampled_at_utc": sampled_at_utc,
        "sampled_at_monotonic_ns": payload.get("sampled_at_monotonic_ns"),
        "status": payload.get("status"),
        "process_id": payload.get("process_id"),
        "cuda_allocator": {
            "status": allocator.get("status"),
            "device": allocator.get("device"),
            "allocated_bytes": allocator.get("allocated_bytes"),
            "reserved_bytes": allocator.get("reserved_bytes"),
            "peak_allocated_bytes": allocator.get("peak_allocated_bytes"),
            "peak_reserved_bytes": allocator.get("peak_reserved_bytes"),
            "sampled_at_monotonic_ns": payload.get("sampled_at_monotonic_ns"),
        },
        "sessions": sessions,
    }


class RuntimeTelemetrySampler:
    """Collect compact allocator/session snapshots without blocking the run."""

    def __init__(self, endpoint: str, output_dir: Path) -> None:
        self.endpoint = endpoint
        self.output_dir = output_dir
        self.samples: list[dict[str, Any]] = []
        self.errors: list[dict[str, str]] = []
        self.post_drain: dict[str, dict[str, Any]] = {}
        self.active: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_raw: dict[str, Any] | None = None
        self.baseline: dict[str, Any] | None = None
        self.postrun_raw: dict[str, Any] | None = None

    def prepare(self) -> None:
        self.baseline_raw = http_json(self.endpoint)
        self.baseline = compact_sample(self.baseline_raw, utc_now())
        if not _valid_allocator(self.baseline["cuda_allocator"]):
            raise RuntimeError(
                "runtime telemetry endpoint did not return measured CUDA allocator counters; "
                "enable DUPLEX_RUNTIME_TELEMETRY=1 and verify the warmed service before acceptance"
            )
        write_json(self.output_dir / "runtime_telemetry_post_warmup_raw.json", self.baseline_raw)

    def start(self) -> None:
        if self.baseline is None:
            raise RuntimeError("prepare() must succeed before the runtime sampler starts")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="acceptance-runtime-telemetry",
            daemon=True,
        )
        self._thread.start()

    def _observe(self, row: dict[str, Any]) -> None:
        allocator = row.get("cuda_allocator") or {}
        allocated_mib = _to_mib(allocator.get("allocated_bytes"))
        reserved_mib = _to_mib(allocator.get("reserved_bytes"))
        for session in row.get("sessions", []):
            branches = session.get("branches", [])
            if self.active is None and (
                session.get("active_response")
                or any(branch.get("playback_pending") for branch in branches)
            ):
                self.active = {
                    "sampled_at_utc": row.get("sampled_at_utc"),
                    "allocated_mib": allocated_mib,
                    "reserved_mib": reserved_mib,
                    "session_id": session.get("session_id"),
                    "response_id": session.get("response_id"),
                    "kv_cache_position": session.get("kv_cache_position"),
                }
            for branch in branches:
                response_id = branch.get("response_id")
                if (
                    not response_id
                    or response_id in self.post_drain
                    or branch.get("finalized") is not True
                    or branch.get("ended_response") is not True
                    or branch.get("playback_pending") is not False
                    or branch.get("finalizer_task_status") != "completed"
                ):
                    continue
                branch_allocator = branch.get("allocator")
                branch_allocator = branch_allocator if isinstance(branch_allocator, Mapping) else {}
                branch_allocated_mib = _to_mib(branch_allocator.get("allocated_bytes"))
                branch_reserved_mib = _to_mib(branch_allocator.get("reserved_bytes"))
                self.post_drain[str(response_id)] = {
                    "sampled_at_utc": branch.get("finalized_at_utc") or row.get("sampled_at_utc"),
                    "allocated_mib": (
                        branch_allocated_mib if branch_allocated_mib is not None else allocated_mib
                    ),
                    "reserved_mib": (
                        branch_reserved_mib if branch_reserved_mib is not None else reserved_mib
                    ),
                    "session_id": session.get("session_id"),
                    **{
                        key: branch.get(key)
                        for key in (
                            "response_id", "generation_id", "speaker_segment_id",
                            "playback_cancelled", "client_consumed_segments",
                            "spoken_prefix_exclusive_cursor", "kv_cache_position",
                            "kv_cache_source",
                        )
                    },
                }

    def _sample_loop(self) -> None:
        samples_path = self.output_dir / "runtime_telemetry_samples.jsonl"
        with samples_path.open("w", encoding="utf-8", newline="\n") as output:
            while not self._stop.is_set():
                iteration_started = time.monotonic()
                sampled_at = utc_now()
                try:
                    raw = http_json(self.endpoint)
                    row = compact_sample(raw, sampled_at)
                    self.samples.append(row)
                    output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    output.flush()
                    self._observe(row)
                except Exception as exc:  # evidence records sampling failures instead of hiding them
                    self.errors.append({
                        "sampled_at_utc": sampled_at,
                        "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    })
                elapsed = time.monotonic() - iteration_started
                self._stop.wait(max(0.0, SAMPLE_INTERVAL_SECONDS - elapsed))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                self.errors.append({
                    "sampled_at_utc": utc_now(),
                    "error": "runtime telemetry sampler did not stop within 5 seconds",
                })
        try:
            self.postrun_raw = http_json(self.endpoint)
            self._observe(compact_sample(self.postrun_raw, utc_now()))
            write_json(self.output_dir / "runtime_telemetry_post_run_raw.json", self.postrun_raw)
        except Exception as exc:
            self.errors.append({
                "sampled_at_utc": utc_now(),
                "error": f"post-run snapshot: {type(exc).__name__}: {str(exc)[:200]}",
            })

    def sidecar(self, report: Mapping[str, Any] | None) -> dict[str, Any]:
        if self.baseline is None:
            raise RuntimeError("runtime baseline is unavailable")
        report = report or {}
        run_window = report.get("run_window_utc")
        if not isinstance(run_window, Mapping):
            run_window = {}
        start = run_window.get("started_at")
        finish = run_window.get("finished_at")
        connection = report.get("connection")
        connection = connection if isinstance(connection, Mapping) else {}
        session_id = connection.get("session_id")
        all_samples = [self.baseline, *self.samples]
        if self.postrun_raw is not None:
            all_samples.append(compact_sample(self.postrun_raw, utc_now()))
        allocators = [row.get("cuda_allocator") or {} for row in all_samples]
        valid_allocators = [row for row in allocators if _valid_allocator(row)]
        latest = valid_allocators[-1] if valid_allocators else (allocators[-1] if allocators else {})
        current_allocators = [row for row in allocators if _nonnegative_int(row.get("allocated_bytes"))]
        current_reserved = [row for row in allocators if _nonnegative_int(row.get("reserved_bytes"))]
        peak_allocated = [row["peak_allocated_bytes"] for row in allocators if _nonnegative_int(row.get("peak_allocated_bytes"))]
        peak_reserved = [row["peak_reserved_bytes"] for row in allocators if _nonnegative_int(row.get("peak_reserved_bytes"))]
        sampled_ns = latest.get("sampled_at_monotonic_ns")
        if not _nonnegative_int(sampled_ns) or sampled_ns == 0:
            sampled_ns = self.baseline.get("sampled_at_monotonic_ns")
        allocator_sidecar = {
            "status": latest.get("status"),
            "device": latest.get("device"),
            "allocated_bytes": latest.get("allocated_bytes"),
            "reserved_bytes": latest.get("reserved_bytes"),
            "peak_allocated_bytes": max(peak_allocated) if peak_allocated else latest.get("peak_allocated_bytes"),
            "peak_reserved_bytes": max(peak_reserved) if peak_reserved else latest.get("peak_reserved_bytes"),
            "sampled_at_monotonic_ns": sampled_ns,
        }

        baseline_allocator = self.baseline.get("cuda_allocator") or {}
        active = self.active or {
            "allocated_mib": _to_mib(baseline_allocator.get("allocated_bytes")),
            "reserved_mib": _to_mib(baseline_allocator.get("reserved_bytes")),
            "sampled_at_utc": self.baseline.get("sampled_at_utc"),
            "basis": "no_active_snapshot_observed",
        }
        max_allocated = max(
            (int(row["allocated_bytes"]) for row in current_allocators),
            default=None,
        )
        max_reserved = max(
            (int(row["reserved_bytes"]) for row in current_reserved),
            default=None,
        )

        scenario = report.get("scenario")
        response_rows = scenario.get("responses", []) if isinstance(scenario, Mapping) else []
        tag_by_id = {
            str(row.get("response_id")): str(row.get("scenario_tag"))
            for row in response_rows
            if isinstance(row, Mapping)
            and row.get("response_id")
            and row.get("scenario_tag")
            and row.get("scenario_tag") != "unassigned"
        } if isinstance(response_rows, list) else {}
        vram_turns: list[dict[str, Any]] = []
        kv_turns: list[dict[str, Any]] = []
        for response_id, snapshot in self.post_drain.items():
            tag = tag_by_id.get(response_id)
            if tag is None:
                continue
            vram_turns.append({
                "response_id": response_id,
                "scenario_tag": tag,
                "post_drain": {
                    "allocated_mib": snapshot.get("allocated_mib"),
                    "reserved_mib": snapshot.get("reserved_mib"),
                },
            })
            kv_turns.append({
                "response_id": response_id,
                "scenario_tag": tag,
                **{
                    key: snapshot.get(key)
                    for key in (
                        "session_id", "generation_id", "speaker_segment_id",
                        "client_consumed_segments", "spoken_prefix_exclusive_cursor",
                        "kv_cache_position", "kv_cache_source", "playback_cancelled",
                    )
                },
            })

        return {
            "schema": "duplex_runtime_telemetry_v1",
            "session_id": session_id,
            "run_window_utc": {"started_at": start, "finished_at": finish},
            "sample_interval_ms": int(SAMPLE_INTERVAL_SECONDS * 1000),
            "cuda_allocator": allocator_sidecar,
            "kv": {
                "source": "/debug/runtime-telemetry host allocator/branch snapshot",
                "turns": kv_turns,
                "sample_count": len(self.samples),
                "post_drain_response_ids": sorted(self.post_drain),
            },
            "vram": {
                "post_warmup_baseline": {
                    "allocated_mib": _to_mib(baseline_allocator.get("allocated_bytes")),
                    "reserved_mib": _to_mib(baseline_allocator.get("reserved_bytes")),
                    "sampled_at_monotonic_ns": self.baseline.get("sampled_at_monotonic_ns"),
                },
                "active": {
                    "allocated_mib": active.get("allocated_mib"),
                    "reserved_mib": active.get("reserved_mib"),
                },
                "peak": {
                    "allocated_mib": _to_mib(max_allocated),
                    "reserved_mib": _to_mib(max_reserved),
                    "basis": (
                        "maximum current allocator value observed by "
                        f"{int(SAMPLE_INTERVAL_SECONDS * 1000)}ms telemetry samples during this run"
                    ),
                    "allocator_process_high_water_after_run": {
                        "allocated_mib": _to_mib(max(peak_allocated) if peak_allocated else None),
                        "reserved_mib": _to_mib(max(peak_reserved) if peak_reserved else None),
                    },
                },
                "turns": vram_turns,
            },
            "collection": {
                "sample_errors": self.errors,
                "sample_count": len(self.samples),
                "post_drain_snapshots_count": len(self.post_drain),
                "sampler_errors": len(self.errors),
                "runtime_endpoint": "GET /debug/runtime-telemetry",
                "allocator_source": "torch.cuda memory_allocated/reserved and max counters; no synchronize or tensor readback",
            },
        }


def _to_mib(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) / MIB


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_allocator(row: Mapping[str, Any]) -> bool:
    values = (
        row.get("allocated_bytes"), row.get("reserved_bytes"),
        row.get("peak_allocated_bytes"), row.get("peak_reserved_bytes"),
        row.get("sampled_at_monotonic_ns"),
    )
    return (
        row.get("status") in {"ok", "available", "measured"}
        and isinstance(row.get("device"), str)
        and row["device"].lower().startswith("cuda")
        and all(_nonnegative_int(value) for value in values)
        and row["sampled_at_monotonic_ns"] > 0
        and row["reserved_bytes"] >= row["allocated_bytes"]
        and row["peak_allocated_bytes"] >= row["allocated_bytes"]
        and row["peak_reserved_bytes"] >= row["reserved_bytes"]
    )


