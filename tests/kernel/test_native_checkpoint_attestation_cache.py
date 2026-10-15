from __future__ import annotations

import hashlib
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from duplex_kernel import native_listener


class NativeCheckpointAttestationCacheTest(unittest.TestCase):
    def test_successful_checkpoint_hash_is_reused_for_same_file_signature(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            checkpoint = tmp_path / "model.safetensors"
            payload = b"small checkpoint test payload"
            checkpoint.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            manifest = {
                "checkpoint_path": str(tmp_path),
                "checkpoint_files": {checkpoint.name: expected},
            }

            real_sha256 = hashlib.sha256
            hash_calls = 0

            def counted_sha256(*args, **kwargs):
                nonlocal hash_calls
                hash_calls += 1
                return real_sha256(*args, **kwargs)

            with patch.object(native_listener.hashlib, "sha256", side_effect=counted_sha256):
                self.assertEqual(native_listener._verify_attested_checkpoint(manifest), tmp_path.resolve())
                self.assertEqual(native_listener._verify_attested_checkpoint(manifest), tmp_path.resolve())
            self.assertEqual(hash_calls, 1)

    def test_checkpoint_cache_invalidates_metadata_and_never_caches_hash_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            checkpoint = tmp_path / "model.safetensors"
            checkpoint.write_bytes(b"first payload")
            expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            manifest = {"checkpoint_path": str(tmp_path), "checkpoint_files": {checkpoint.name: expected}}

            real_sha256 = hashlib.sha256
            hash_calls = 0

            def counted_sha256(*args, **kwargs):
                nonlocal hash_calls
                hash_calls += 1
                return real_sha256(*args, **kwargs)

            with patch.object(native_listener.hashlib, "sha256", side_effect=counted_sha256):
                native_listener._verify_attested_checkpoint(manifest)

                checkpoint.write_bytes(b"second payload with new size")
                stat = checkpoint.stat()
                os.utime(checkpoint, ns=(stat.st_atime_ns, max(time.time_ns(), stat.st_mtime_ns + 1_000_000)))
                updated = checkpoint.read_bytes()
                manifest["checkpoint_files"][checkpoint.name] = real_sha256(updated).hexdigest()
                native_listener._verify_attested_checkpoint(manifest)
                self.assertEqual(hash_calls, 2)

                bad_path = tmp_path / "bad.safetensors"
                bad_path.write_bytes(b"unattested payload")
                bad_manifest = {
                    "checkpoint_path": str(tmp_path),
                    "checkpoint_files": {bad_path.name: "0" * 64},
                }
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError, "hash checkpoint non valido"):
                        native_listener._verify_attested_checkpoint(bad_manifest)
                self.assertEqual(hash_calls, 4)

                bad_manifest["checkpoint_files"][bad_path.name] = real_sha256(bad_path.read_bytes()).hexdigest()
                native_listener._verify_attested_checkpoint(bad_manifest)
                self.assertEqual(hash_calls, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
