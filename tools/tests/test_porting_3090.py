# test_porting_3090.py - Unit test suite for RTX 3090 porting, paths, and Docker configuration
# Pure ASCII encoding - no non-ASCII characters, no emojis.

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import paths


class TestPorting3090(unittest.TestCase):

    def test_paths_environment_switching(self):
        """Verify environment switching and directory resolution."""
        # 1. Test local3090 environment
        os.environ["PIPELINE_ENV"] = "local3090"
        paths._ROOT_CACHE = None
        paths._BIG_DISK_CACHE = None

        env = paths.get_pipeline_env()
        self.assertEqual(env, "local3090")

        root = paths.get_project_root()
        self.assertTrue(len(root) > 0)
        self.assertTrue(paths.get_data_dir().replace("\\", "/").endswith("data"))
        self.assertTrue(paths.get_checkpoints_dir().replace("\\", "/").endswith("checkpoints"))
        self.assertTrue(paths.get_samples_dir().replace("\\", "/").endswith("samples"))
        self.assertTrue(paths.get_logs_dir().replace("\\", "/").endswith("logs"))
        self.assertTrue(paths.get_subtalker_dir().replace("\\", "/").endswith("qwen_subtalker"))
        self.assertTrue(paths.get_code2wav_dir().replace("\\", "/").endswith("code2wav_weights"))

        # 2. Test Kaggle environment override
        os.environ["PIPELINE_ENV"] = "kaggle"
        paths._ROOT_CACHE = None
        paths._BIG_DISK_CACHE = None

        self.assertEqual(paths.get_pipeline_env(), "kaggle")
        self.assertEqual(paths.get_project_root().replace("\\", "/"), "/kaggle/working")
        self.assertEqual(paths.get_data_dir().replace("\\", "/"), "/kaggle/working/data")
        self.assertEqual(paths.get_checkpoints_dir().replace("\\", "/"), "/kaggle/working/checkpoints")
        self.assertEqual(paths.get_big_disk_dir().replace("\\", "/"), "/root/kaggle_big")

        # Reset environment
        os.environ["PIPELINE_ENV"] = "local3090"
        paths._ROOT_CACHE = None
        paths._BIG_DISK_CACHE = None

    def test_path_resolution(self):
        """Verify resolve_path helper."""
        os.environ["PIPELINE_ENV"] = "local3090"
        paths._ROOT_CACHE = None
        res = paths.resolve_path("checkpoints/test.pt")
        self.assertTrue(os.path.isabs(res))
        self.assertTrue(res.replace("\\", "/").endswith("checkpoints/test.pt"))

    def test_ensure_all_dirs(self):
        """Verify directory creation dictionary."""
        dirs = paths.ensure_all_dirs()
        for key in ["root", "data", "checkpoints", "samples", "logs", "subtalker", "code2wav", "big_disk"]:
            self.assertIn(key, dirs)
            self.assertTrue(len(dirs[key]) > 0)

    def test_run_3090_import_and_parser(self):
        """Verify run_3090 module can be imported and parses arguments."""
        import run_3090
        self.assertTrue(hasattr(run_3090, "main"))
        self.assertTrue(hasattr(run_3090, "DualLogger"))

    def test_ascii_purity_of_all_created_files(self):
        """Verify that all newly created and modified files are 100% pure ASCII (ord <= 127)."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        files_to_check = [
            "paths.py",
            "run_3090.py",
            "Dockerfile",
            "Dockerfile.phase1",
            "Dockerfile.phase2",
            "docker-compose.yml",
            "docker_run_3090.sh",
            "setup_3090.sh",
            "docs/REPORT_PORTING_3090.md",
        ]
        for rel_path in files_to_check:
            abs_path = os.path.join(project_root, rel_path)
            self.assertTrue(os.path.isfile(abs_path), f"File {rel_path} does not exist")
            with open(abs_path, "rb") as f:
                content = f.read()
            for idx, byte in enumerate(content):
                self.assertLessEqual(
                    byte,
                    127,
                    f"Non-ASCII byte {byte} found at index {idx} in {rel_path}",
                )

    def test_docker_security_and_flags(self):
        """Verify that security rules and necessary flags are present in Docker configurations."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

        # 1. Check docker-compose.yml
        compose_path = os.path.join(project_root, "docker-compose.yml")
        with open(compose_path, "r", encoding="ascii") as f:
            compose_content = f.read()

        self.assertNotIn("privileged", compose_content.lower())
        self.assertNotIn("network_mode: host", compose_content.lower())
        self.assertIn("ipc: host", compose_content)
        self.assertIn('user: "1000:1000"', compose_content)
        self.assertIn("gemma4-omni:latest", compose_content)

        # 2. Check docker_run_3090.sh
        sh_path = os.path.join(project_root, "docker_run_3090.sh")
        with open(sh_path, "r", encoding="ascii") as f:
            sh_content = f.read()

        self.assertNotIn("--privileged", sh_content)
        self.assertNotIn("--network host", sh_content)
        self.assertIn("--ipc=host", sh_content)
        self.assertIn("--user", sh_content)
        self.assertIn("--gpus all", sh_content)


if __name__ == "__main__":
    unittest.main()
