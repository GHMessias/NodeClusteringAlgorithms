"""Wall-clock budgets, process cleanup, and persisted OOT failures."""

import csv
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import main as benchmark


class ModelTimeoutTests(unittest.TestCase):
    def test_timeout_kills_descendant_ignoring_sigterm(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            pid_path = folder / "child.pid"
            child = (
                "import os,signal,time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                f"Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(30)"
            )
            source = (
                "import subprocess,sys,time; "
                f"subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(30)"
            )
            command = benchmark.Command([sys.executable, "-c", source], ROOT)
            with (folder / "stdout.log").open("w") as stdout, (folder / "stderr.log").open("w") as stderr:
                with self.assertRaisesRegex(benchmark.ModelTimeoutError, "OOT"):
                    benchmark.run_bounded_command(
                        command, os.environ.copy(), stdout, stderr,
                        time.perf_counter() + 0.7, 0.7, "training",
                    )
            self.assertTrue(pid_path.is_file(), "Descendant must start before timeout")
            pid = int(pid_path.read_text())
            for _ in range(50):
                stat = Path(f"/proc/{pid}/stat")
                if not stat.exists() or stat.read_text().split(")", 1)[1].split()[0] == "Z":
                    break
                time.sleep(0.02)
            else:
                self.fail("Descendant survived timeout cleanup")

    def test_multiple_stages_share_one_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            context = benchmark.RunContext("test", None, "cpu", 1, folder / "run" / "dgc_efr")
            third_started = folder / "third_started"
            commands = [
                benchmark.Command([sys.executable, "-c", "import time; time.sleep(0.35)"], ROOT),
                benchmark.Command([sys.executable, "-c", "import time; time.sleep(0.35)"], ROOT),
                benchmark.Command([sys.executable, "-c", f"from pathlib import Path; Path({str(third_started)!r}).touch()"], ROOT),
            ]
            with patch.dict(benchmark.MODEL_REGISTRY, {"dgc_efr": lambda context, params: commands}):
                row = benchmark.run_model(
                    "dgc_efr", context, {}, {}, ["runtime_seconds"],
                    {"save_embeddings": False}, "test", timeout_seconds=0.6,
                )
            self.assertEqual(row["status"], "failed")
            self.assertIn("OOT (out of time)", row["error"])
            self.assertIn("stage 2/3", row["error"])
            self.assertFalse(third_started.exists())
            self.assertLess(row["runtime_seconds"], 3)
            metadata = yaml.safe_load((context.model_dir / "metadata.yaml").read_text())
            self.assertEqual(metadata["error_code"], "OOT")
            self.assertEqual(metadata["timeout_seconds"], 0.6)
            self.assertIn("OOT", (context.model_dir / "stderr.log").read_text())

    def test_evaluation_uses_remaining_model_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            labels = folder / "labels.npy"
            np.save(labels, [0, 0, 1, 1])
            context = benchmark.RunContext("test", folder / "unused.pt", "cpu", 1, folder / "run" / "ucode")
            source = (
                "import shutil; "
                f"shutil.copyfile({str(labels)!r}, {str(context.model_dir / 'fake_labels.npy')!r})"
            )
            command = benchmark.Command([sys.executable, "-c", source], ROOT)
            with patch.dict(benchmark.MODEL_REGISTRY, {"ucode": lambda context, params: [command]}), patch.object(
                benchmark, "python_script", return_value=[sys.executable, "-c", "import time; time.sleep(30)"]
            ):
                row = benchmark.run_model(
                    "ucode", context, {}, {"num_nodes": 4}, ["n_clusters", "runtime_seconds"],
                    {"save_embeddings": False}, "test", timeout_seconds=0.5,
                )
            self.assertEqual(row["status"], "failed")
            self.assertIn("OOT", row["error"])
            self.assertIn("evaluation", row["error"])
            self.assertTrue(np.isnan(row["n_clusters"]))

    def test_oot_is_written_to_csv_and_next_seed_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            labels = folder / "labels.npy"
            np.save(labels, [0, 0, 1, 1])
            data_pt = folder / "data.pt"
            torch.save({"num_nodes": 4, "y": torch.tensor([0, 0, 1, 1])}, data_pt)
            config = {
                "experiment": {"dataset": "test", "data_pt": str(data_pt), "output_root": str(folder / "results"),
                               "device": "cpu", "seed": 1, "runs": 2, "continue_on_error": True, "timeout_seconds": 0.4},
                "output": {"save_embeddings": False, "save_metadata": True, "generate_figures": False},
                "metrics": {"include": ["runtime_seconds"]},
                "models": {"ucode": {"enabled": True, "params": {}}},
            }
            config_path = folder / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))

            def builder(context, params):
                if context.seed == 1:
                    source = "import time; time.sleep(30)"
                else:
                    source = (
                        "import shutil; "
                        f"shutil.copyfile({str(labels)!r}, {str(context.model_dir / 'fake_labels.npy')!r})"
                    )
                return [benchmark.Command([sys.executable, "-c", source], ROOT)]

            with patch.dict(benchmark.MODEL_REGISTRY, {"ucode": builder}), patch.object(
                sys, "argv", ["main.py", "--config", str(config_path), "--model", "all"]
            ), warnings.catch_warnings():
                warnings.simplefilter("ignore", benchmark.ConfigWarning)
                self.assertEqual(benchmark.main(), 1)
            rows = []
            for path in sorted((folder / "results/test/runs").glob("*/metrics.csv")):
                with path.open() as stream:
                    rows.extend(csv.DictReader(stream))
            self.assertEqual([row["status"] for row in rows], ["failed", "success"])
            self.assertTrue(rows[0]["error"].startswith("OOT (out of time)"))
            self.assertGreater(float(rows[0]["runtime_seconds"]), 0)
            self.assertEqual(rows[1]["error"], "")

    def test_nonzero_exit_is_a_regular_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            command = benchmark.Command([sys.executable, "-c", "raise SystemExit(7)"], ROOT)
            with (folder / "stdout.log").open("w") as stdout, (folder / "stderr.log").open("w") as stderr:
                with self.assertRaises(subprocess.CalledProcessError) as caught:
                    benchmark.run_bounded_command(command, os.environ.copy(), stdout, stderr,
                                                  time.perf_counter() + 5, 5, "training")
            self.assertEqual(caught.exception.returncode, 7)


if __name__ == "__main__":
    unittest.main()
