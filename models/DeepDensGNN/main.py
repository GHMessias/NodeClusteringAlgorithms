"""Run DeepDensGNN with the same YAML experiment engine as DensGNN."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
DENSGNN_SRC = ROOT.parent / "DensGNN" / "src"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(DENSGNN_SRC))

from graph_benchmark.config.yaml import apply_overrides, load_yaml_config  # noqa: E402
from graph_benchmark.runner import run_from_config  # noqa: E402
from graph_benchmark.utils.seed import set_random_seed  # noqa: E402
import clustering  # noqa: E402,F401 - registers the DeepDensGNN algorithm


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepDensGNN from a YAML configuration")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = load_yaml_config(args.cfg)
    if args.set:
        config = apply_overrides(config, args.set)
    seed = config.get("run", {}).get("seed", 42)
    set_random_seed(None if seed is None else int(seed))
    result = run_from_config(config)
    print(f"Results saved to: {result['artifacts']['output_dir']}")


if __name__ == "__main__":
    main()
