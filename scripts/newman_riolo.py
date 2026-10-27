#!/usr/bin/env python3
"""Run the unmodified Newman-Riolo community-count estimator."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import yaml

from communities.pyg_to_gml import normalized_edges, write_gml


SCRIPT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = SCRIPT_ROOT / "communities"
BUILD_ROOT = SOURCE_ROOT / ".build"
SOURCE_FILES = (
    SOURCE_ROOT / "communities.c",
    SOURCE_ROOT / "readgml.c",
    SOURCE_ROOT / "readgml.h",
    SOURCE_ROOT / "network.h",
)
EXPECTED_SWEEPS = 10_000
MAX_GROUPS = 25


class NewmanRioloError(RuntimeError):
    """Raised when the external estimator cannot produce a valid K."""


@dataclass(frozen=True)
class NewmanRioloResult:
    selected_k: int
    artifact_dir: Path
    summary: dict[str, Any]


def compile_estimator() -> Path:
    """Compile the original C source without modifying it."""
    compiler = shutil.which("gcc")
    if compiler is None:
        raise NewmanRioloError("gcc is required to compile Newman-Riolo")

    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    executable = BUILD_ROOT / "communities"
    newest_source = max(path.stat().st_mtime_ns for path in SOURCE_FILES)
    if executable.is_file() and executable.stat().st_mtime_ns >= newest_source:
        return executable

    command = [
        compiler,
        "-O",
        "-o",
        str(executable),
        str(SOURCE_ROOT / "communities.c"),
        str(SOURCE_ROOT / "readgml.c"),
        "-lm",
        "-lgsl",
        "-lgslcblas",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise NewmanRioloError(
            "Could not compile the original Newman-Riolo code. Install the GNU "
            "Scientific Library development package (on openSUSE: "
            "'sudo zypper install gsl-devel')."
            + (f" Compiler output: {details}" if details else "")
        )
    return executable


def parse_samples(path: Path, burn_in: int) -> np.ndarray:
    rows: list[tuple[int, int, float, float]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if len(fields) != 4:
                raise NewmanRioloError(
                    f"Invalid estimator output at {path}:{line_number}"
                )
            try:
                row = (int(fields[0]), int(fields[1]), float(fields[2]), float(fields[3]))
            except ValueError as exc:
                raise NewmanRioloError(
                    f"Invalid numeric estimator output at {path}:{line_number}"
                ) from exc
            if row[0] >= burn_in:
                rows.append(row)
    if not rows:
        raise NewmanRioloError("No Newman-Riolo samples remain after burn-in")
    return np.asarray(rows, dtype=np.float64)


def posterior_table(samples: np.ndarray) -> list[dict[str, Any]]:
    values, counts = np.unique(samples[:, 1].astype(np.int64), return_counts=True)
    total = int(counts.sum())
    return [
        {
            "k": int(value),
            "count": int(count),
            "probability": float(count / total),
        }
        for value, count in zip(values, counts)
    ]


def posterior_mode(table: list[dict[str, Any]]) -> int:
    # Resolve exact ties toward the smaller model.
    winner = max(table, key=lambda row: (row["count"], -row["k"]))
    return int(winner["k"])


def write_posterior(path: Path, table: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("k", "count", "probability"))
        writer.writeheader()
        writer.writerows(table)


def transition_rate(samples: np.ndarray) -> float:
    values = samples[:, 1]
    if values.size < 2:
        return 0.0
    return float(np.count_nonzero(values[1:] != values[:-1]) / (values.size - 1))


def lag_one_autocorrelation(samples: np.ndarray) -> float:
    values = samples[:, 1]
    if values.size < 2 or float(values.std()) == 0.0:
        return 1.0
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def estimate_newman_riolo(
    data: Any,
    options: Mapping[str, Any],
    artifact_dir: Path,
) -> NewmanRioloResult:
    """Estimate K once from graph topology and retain the complete trace."""
    burn_in = int(options.get("burn_in", 1000))
    timeout_value = options.get("timeout_seconds")
    timeout = None if timeout_value is None else float(timeout_value)
    if burn_in < 0 or burn_in >= EXPECTED_SWEEPS:
        raise NewmanRioloError(
            f"burn_in must satisfy 0 <= burn_in < {EXPECTED_SWEEPS}"
        )
    if timeout is not None and timeout <= 0:
        raise NewmanRioloError("timeout_seconds must be positive or null")

    artifact_dir.mkdir(parents=True, exist_ok=False)
    topology_data = SimpleNamespace(**data) if isinstance(data, dict) else data
    edges, num_nodes = normalized_edges(
        topology_data,
        directed=False,
        keep_self_loops=False,
        weight_attr=None,
    )
    if num_nodes < MAX_GROUPS:
        raise NewmanRioloError(
            f"The unmodified estimator initializes {MAX_GROUPS} groups and requires "
            f"at least {MAX_GROUPS} nodes; received {num_nodes}."
        )
    if not edges:
        raise NewmanRioloError("Newman-Riolo requires at least one graph edge")

    graph_path = artifact_dir / "graph.gml"
    trace_path = artifact_dir / "samples.out"
    stderr_path = artifact_dir / "stderr.log"
    write_gml(graph_path, num_nodes, edges, directed=False)
    executable = compile_estimator()

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    try:
        with graph_path.open("r", encoding="utf-8") as graph_stream, trace_path.open(
            "w", encoding="utf-8"
        ) as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            subprocess.run(
                [str(executable)],
                stdin=graph_stream,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=True,
                timeout=timeout,
            )
    except subprocess.TimeoutExpired as exc:
        raise NewmanRioloError(
            f"Newman-Riolo exceeded timeout_seconds={timeout}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        details = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
        raise NewmanRioloError(
            f"Newman-Riolo exited with status {exc.returncode}"
            + (f": {details}" if details else "")
        ) from exc
    runtime = time.perf_counter() - started

    samples = parse_samples(trace_path, burn_in)
    posterior = posterior_table(samples)
    selected_k = posterior_mode(posterior)
    warnings: list[str] = []
    if int(samples[:, 1].max()) == MAX_GROUPS:
        warnings.append(
            "Post-burn-in samples reached the hard-coded K=25 ceiling."
        )
    rate = transition_rate(samples)
    autocorrelation = lag_one_autocorrelation(samples)
    if rate < 0.01:
        warnings.append(
            "The sampled K changed in fewer than 1% of transitions; inspect convergence."
        )
    if autocorrelation > 0.9:
        warnings.append(
            "Lag-1 autocorrelation of K exceeds 0.9; posterior frequencies are highly correlated."
        )

    write_posterior(artifact_dir / "posterior.csv", posterior)
    summary: dict[str, Any] = {
        "configured_method": "newman-riolo",
        "resolved_method": "newman-riolo-2017",
        "selected_k": selected_k,
        "selection": "posterior_mode",
        "burn_in": burn_in,
        "total_sweeps": EXPECTED_SWEEPS,
        "retained_samples": int(samples.shape[0]),
        "hard_coded_max_groups": MAX_GROUPS,
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "mean_k": float(samples[:, 1].mean()),
        "mean_effective_k": float(samples[:, 2].mean()),
        "transition_rate": rate,
        "lag_one_autocorrelation": autocorrelation,
        "posterior": posterior,
        "runtime_seconds": runtime,
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE_ROOT / "communities.c"),
        "source_modified_by_wrapper": False,
        "rng_seed_policy": "original time(NULL)",
        "deterministic": False,
        "warnings": warnings,
    }
    with (artifact_dir / "summary.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(summary, stream, sort_keys=False, allow_unicode=True)
    return NewmanRioloResult(selected_k, artifact_dir, summary)
