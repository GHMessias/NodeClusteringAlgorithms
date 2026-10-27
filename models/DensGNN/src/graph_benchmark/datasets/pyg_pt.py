from __future__ import annotations

"""Loader for PyTorch/PyG graph files supplied by the outer benchmark."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from graph_benchmark.registry import DATASET_REGISTRY
from graph_benchmark.utils.io import resolve_project_path


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _is_graph_data(value: Any) -> bool:
    if isinstance(value, dict):
        return "x" in value or "edge_index" in value
    return getattr(value, "x", None) is not None or getattr(value, "edge_index", None) is not None


def _unwrap_graph_data(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        for item in value:
            if _is_graph_data(item):
                return item
        if not value:
            raise ValueError("The data_pt file contains an empty sequence.")
        return value[0]
    if isinstance(value, dict) and "data" in value:
        return value["data"]
    return value


def _as_pyg_data(value: Any) -> Data:
    value = _unwrap_graph_data(value)
    if isinstance(value, Data):
        return value
    if isinstance(value, dict):
        return Data(**value)
    if getattr(value, "x", None) is not None and getattr(value, "edge_index", None) is not None:
        return value
    raise TypeError("The data_pt file does not contain a PyG-compatible graph.")


@DATASET_REGISTRY.register("pyg_pt")
def load_pyg_pt_from_config(dataset_config: dict[str, Any]) -> dict[str, Any]:
    raw_path = dataset_config.get("path")
    if not raw_path:
        raise KeyError("Missing 'dataset.path' for the pyg_pt loader.")

    data_path = resolve_project_path(raw_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"data_pt file not found: {data_path}")

    data = _as_pyg_data(_torch_load(data_path))
    if getattr(data, "x", None) is None or getattr(data, "edge_index", None) is None:
        raise ValueError("DensGNN requires data.x and data.edge_index.")

    labels = getattr(data, "y", None)
    num_classes = 0
    if labels is not None:
        labels_array = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
        if labels_array.ndim > 1:
            labels_array = labels_array.argmax(axis=-1)
        num_classes = int(np.unique(labels_array.reshape(-1)).size)

    dataset_name = str(dataset_config.get("name") or data_path.stem)
    dataset = SimpleNamespace(
        num_features=int(data.x.shape[-1]),
        num_classes=num_classes,
    )
    return {
        "dataset": dataset,
        "data": data,
        "metadata": {
            "loader": "pyg_pt",
            "name": dataset_name,
            "path": str(data_path),
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.num_edges),
            "num_features": int(dataset.num_features),
            "num_classes": int(dataset.num_classes),
        },
    }
