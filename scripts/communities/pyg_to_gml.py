#!/usr/bin/env python3
"""Convert a PyTorch Geometric data.pt file to the GML format used here."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def load_torch_object(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "This converter needs PyTorch installed to load .pt files."
        ) from exc

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def is_data_like(obj: Any) -> bool:
    if isinstance(obj, type):
        return False
    edge_index = getattr(obj, "edge_index", None)
    return edge_index is not None and not isinstance(edge_index, property)


def data_from_dict(mapping: dict[str, Any], data_cls: Any | None = None) -> Any:
    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise SystemExit(
            "This file stores PyG data as a dictionary. Install torch_geometric "
            "to rebuild the Data object."
        ) from exc

    cls = data_cls if isinstance(data_cls, type) else Data
    if hasattr(cls, "from_dict"):
        return cls.from_dict(mapping)
    return cls(**mapping)


def maybe_first_graph(obj: Any, graph_index: int) -> Any:
    if is_data_like(obj):
        return obj

    if isinstance(obj, dict):
        if "edge_index" in obj:
            return data_from_dict(obj)
        for key in ("data", "graph", "pyg_data"):
            if key in obj and is_data_like(obj[key]):
                return obj[key]
            if key in obj and isinstance(obj[key], dict) and "edge_index" in obj[key]:
                return data_from_dict(obj[key])

    if isinstance(obj, (list, tuple)):
        if len(obj) == 3 and isinstance(obj[0], dict):
            data = data_from_dict(obj[0], obj[2])
            slices = obj[1]
            if isinstance(slices, dict):
                return separate_collated_data(data, slices, graph_index)
            return data
        if len(obj) == 2 and isinstance(obj[0], dict):
            data = data_from_dict(obj[0])
            slices = obj[1]
            if isinstance(slices, dict):
                return separate_collated_data(data, slices, graph_index)
            return data
        if len(obj) == 2 and is_data_like(obj[0]) and isinstance(obj[1], dict):
            return separate_collated_data(obj[0], obj[1], graph_index)
        for item in obj:
            if is_data_like(item):
                return item

    raise SystemExit(
        "Could not find a PyTorch Geometric Data object with an edge_index field."
    )


def separate_collated_data(data: Any, slices: dict[str, Any], graph_index: int) -> Any:
    """Best-effort extraction from the (data, slices) format used by PyG."""
    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise SystemExit(
            "This file looks like a PyG dataset tuple. Install torch_geometric "
            "to extract a graph from it."
        ) from exc

    import torch

    if not slices:
        return data

    max_graphs = min(len(v) - 1 for v in slices.values() if hasattr(v, "__len__"))
    if graph_index < 0 or graph_index >= max_graphs:
        raise SystemExit(f"--graph-index must be between 0 and {max_graphs - 1}.")

    out = Data()
    node_offset = 0
    if "x" in slices:
        node_offset = int(slices["x"][graph_index])

    for key, value in data:
        if key not in slices:
            setattr(out, key, value)
            continue

        start = int(slices[key][graph_index])
        end = int(slices[key][graph_index + 1])

        if torch.is_tensor(value):
            cat_dim = data.__cat_dim__(key, value)
            if cat_dim is None or value.dim() == 0:
                sliced = value
            else:
                sliced = value.narrow(cat_dim, start, end - start).clone()
        elif isinstance(value, list):
            sliced = value[start:end]
        else:
            sliced = value

        setattr(out, key, sliced)

    if hasattr(out, "edge_index") and node_offset:
        out.edge_index = out.edge_index - node_offset

    if "x" in slices:
        out.num_nodes = int(slices["x"][graph_index + 1] - slices["x"][graph_index])

    return out


def tensor_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def infer_num_nodes(data: Any, edges: list[tuple[int, int]]) -> int:
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is not None:
        return int(num_nodes)
    x = getattr(data, "x", None)
    if x is not None and hasattr(x, "size"):
        return int(x.size(0))
    if not edges:
        return 0
    return max(max(src, dst) for src, dst in edges) + 1


def edge_weights(data: Any, attr_name: str | None, edge_count: int) -> list[float] | None:
    candidates = [attr_name] if attr_name else ["edge_weight", "weight", "edge_attr"]

    for name in candidates:
        if not name or not hasattr(data, name):
            continue
        raw = tensor_to_list(getattr(data, name))
        if len(raw) != edge_count:
            continue

        weights = []
        for item in raw:
            if isinstance(item, list):
                item = item[0] if item else 1.0
            weights.append(float(item))
        return weights

    return None


def normalized_edges(
    data: Any,
    directed: bool,
    keep_self_loops: bool,
    weight_attr: str | None,
) -> tuple[list[tuple[int, int, float]], int]:
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        raise SystemExit("Data object does not contain edge_index.")

    edge_rows = tensor_to_list(edge_index)
    if len(edge_rows) != 2:
        raise SystemExit("edge_index must have shape [2, num_edges].")

    raw_edges = [(int(src), int(dst)) for src, dst in zip(edge_rows[0], edge_rows[1])]
    weights = edge_weights(data, weight_attr, len(raw_edges))

    seen: dict[tuple[int, int], float] = {}
    result: list[tuple[int, int, float]] = []

    for index, (src, dst) in enumerate(raw_edges):
        if src == dst and not keep_self_loops:
            continue

        key = (src, dst) if directed else (min(src, dst), max(src, dst))
        weight = weights[index] if weights is not None else 1.0

        if key in seen:
            continue
        seen[key] = weight
        result.append((key[0], key[1], weight))

    return result, infer_num_nodes(data, raw_edges)


def escape_gml_label(label: str) -> str:
    return label.replace("\\", "\\\\").replace('"', '\\"')


def write_gml(path: Path, num_nodes: int, edges: list[tuple[int, int, float]], directed: bool) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write('Creator "pyg_to_gml.py"\n')
        handle.write("graph\n")
        handle.write("[\n")
        handle.write(f"  directed {1 if directed else 0}\n")

        for node_id in range(num_nodes):
            label = escape_gml_label(str(node_id))
            handle.write("  node\n")
            handle.write("  [\n")
            handle.write(f"    id {node_id}\n")
            handle.write(f'    label "{label}"\n')
            handle.write("  ]\n")

        for src, dst, weight in edges:
            handle.write("  edge\n")
            handle.write("  [\n")
            handle.write(f"    source {src}\n")
            handle.write(f"    target {dst}\n")
            handle.write(f"    value {weight:g}\n")
            handle.write("  ]\n")

        handle.write("]\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch Geometric .pt graph to this repository's GML input format."
    )
    parser.add_argument("input", type=Path, help="Input .pt file saved from PyTorch/PyG.")
    parser.add_argument("output", type=Path, help="Output .gml file.")
    parser.add_argument(
        "--directed",
        action="store_true",
        help="Write directed GML without deduplicating reverse edges.",
    )
    parser.add_argument(
        "--keep-self-loops",
        action="store_true",
        help="Keep self-loop edges. By default they are skipped.",
    )
    parser.add_argument(
        "--weight-attr",
        help="Edge weight attribute to use. Defaults to edge_weight, weight, then edge_attr.",
    )
    parser.add_argument(
        "--graph-index",
        type=int,
        default=0,
        help="Graph index to extract when loading a PyG (data, slices) dataset tuple.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    obj = load_torch_object(args.input)
    data = maybe_first_graph(obj, args.graph_index)
    edges, num_nodes = normalized_edges(
        data,
        directed=args.directed,
        keep_self_loops=args.keep_self_loops,
        weight_attr=args.weight_attr,
    )
    write_gml(args.output, num_nodes, edges, args.directed)
    print(f"Wrote {args.output} with {num_nodes} nodes and {len(edges)} edges.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
