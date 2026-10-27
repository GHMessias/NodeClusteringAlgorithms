#!/usr/bin/env python3
"""Configuration-driven orchestration for all clustering baselines."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import shlex
import signal
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml
import numpy as np
import torch

from evaluate import (
    EMBEDDING_METRICS,
    SUPERVISED_METRICS,
    SUPPORTED_METRICS,
    as_label_vector,
    get_attr,
    load_data,
    load_pred_labels,
    validate_requested_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = REPO_ROOT / "models"
IDENTIFICATION_COLUMNS = [
    "dataset",
    "run_id",
    "model",
    "seed",
    "status",
    "error",
]
DEFAULT_TIMEOUT_SECONDS = 600.0


class ConfigError(ValueError):
    pass


class ConfigWarning(UserWarning):
    pass


class ModelTimeoutError(TimeoutError):
    """The shared execution budget for one model/seed has expired."""


@dataclass(frozen=True)
class Command:
    argv: list[str]
    cwd: Path


@dataclass(frozen=True)
class RunContext:
    dataset: str
    data_pt: Path | None
    device: str
    seed: int
    model_dir: Path


CommandBuilder = Callable[[RunContext, Mapping[str, Any]], list[Command]]


def python_script(path: Path) -> list[str]:
    return [sys.executable, str(path)]


def append_params(
    command: list[str],
    params: Mapping[str, Any],
    flag_map: Mapping[str, str],
    *,
    true_flags: set[str] | None = None,
) -> None:
    true_flags = true_flags or set()
    unknown = sorted(set(params) - set(flag_map))
    if unknown:
        raise ConfigError(f"Unknown model parameters: {', '.join(unknown)}")

    for key, value in params.items():
        if value is None:
            continue
        flag = flag_map[key]
        if key in true_flags:
            if value:
                command.append(flag)
            continue
        command.append(flag)
        if isinstance(value, (list, tuple)):
            command.extend(str(item) for item in value)
        elif isinstance(value, bool):
            command.append(str(value).lower())
        else:
            command.append(str(value))


COANE_FLAGS = {
    key: f"--{key}"
    for key in (
        "window_hsize",
        "emb_dim",
        "num_epochs",
        "num_Vbatch",
        "num_negative",
        "num_walks",
        "p",
        "q",
        "walk_length",
        "contoller",
        "c_f",
        "decay",
        "val_frac",
        "test_frac",
        "subsample_rate",
        "kmeans_clusters",
    )
}
COANE_FLAGS.update(
    {
        "verbose": "--verbose",
        "reload": "--reload",
        "save": "--save",
        "splitting": "--Spliting",
        "directed": "--dircted",
        "evaluate": "--eval",
    }
)
COANE_TOPK_FLAGS = {
    **COANE_FLAGS,
    "evaluate_each_epoch": "--eval_iter",
    "index": "--index",
    "star": "--star",
    "end": "--end",
}


def build_coane(context: RunContext, params: Mapping[str, Any], *, topk: bool) -> list[Command]:
    variant = "CoANE topk with attributes preservation" if topk else "CoANE with attributes preservation"
    script_name = "main_CoANE_topk.py" if topk else "main_CoANE.py"
    script = MODELS_ROOT / "CoANE" / variant / script_name
    command = python_script(script) + [
        "--dataset",
        context.dataset,
        "--device",
        context.device,
        "--output_dir",
        str(context.model_dir),
        "--seed",
        str(context.seed),
        "--kmeans",
    ]
    if context.data_pt:
        command += ["--data_pt", str(context.data_pt)]
    if not topk:
        command += ["--embedding_output", str(context.model_dir / "embedding.pkl")]
    append_params(command, params, COANE_TOPK_FLAGS if topk else COANE_FLAGS)
    return [Command(command, script.parent)]


DGC_PREAE_FLAGS = {
    key: f"--{key}"
    for key in (
        "k",
        "epochs",
        "lr",
        "n_clusters",
        "n_z",
        "hidden1_dim",
        "hidden2_dim",
        "hidden3_dim",
        "batch_size",
        "kmeans_n_init",
        "gamma",
        "decay_period",
    )
}
DGC_PREGAE_FLAGS = {
    key: f"--{key}"
    for key in (
        "lr",
        "epochs",
        "k",
        "n_clusters",
        "hidden1_dim",
        "hidden2_dim",
        "hidden3_dim",
        "weight_decay",
        "alpha",
        "diffusion_steps",
        "kmeans_n_init",
    )
}
DGC_TRAIN_FLAGS = {
    key: f"--{key}"
    for key in (
        "lr",
        "k",
        "epochs",
        "lambda1",
        "n_clusters",
        "update_interval",
        "hidden1_dim",
        "hidden2_dim",
        "hidden3_dim",
        "n_z",
        "weight_decay",
        "alpha",
        "sigma",
        "diffusion_steps",
        "kmeans_n_init",
        "gae_reconstruction_weight",
        "ae_reconstruction_weight",
        "gamma",
        "decay_period",
    )
}


def build_dgc_efr(context: RunContext, params: Mapping[str, Any]) -> list[Command]:
    expected = {"preae", "pregae", "train"}
    unknown = sorted(set(params) - expected)
    if unknown:
        raise ConfigError(f"Unknown DGC-EFR stages: {', '.join(unknown)}")
    missing = sorted(expected - set(params))
    if missing:
        raise ConfigError(f"Missing DGC-EFR stage configuration: {', '.join(missing)}")

    root = MODELS_ROOT / "DGC-EFR"
    artifact_dir = context.model_dir / "artifacts"
    preae_path = artifact_dir / f"preae_{context.dataset}.pkl"
    pregae_path = artifact_dir / f"pregae_{context.dataset}.pkl"
    common = ["--name", context.dataset, "--device", context.device, "--seed", str(context.seed)]
    if context.data_pt:
        common += ["--data_pt", str(context.data_pt)]

    preae = python_script(root / "preae.py") + common + ["--preae_path", str(preae_path)]
    append_params(preae, params["preae"], DGC_PREAE_FLAGS)
    pregae = python_script(root / "pregae.py") + common + ["--pregae_path", str(pregae_path)]
    append_params(pregae, params["pregae"], DGC_PREGAE_FLAGS)
    train = python_script(root / "dgc_efr.py") + common + [
        "--output_dir",
        str(context.model_dir),
        "--preae_path",
        str(preae_path),
        "--pregae_path",
        str(pregae_path),
        "--save_dir",
        str(artifact_dir / "experiments"),
    ]
    append_params(train, params["train"], DGC_TRAIN_FLAGS)
    return [Command(preae, root), Command(pregae, root), Command(train, root)]


SOLI_FLAGS = {
    key: f"--{key}"
    for key in (
        "epochs",
        "kmeans_clusters",
        "patience",
        "lr",
        "weight_decay",
        "hidden_dim",
        "autoencoder_latent_dim",
        "kmeans_n_init",
        "pretrain_epochs",
        "pretrain_batch_size",
        "pretrain_lr",
        "pretrain_momentum",
        "clustering_lr",
        "clustering_momentum",
        "max_iterations",
        "update_interval",
        "tolerance",
    )
}


def build_soli(context: RunContext, params: Mapping[str, Any]) -> list[Command]:
    script = MODELS_ROOT / "SOLI" / "SOLI_GAE" / "main.py"
    command = python_script(script) + [
        "--dataset",
        context.dataset,
        "--device",
        "cpu",
        "--seed",
        str(context.seed),
        "--output_dir",
        str(context.model_dir),
    ]
    if context.data_pt:
        command += ["--data_pt", str(context.data_pt)]
    append_params(command, params, SOLI_FLAGS)
    return [Command(command, script.parent)]


UCODE_FLAGS = {
    key: f"--{key}"
    for key in (
        "epochs",
        "kmeans_clusters",
        "hid_units",
        "hid_dimension",
        "num_experiments",
        "lr",
        "weight_decay",
    )
}


def build_ucode(context: RunContext, params: Mapping[str, Any]) -> list[Command]:
    script = MODELS_ROOT / "UCODE" / "main.py"
    command = python_script(script) + [
        "--device",
        context.device,
        "--seed",
        str(context.seed),
        "--output_dir",
        str(context.model_dir),
    ]
    if context.data_pt:
        command += ["--data_pt", str(context.data_pt)]
    append_params(command, params, UCODE_FLAGS)
    return [Command(command, script.parent)]


NS4GC_FLAGS = {
    key: f"--{key}"
    for key in (
        "epochs",
        "kmeans_clusters",
        "lambda_neighbor",
        "gamma_sparsity",
        "similarity_threshold",
        "temperature",
        "hidden_dims",
        "lr",
        "weight_decay",
        "edge_drop_rate_1",
        "edge_drop_rate_2",
        "feature_drop_rate_1",
        "feature_drop_rate_2",
        "kmeans_n_init",
    )
}


def build_ns4gc(context: RunContext, params: Mapping[str, Any]) -> list[Command]:
    script = MODELS_ROOT / "NS4GC" / "main.py"
    command = python_script(script) + [
        "--device",
        context.device,
        "--seed",
        str(context.seed),
        "--output_dir",
        str(context.model_dir),
    ]
    if context.data_pt:
        command += ["--data_pt", str(context.data_pt)]
    append_params(command, params, NS4GC_FLAGS)
    return [Command(command, script.parent)]


DGCLUSTER_FLAGS = {
    key: f"--{key}"
    for key in (
        "lam",
        "alp",
        "epochs",
        "base_model",
        "hidden_dims",
        "out_dim",
        "dropout",
        "lr",
        "weight_decay",
        "adam_betas",
        "amsgrad",
        "birch_threshold",
        "n_clusters",
    )
}


def build_dgcluster(context: RunContext, params: Mapping[str, Any]) -> list[Command]:
    script = MODELS_ROOT / "DGCluster" / "main.py"
    command = python_script(script) + [
        "--dataset",
        context.dataset,
        "--device",
        context.device,
        "--seed",
        str(context.seed),
        "--output_dir",
        str(context.model_dir),
        "--results_dir",
        str(context.model_dir / "artifacts"),
    ]
    if context.data_pt:
        command += ["--data_pt", str(context.data_pt)]
    append_params(command, params, DGCLUSTER_FLAGS)
    return [Command(command, script.parent)]


DGCSS_FLAGS = {
    "centroid_method": "--find_centroids_alg",
    "n_clusters": "--n_clusters",
    "epochs": "--epochs",
    "clustering_loss_gamma": "--c_loss_gama",
    "p_interval": "--p_interval",
    "hidden_layer": "--hidden_layer",
    "output_layer": "--output_layer",
    "b_matrix_degree": "--b_matrix_degree",
    "learning_rate": "--learning_rate",
    "lr_change_gamma": "--lr_change_gamma",
    "lr_change_epochs": "--lr_change_epochs",
    "update_clusters_step_size": "--update_clusters_step_size",
    "rechose_centroids": "--rechose_centroids",
    "not_improving_limit": "--not_improving_limit",
    "num_layers": "--num_layers",
    "attention_heads": "--attention_heads",
    "dropout": "--dropout",
    "concat": "--concat",
    "kmeans_random_state": "--kmeans_random_state",
    "kmeans_n_init": "--kmeans_n_init",
    "save_native_plots": "--save_native_plots",
    "debug": "--debug",
}


def build_dgcss(context: RunContext, params: Mapping[str, Any]) -> list[Command]:
    if context.data_pt is None:
        raise ConfigError("DGCSS requires experiment.data_pt")
    root = MODELS_ROOT / "DGCSS-main"
    script = root / "benchmark.py"
    command = python_script(script) + [
        "--data_pt",
        str(context.data_pt),
        "--output_dir",
        str(context.model_dir),
        "--device",
        context.device,
        "--seed",
        str(context.seed),
    ]
    append_params(command, params, DGCSS_FLAGS)
    return [Command(command, root)]


SANECE_FLAGS = {
    key: ("--epoch" if key == "epochs" else f"--{key}")
    for key in (
        "weighted",
        "directed",
        "classification_eval",
        "neg_mean",
        "epochs",
        "area",
        "kmeans_clusters",
        "reg",
        "epsilon",
        "dimension",
        "gcn_hidden_dim",
        "attribute_hidden_dim",
        "lr",
        "lr_sg",
        "batch_size",
        "batch_size_sg",
        "keep_prob",
        "walk_p",
        "walk_q",
        "num_walks",
        "walk_length",
        "window_size",
        "num_sampled",
        "margin",
    )
}


def build_sanece(context: RunContext, params: Mapping[str, Any]) -> list[Command]:
    script = MODELS_ROOT / "SANECE" / "main.py"
    command = python_script(script) + [
        "--dataset",
        context.dataset,
        "--seed",
        str(context.seed),
        "--output_dir",
        str(context.model_dir),
    ]
    if context.data_pt:
        command += ["--data_pt", str(context.data_pt)]
    append_params(command, params, SANECE_FLAGS, true_flags={"classification_eval"})
    return [Command(command, script.parent)]


EGC_VGAE_FLAGS = {
    key: f"--{key}"
    for key in (
        "epochs",
        "hidden1_dim",
        "hidden2_dim",
        "pretrain_lr",
        "train_lr",
        "reconstruction_scale",
        "numerical_stability",
        "n_clusters",
    )
}


def build_egc_vgae(context: RunContext, params: Mapping[str, Any]) -> list[Command]:
    script = MODELS_ROOT / "EG-VGAE" / "EGC_VGAE.py"
    command = python_script(script) + [
        "--dataset",
        context.dataset,
        "--seed",
        str(context.seed),
        "--output_dir",
        str(context.model_dir),
    ]
    if context.data_pt:
        command += ["--data_pt", str(context.data_pt)]
    append_params(command, params, EGC_VGAE_FLAGS)
    return [Command(command, script.parent)]


DSAGC_FLAGS = {
    key: f"--{key}"
    for key in (
        "epochs",
        "max_epoch",
        "lr",
        "n_clusters",
        "update_interval",
        "hidden_size",
        "embedding_size",
        "weight_decay",
        "alpha",
        "reliable_threshold",
        "kmeans_n_init",
        "self_supervised_hidden",
        "self_supervised_dropout",
        "self_supervised_heads",
        "self_supervised_lr",
        "self_supervised_weight_decay",
    )
}


def build_dsagc(context: RunContext, params: Mapping[str, Any]) -> list[Command]:
    script = MODELS_ROOT / "DSAGC" / "main.py"
    command = python_script(script) + [
        "--name",
        context.dataset,
        "--device",
        context.device,
        "--seed",
        str(context.seed),
        "--output_dir",
        str(context.model_dir),
        "--checkpoint_dir",
        str(context.model_dir / "checkpoints"),
    ]
    if context.data_pt:
        command += ["--data_pt", str(context.data_pt)]
    append_params(command, params, DSAGC_FLAGS)
    return [Command(command, script.parent)]


SCGC_FLAGS = {
    key: f"--{key}"
    for key in (
        "k",
        "lr",
        "n_clusters",
        "n_z",
        "model",
        "mode",
        "influence",
        "epochs",
        "iterations",
        "verbosity",
        "alpha",
        "beta",
        "batch_size",
        "order",
        "tau",
    )
}


def build_scgc(context: RunContext, params: Mapping[str, Any]) -> list[Command]:
    script = MODELS_ROOT / "SCGC" / "train.py"
    command = python_script(script) + [
        "--name",
        context.dataset,
        "--device",
        context.device,
        "--seed",
        str(context.seed),
        "--output_dir",
        str(context.model_dir),
    ]
    if context.data_pt:
        command += ["--data_pt", str(context.data_pt)]
    append_params(command, params, SCGC_FLAGS, true_flags={"influence"})
    return [Command(command, script.parent)]


DENSGNN_PARAMS = {
    "encoder_type",
    "encoder_hidden_channels",
    "embedding_channels",
    "epochs",
    "warmup_epochs",
    "learning_rate",
    "weight_decay",
    "dropout",
    "clustering_loss_gamma",
    "update_p_interval",
    "point_selection",
    "point_probability_threshold",
    "hdbscan_min_cluster_size",
    "hdbscan_min_samples",
    "hdbscan_cluster_selection_method",
    "mrd_k",
    "support_distance",
    "verbose",
    "log_interval",
    "evaluation_interval",
    "save_epoch_nmi",
}


def yaml_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    return str(value)


def build_densgnn(context: RunContext, params: Mapping[str, Any]) -> list[Command]:
    unknown = sorted(set(params) - DENSGNN_PARAMS)
    if unknown:
        raise ConfigError(f"Unknown DensGNN parameters: {', '.join(unknown)}")
    if params.get("encoder_type", "gcn") not in ("gcn", "gat", "sage"):
        raise ConfigError("DensGNN encoder_type must be gcn, gat or sage")
    if context.data_pt is None:
        raise ConfigError("DensGNN requires experiment.data_pt")

    root = MODELS_ROOT / "DensGNN"
    script = root / "main.py"
    base_config = root / "configs" / "cora_densgnn.yaml"
    device = "cuda" if context.device == "gpu" else context.device
    overrides = [
        f"run.seed={context.seed}",
        "dataset.loader=pyg_pt",
        f"dataset.name={context.dataset}",
        f"dataset.path={context.data_pt}",
        "algorithm.name=densgnn",
        f"algorithm.params.random_state={context.seed}",
        f"algorithm.params.device={device}",
        f"output.dir={context.model_dir}",
        "output.save_assignments=true",
        "output.save_features=true",
    ]
    overrides.extend(
        f"algorithm.params.{key}={yaml_override_value(value)}"
        for key, value in params.items()
        if value is not None
    )
    command = python_script(script) + ["--cfg", str(base_config), "--set", *overrides]
    return [Command(command, root)]


DEEPDENSGNN_PARAMS = DENSGNN_PARAMS | {
    "encoder_type",
    "candidate_k",
    "density_temperature",
    "log_density_scales",
}


def build_deep_densgnn(context: RunContext, params: Mapping[str, Any]) -> list[Command]:
    unknown = sorted(set(params) - DEEPDENSGNN_PARAMS)
    if unknown:
        raise ConfigError(f"Unknown DeepDensGNN parameters: {', '.join(unknown)}")
    if params.get("encoder_type", "gcn") not in ("gcn", "gat", "sage"):
        raise ConfigError("DeepDensGNN encoder_type must be gcn, gat or sage")
    if context.data_pt is None:
        raise ConfigError("DeepDensGNN requires experiment.data_pt")

    root = MODELS_ROOT / "DeepDensGNN"
    script = root / "main.py"
    base_config = MODELS_ROOT / "DensGNN" / "configs" / "cora_densgnn.yaml"
    device = "cuda" if context.device == "gpu" else context.device
    overrides = [
        f"run.seed={context.seed}",
        "dataset.loader=pyg_pt",
        f"dataset.name={context.dataset}",
        f"dataset.path={context.data_pt}",
        "algorithm.name=deep_densgnn",
        f"algorithm.params.random_state={context.seed}",
        f"algorithm.params.device={device}",
        f"output.dir={context.model_dir}",
        "output.save_assignments=true",
        "output.save_features=true",
    ]
    overrides.extend(
        f"algorithm.params.{key}={yaml_override_value(value)}"
        for key, value in params.items()
        if value is not None
    )
    return [Command(python_script(script) + ["--cfg", str(base_config), "--set", *overrides], root)]


DMON_FLAGS = {
    key: f"--{key}"
    for key in (
        "n_clusters",
        "epochs",
        "hidden_channels",
        "num_layers",
        "lr",
        "weight_decay",
        "dropout",
        "log_interval",
    )
}


def build_dmon(context: RunContext, params: Mapping[str, Any]) -> list[Command]:
    if context.data_pt is None:
        raise ConfigError("DMoN requires experiment.data_pt")
    script = MODELS_ROOT / "DMoN" / "main.py"
    command = python_script(script) + [
        "--data_pt",
        str(context.data_pt),
        "--output_dir",
        str(context.model_dir),
        "--device",
        context.device,
        "--seed",
        str(context.seed),
    ]
    append_params(command, params, DMON_FLAGS)
    return [Command(command, script.parent)]


KMEANS_ATTRIBUTES_FLAGS = {
    key: f"--{key}"
    for key in (
        "n_clusters",
        "n_init",
        "max_iter",
        "tol",
        "algorithm",
    )
}


def build_kmeans_attributes(
    context: RunContext,
    params: Mapping[str, Any],
) -> list[Command]:
    if context.data_pt is None:
        raise ConfigError("KMeans on attributes requires experiment.data_pt")
    script = MODELS_ROOT / "KMeansAttributes" / "main.py"
    command = python_script(script) + [
        "--data_pt",
        str(context.data_pt),
        "--output_dir",
        str(context.model_dir),
        "--seed",
        str(context.seed),
    ]
    append_params(command, params, KMEANS_ATTRIBUTES_FLAGS)
    return [Command(command, script.parent)]


NODE2VEC_KMEANS_FLAGS = {
    key: f"--{key}"
    for key in (
        "n_clusters",
        "embedding_dim",
        "walk_length",
        "context_size",
        "walks_per_node",
        "num_negative_samples",
        "p",
        "q",
        "directed",
        "epochs",
        "batch_size",
        "lr",
        "num_workers",
        "kmeans_n_init",
        "kmeans_max_iter",
        "log_interval",
    )
}


def build_node2vec_kmeans(
    context: RunContext,
    params: Mapping[str, Any],
) -> list[Command]:
    if context.data_pt is None:
        raise ConfigError("Node2Vec + KMeans requires experiment.data_pt")
    script = MODELS_ROOT / "Node2VecKMeans" / "main.py"
    command = python_script(script) + [
        "--data_pt",
        str(context.data_pt),
        "--output_dir",
        str(context.model_dir),
        "--device",
        context.device,
        "--seed",
        str(context.seed),
    ]
    append_params(command, params, NODE2VEC_KMEANS_FLAGS)
    return [Command(command, script.parent)]


GRAPH_AUTOENCODER_KMEANS_FLAGS = {
    key: f"--{key}"
    for key in (
        "n_clusters",
        "epochs",
        "hidden_channels",
        "embedding_dim",
        "lr",
        "weight_decay",
        "dropout",
        "kmeans_n_init",
        "kmeans_max_iter",
        "log_interval",
    )
}


def build_graph_autoencoder_kmeans(
    context: RunContext,
    params: Mapping[str, Any],
    *,
    directory: str,
    extra_flags: Mapping[str, str] | None = None,
) -> list[Command]:
    if context.data_pt is None:
        raise ConfigError(f"{directory} requires experiment.data_pt")
    script = MODELS_ROOT / directory / "main.py"
    command = python_script(script) + [
        "--data_pt",
        str(context.data_pt),
        "--output_dir",
        str(context.model_dir),
        "--device",
        context.device,
        "--seed",
        str(context.seed),
    ]
    flags = dict(GRAPH_AUTOENCODER_KMEANS_FLAGS)
    if extra_flags:
        flags.update(extra_flags)
    append_params(command, params, flags)
    return [Command(command, script.parent)]


def build_gae_kmeans(context: RunContext, params: Mapping[str, Any]) -> list[Command]:
    return build_graph_autoencoder_kmeans(
        context,
        params,
        directory="GAEKMeans",
    )


def build_vgae_kmeans(context: RunContext, params: Mapping[str, Any]) -> list[Command]:
    return build_graph_autoencoder_kmeans(
        context,
        params,
        directory="VGAEKMeans",
        extra_flags={"kl_weight": "--kl_weight"},
    )


def build_arga_kmeans(context: RunContext, params: Mapping[str, Any]) -> list[Command]:
    return build_graph_autoencoder_kmeans(
        context,
        params,
        directory="ARGAKMeans",
        extra_flags={
            "discriminator_hidden_channels": "--discriminator_hidden_channels",
            "discriminator_lr": "--discriminator_lr",
            "discriminator_steps": "--discriminator_steps",
            "adversarial_weight": "--adversarial_weight",
        },
    )


MODEL_REGISTRY: dict[str, CommandBuilder] = {
    "kmeans_attributes": build_kmeans_attributes,
    "node2vec_kmeans": build_node2vec_kmeans,
    "gae_kmeans": build_gae_kmeans,
    "vgae_kmeans": build_vgae_kmeans,
    "arga_kmeans": build_arga_kmeans,
    "coane": lambda context, params: build_coane(context, params, topk=False),
    "coane_topk": lambda context, params: build_coane(context, params, topk=True),
    "dgc_efr": build_dgc_efr,
    "soli": build_soli,
    "ucode": build_ucode,
    "ns4gc": build_ns4gc,
    "dgcluster": build_dgcluster,
    "dgcss": build_dgcss,
    "sanece": build_sanece,
    "egc_vgae": build_egc_vgae,
    "dsagc": build_dsagc,
    "scgc": build_scgc,
    "densgnn": build_densgnn,
    "deep_densgnn": build_deep_densgnn,
    "dmon": build_dmon,
}


CLUSTER_COUNT_PARAMETER_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "kmeans_attributes": (("n_clusters",),),
    "node2vec_kmeans": (("n_clusters",),),
    "gae_kmeans": (("n_clusters",),),
    "vgae_kmeans": (("n_clusters",),),
    "arga_kmeans": (("n_clusters",),),
    "coane": (("kmeans_clusters",),),
    "coane_topk": (("kmeans_clusters",),),
    "dgc_efr": (
        ("preae", "n_clusters"),
        ("pregae", "n_clusters"),
        ("train", "n_clusters"),
    ),
    "soli": (("kmeans_clusters",),),
    "ucode": (("kmeans_clusters",),),
    "ns4gc": (("kmeans_clusters",),),
    "dgcluster": (("n_clusters",),),
    "dgcss": (("n_clusters",),),
    "sanece": (("kmeans_clusters",),),
    "egc_vgae": (("n_clusters",),),
    "dsagc": (("n_clusters",),),
    "scgc": (("n_clusters",),),
    "dmon": (("n_clusters",),),
}


def configured_cluster_count_method(config: Mapping[str, Any]) -> str | None:
    section = config.get("cluster_count", {})
    if not isinstance(section, dict):
        raise ConfigError("cluster_count must be a mapping")
    method = section.get("method")
    if method is None:
        return None
    normalized = str(method).strip().lower().replace("_", "-")
    if normalized != "newman-riolo":
        raise ConfigError("cluster_count.method must be null or newman-riolo")
    return normalized


def validate_cluster_count_config(config: Mapping[str, Any]) -> None:
    configured_cluster_count_method(config)
    section = config.get("cluster_count", {})
    allowed_section = {"method", "newman_riolo"}
    unknown_section = sorted(set(section) - allowed_section)
    if unknown_section:
        raise ConfigError(
            "Unknown cluster_count options: " + ", ".join(unknown_section)
        )
    options = section.get("newman_riolo", {})
    if not isinstance(options, dict):
        raise ConfigError("cluster_count.newman_riolo must be a mapping")
    allowed_options = {"burn_in", "timeout_seconds"}
    unknown_options = sorted(set(options) - allowed_options)
    if unknown_options:
        raise ConfigError(
            "Unknown Newman-Riolo options: " + ", ".join(unknown_options)
        )
    burn_in = options.get("burn_in", 1000)
    if isinstance(burn_in, bool) or not isinstance(burn_in, int):
        raise ConfigError("cluster_count.newman_riolo.burn_in must be an integer")
    if burn_in < 0 or burn_in >= 10_000:
        raise ConfigError(
            "cluster_count.newman_riolo.burn_in must satisfy 0 <= burn_in < 10000"
        )
    timeout = options.get("timeout_seconds")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise ConfigError(
            "cluster_count.newman_riolo.timeout_seconds must be positive or null"
        )


def inject_cluster_count(
    model_name: str,
    params: Mapping[str, Any],
    selected_k: int | str | None,
) -> dict[str, Any]:
    updated = copy.deepcopy(dict(params))
    if selected_k is None:
        return updated
    for path in CLUSTER_COUNT_PARAMETER_PATHS.get(model_name, ()):
        cursor = updated
        for key in path[:-1]:
            value = cursor.get(key)
            if not isinstance(value, dict):
                raise ConfigError(
                    f"Cannot inject K into models.{model_name}.params.{'.'.join(path)}"
                )
            cursor = value
        cursor[path[-1]] = selected_k
    return updated


def ground_truth_cluster_count(data: Any) -> int:
    labels = get_attr(data, "y")
    if labels is None:
        raise ConfigError(
            "cluster_count.method is null, but data.y is unavailable. "
            "Use method: newman-riolo or provide an explicit supported policy."
        )
    count = int(np.unique(as_label_vector(labels)).size)
    if count < 2:
        raise ConfigError(f"data.y defines only {count} distinct cluster(s)")
    return count


def resolve_cluster_count(
    config: Mapping[str, Any],
    selected_models: list[str],
    data: Any,
    dataset_root: Path,
) -> dict[str, Any] | None:
    applicable = [
        model for model in selected_models if model in CLUSTER_COUNT_PARAMETER_PATHS
    ]
    if not applicable:
        return None

    method = configured_cluster_count_method(config)
    if method is None:
        selected_k = ground_truth_cluster_count(data)
        return {
            "configured_method": None,
            "resolved_method": "data.y",
            "selected_k": selected_k,
            "applies_to": applicable,
            "artifact_dir": None,
            "deterministic": True,
        }

    from newman_riolo import NewmanRioloError, estimate_newman_riolo

    identifier = datetime.now().strftime("%Y%m%d_%H%M%S_%f_newman_riolo")
    artifact_dir = dataset_root / "cluster_count" / identifier
    try:
        result = estimate_newman_riolo(
            data,
            config.get("cluster_count", {}).get("newman_riolo", {}),
            artifact_dir,
        )
    except NewmanRioloError as exc:
        raise ConfigError(str(exc)) from exc
    return {
        "configured_method": method,
        "resolved_method": result.summary["resolved_method"],
        "selected_k": result.selected_k,
        "applies_to": applicable,
        "artifact_dir": str(result.artifact_dir),
        "deterministic": result.summary["deterministic"],
        "warnings": result.summary["warnings"],
    }


def resolve_repo_path(value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ConfigError("Configuration root must be a YAML mapping")
    for section in ("experiment", "output", "metrics", "models"):
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"Missing or invalid configuration section: {section}")
    return config


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    experiment = config["experiment"]
    for cli_name, config_name in (
        ("dataset", "dataset"),
        ("data_pt", "data_pt"),
        ("device", "device"),
        ("seed", "seed"),
        ("runs", "runs"),
        ("timeout_seconds", "timeout_seconds"),
    ):
        value = getattr(args, cli_name, None)
        if value is not None:
            experiment[config_name] = value


def validate_config(config: Mapping[str, Any]) -> list[str]:
    validate_cluster_count_config(config)
    experiment = config["experiment"]
    dataset = str(experiment.get("dataset", "")).strip()
    if not dataset or Path(dataset).name != dataset or dataset in {".", ".."}:
        raise ConfigError("experiment.dataset must be a non-empty path-safe name")
    if int(experiment.get("runs", 0)) < 1:
        raise ConfigError("experiment.runs must be at least 1")
    timeout = experiment.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ConfigError("experiment.timeout_seconds must be a finite positive number")
    ignore_ground_truth = experiment.get("ignore_ground_truth", False)
    if not isinstance(ignore_ground_truth, bool):
        raise ConfigError("experiment.ignore_ground_truth must be true or false")
    if not isinstance(experiment.get("select_best_nmi", False), bool):
        raise ConfigError("experiment.select_best_nmi must be true or false")

    output = config["output"]
    save_embeddings = output.get("save_embeddings", True)
    if not isinstance(save_embeddings, bool):
        raise ConfigError("output.save_embeddings must be true or false")
    generate_figures = output.get("generate_figures", False)
    if not isinstance(generate_figures, bool):
        raise ConfigError("output.generate_figures must be true or false")
    if generate_figures:
        required_artifacts = {
            "save_labels": output.get("save_labels", True),
            "save_embeddings": save_embeddings,
            "save_metadata": output.get("save_metadata", True),
        }
        disabled = [name for name, enabled in required_artifacts.items() if not enabled]
        if disabled:
            raise ConfigError(
                "output.generate_figures requires "
                + ", ".join(f"output.{name}=true" for name in disabled)
            )
        numeric_figure_options = {
            "figure_dpi": int,
            "figure_max_points": int,
            "figure_perplexity": (int, float),
            "figure_tsne_iterations": int,
        }
        for name, expected_type in numeric_figure_options.items():
            if name in output and (
                isinstance(output[name], bool)
                or not isinstance(output[name], expected_type)
                or output[name] <= 0
            ):
                raise ConfigError(f"output.{name} must be a positive number")

    models = config["models"]
    unknown_models = sorted(set(models) - set(MODEL_REGISTRY))
    missing_models = sorted(set(MODEL_REGISTRY) - set(models))
    if unknown_models:
        raise ConfigError(f"Unknown models in configuration: {', '.join(unknown_models)}")
    if missing_models:
        warnings.warn(
            "Models omitted from configuration will be unavailable for this run: "
            + ", ".join(missing_models),
            ConfigWarning,
            stacklevel=2,
        )
    for name, model_config in models.items():
        if not isinstance(model_config, dict) or "enabled" not in model_config:
            raise ConfigError(f"models.{name} must define enabled and params")
        if not isinstance(model_config.get("params", {}), dict):
            raise ConfigError(f"models.{name}.params must be a mapping")

    requested = config["metrics"].get("include", [])
    if not isinstance(requested, list):
        raise ConfigError("metrics.include must be a list")
    if ignore_ground_truth:
        forbidden = sorted(SUPERVISED_METRICS.intersection(requested))
        if forbidden:
            raise ConfigError(
                "experiment.ignore_ground_truth=true is incompatible with "
                "supervised metrics: " + ", ".join(forbidden)
            )
    return validate_requested_metrics(requested)


def without_ground_truth(data: Any) -> tuple[Any, list[int] | None]:
    """Return an isolated graph copy with data.y removed."""

    labels = get_attr(data, "y")
    label_shape = None
    if labels is not None:
        label_values = labels.cpu() if isinstance(labels, torch.Tensor) else labels
        label_shape = [int(value) for value in np.asarray(label_values).shape]

    sanitized = copy.deepcopy(data)
    if isinstance(sanitized, dict):
        sanitized.pop("y", None)
    elif hasattr(sanitized, "y"):
        sanitized.y = None
    return sanitized, label_shape


def materialize_ground_truth_free_input(
    data: Any,
    source_path: Path,
    dataset_root: Path,
    label_shape: list[int] | None,
) -> Path:
    """Persist the label-free graph used by every subprocess in the benchmark."""

    input_dir = dataset_root / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_path = input_dir / "data_without_ground_truth.pt"
    temporary_path = input_dir / f".data_without_ground_truth.{os.getpid()}.tmp"
    torch.save(data, temporary_path)
    temporary_path.replace(output_path)
    write_metadata(
        input_dir / "data_without_ground_truth.yaml",
        {
            "source_data_pt": str(source_path),
            "derived_data_pt": str(output_path),
            "ground_truth_removed": True,
            "source_ground_truth_shape": label_shape,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return output_path


def generate_configured_run_figures(
    run_root: Path,
    data_pt: Path,
    output_config: Mapping[str, Any],
) -> None:
    """Generate optional figures in an isolated process.

    Figure generation calls native dimensionality-reduction libraries.  A
    native crash must not terminate the benchmark after model results have
    already been recorded, so visualization runs in its own subprocess.
    """

    if not output_config.get("generate_figures", False):
        return

    figures_dir = run_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_figures.py"),
        "--run-dir",
        str(run_root),
        "--data-pt",
        str(data_pt),
        "--dpi",
        str(int(output_config.get("figure_dpi", 180))),
        "--max-points",
        str(int(output_config.get("figure_max_points", 5000))),
        "--perplexity",
        str(float(output_config.get("figure_perplexity", 30.0))),
        "--tsne-iterations",
        str(int(output_config.get("figure_tsne_iterations", 1000))),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        (figures_dir / "stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (figures_dir / "stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            if completed.returncode < 0:
                detail = (
                    f"subprocess terminated by signal {-completed.returncode}"
                    + (f": {detail}" if detail else "")
                )
            warnings.warn(
                f"Figure generation failed for run {run_root.name} "
                f"(exit code {completed.returncode}): {detail}",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        manifest_path = figures_dir / "manifest.yaml"
        report = {}
        if manifest_path.is_file():
            with manifest_path.open(encoding="utf-8") as stream:
                loaded = yaml.safe_load(stream)
            report = loaded if isinstance(loaded, dict) else {}
        print(
            f"Figures: {len(report.get('generated', []))} generated in "
            f"{figures_dir}",
            flush=True,
        )
        for skipped in report.get("skipped", []):
            if "projection skipped" in str(skipped).lower():
                warnings.warn(str(skipped), RuntimeWarning, stacklevel=2)
    except Exception as exc:
        warnings.warn(
            f"Figure generation failed for run {run_root.name}: "
            f"{type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def select_models(config: Mapping[str, Any], requested_model: str) -> list[str]:
    if requested_model != "all":
        model_config = config["models"].get(requested_model)
        if model_config is None:
            raise ConfigError(
                f"Model '{requested_model}' is not defined in the configuration; "
                f"add a models.{requested_model} entry"
            )
        if not model_config.get("enabled", False):
            raise ConfigError(
                f"Model '{requested_model}' is disabled in the configuration; set models.{requested_model}.enabled=true"
            )
        return [requested_model]
    selected = [
        name
        for name in MODEL_REGISTRY
        if config["models"].get(name, {}).get("enabled", False)
    ]
    if not selected:
        raise ConfigError("No enabled models were selected")
    return selected


def current_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def make_run_id(seed: int, run_number: int, total_runs: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = f"_run{run_number:02d}" if total_runs > 1 else ""
    return f"{timestamp}_seed{seed}{suffix}"


def subprocess_environment(device: str, seed: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(seed)
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


def effective_model_device(model_name: str, configured_device: str) -> str:
    return "cpu" if model_name in {"soli", "kmeans_attributes"} else configured_device


def newest_native_labels(model_dir: Path) -> Path:
    candidates = [path for path in model_dir.glob("*_labels.npy") if path.name != "labels.npy"]
    assignments_path = model_dir / "assignments.npy"
    if assignments_path.is_file():
        candidates.append(assignments_path)
    if not candidates:
        raise FileNotFoundError(
            f"Model did not produce a *_labels.npy or assignments.npy file in {model_dir}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def native_embeddings_path(model_dir: Path) -> Path:
    """Locate the learned node representation emitted by a model adapter."""
    for filename in ("embeddings.npy", "features.npy"):
        candidate = model_dir / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No model embeddings found in {model_dir}; expected embeddings.npy or features.npy"
    )


def normalize_model_embeddings(
    model_dir: Path,
    normalized_path: Path,
    expected_rows: int,
) -> np.ndarray:
    """Normalize and validate the learned node representation from a baseline."""

    native_path = native_embeddings_path(model_dir)
    if native_path != normalized_path:
        shutil.copy2(native_path, normalized_path)

    embeddings = np.load(normalized_path, allow_pickle=False)
    if embeddings.ndim != 2:
        raise ValueError(
            f"Model embeddings must be two-dimensional, got shape {embeddings.shape}"
        )
    if embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"Embedding row count ({embeddings.shape[0]}) does not match graph node "
            f"count ({expected_rows})"
        )
    if not np.issubdtype(embeddings.dtype, np.number):
        raise TypeError(f"Model embeddings must be numeric, got dtype {embeddings.dtype}")
    return embeddings


def load_native_algorithm_metadata(model_dir: Path) -> dict[str, Any]:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.is_file():
        return {}
    with metrics_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    algorithm_metadata = payload.get("algorithm", {})
    if not isinstance(algorithm_metadata, dict):
        raise ValueError(f"Invalid algorithm metadata in {metrics_path}")
    return algorithm_metadata


def tail_text(path: Path, max_characters: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_characters:].strip()


def write_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(dict(metadata), stream, sort_keys=False, allow_unicode=True)
    temporary.replace(path)


def append_metrics_row(csv_path: Path, metric_names: list[str], row: Mapping[str, Any]) -> None:
    fieldnames = IDENTIFICATION_COLUMNS + metric_names
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    if exists:
        with csv_path.open(newline="", encoding="utf-8") as stream:
            existing_header = next(csv.reader(stream), [])
        if existing_header != fieldnames:
            raise ConfigError(
                f"Existing run CSV schema differs from configured metrics: {csv_path}. "
                "Use a different csv_filename or restore the original metrics.include list."
            )
    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def remaining_model_time(deadline: float, timeout_seconds: float, stage: str) -> float:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise ModelTimeoutError(
            f"OOT (out of time): exceeded {timeout_seconds:g} seconds during {stage}"
        )
    return remaining


def stop_process_group(process: subprocess.Popen) -> None:
    """Stop a stage and its descendants, including workers that ignore SIGTERM."""
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_bounded_command(
    command: Command,
    environment: Mapping[str, str],
    stdout: Any,
    stderr: Any,
    deadline: float,
    timeout_seconds: float,
    stage: str,
) -> None:
    remaining_model_time(deadline, timeout_seconds, stage)
    process = subprocess.Popen(
        command.argv,
        cwd=command.cwd,
        env=environment,
        stdout=stdout,
        stderr=stderr,
        text=True,
        start_new_session=True,
    )
    try:
        process.wait(timeout=remaining_model_time(deadline, timeout_seconds, stage))
    except subprocess.TimeoutExpired as exc:
        stop_process_group(process)
        raise ModelTimeoutError(
            f"OOT (out of time): exceeded {timeout_seconds:g} seconds during {stage}"
        ) from exc
    except BaseException:
        stop_process_group(process)
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command.argv)


def evaluate_model_with_deadline(
    context: RunContext,
    data: Any,
    requested_metrics: list[str],
    embeddings: np.ndarray | None,
    environment: Mapping[str, str],
    deadline: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Evaluate in a process so native metric calculations share the hard deadline."""
    metrics = [metric for metric in requested_metrics if metric != "runtime_seconds"]
    if not metrics:
        return {}
    remaining_model_time(deadline, timeout_seconds, "evaluation")
    temporary_data = None
    data_pt = context.data_pt
    if data_pt is None:
        # Direct callers may supply an in-memory graph instead of a dataset path.
        temporary_data = context.model_dir / ".evaluation_data.pt"
        torch.save(data, temporary_data)
        data_pt = temporary_data
    result_path = context.model_dir / "evaluation.json"
    command = python_script(REPO_ROOT / "scripts" / "evaluate.py") + [
        "--data-pt", str(data_pt),
        "--labels", str(context.model_dir / "labels.npy"),
        "--output-json", str(result_path),
        "--metrics", *metrics,
    ]
    if embeddings is not None:
        command += ["--embeddings", str(context.model_dir / "embeddings.npy")]
    try:
        with (context.model_dir / "stdout.log").open("a", encoding="utf-8") as stdout, (
            context.model_dir / "stderr.log"
        ).open("a", encoding="utf-8") as stderr:
            stdout.write(f"evaluation: {shlex.join(command)}\n")
            stdout.flush()
            run_bounded_command(
                Command(command, REPO_ROOT), environment, stdout, stderr,
                deadline, timeout_seconds, "evaluation",
            )
        remaining_model_time(deadline, timeout_seconds, "evaluation")
        return json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        if temporary_data is not None:
            temporary_data.unlink(missing_ok=True)


def run_model(
    model_name: str,
    context: RunContext,
    model_params: Mapping[str, Any],
    data: Any,
    requested_metrics: list[str],
    output_config: Mapping[str, Any],
    git_commit: str,
    cluster_count: Mapping[str, Any] | None = None,
    select_best_nmi: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    context.model_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = context.model_dir / "stdout.log"
    stderr_path = context.model_dir / "stderr.log"
    metadata_path = context.model_dir / "metadata.yaml"
    commands = MODEL_REGISTRY[model_name](context, model_params)
    execution_device = effective_model_device(model_name, context.device)
    start_time = datetime.now(timezone.utc)
    started = time.perf_counter()
    deadline = started + timeout_seconds
    status = "success"
    error = ""
    metric_values: dict[str, Any] = {}
    native_algorithm_metadata: dict[str, Any] = {}
    embedding_error: str | None = None
    embedding_shape: list[int] | None = None
    embedding_dtype: str | None = None
    normalized_labels = context.model_dir / "labels.npy"
    normalized_embeddings = context.model_dir / "embeddings.npy"
    save_embeddings = bool(output_config.get("save_embeddings", True))
    embeddings_required = bool(EMBEDDING_METRICS.intersection(requested_metrics))
    selection_info: dict[str, Any] = {
        "requested": select_best_nmi,
        "criterion": "native",
    }
    selection_enabled = select_best_nmi and get_attr(data, "y") is not None
    selection_dir = context.model_dir / "selection"
    truth_path = selection_dir / "ground_truth.npy"
    if select_best_nmi and not selection_enabled:
        selection_info["reason"] = "data.y unavailable; using native result selection"
        warnings.warn(selection_info["reason"], RuntimeWarning, stacklevel=2)

    try:
        environment = subprocess_environment(execution_device, context.seed)
        # An inherited environment must never enable selection in another run.
        for key in list(environment):
            if key.startswith("BENCHMARK_NMI_") or key == "BENCHMARK_SELECT_BEST_NMI":
                del environment[key]
        if selection_enabled:
            selection_dir.mkdir()
            np.save(truth_path, as_label_vector(get_attr(data, "y")))
            environment.update({
                "BENCHMARK_SELECT_BEST_NMI": "1",
                "BENCHMARK_NMI_LABELS": str(truth_path),
                "BENCHMARK_NMI_DIR": str(selection_dir),
                "BENCHMARK_NMI_SEED": str(context.seed),
                "PYTHONPATH": os.pathsep.join(filter(None, [
                    str(MODELS_ROOT / "_shared"), environment.get("PYTHONPATH", "")
                ])),
            })
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            for stage, command in enumerate(commands, start=1):
                stdout.write(f"stage {stage}/{len(commands)}: {shlex.join(command.argv)}\n")
                stdout.flush()
                run_bounded_command(
                    command,
                    (environment if stage == len(commands) else {
                        key: value for key, value in environment.items()
                        if key != "BENCHMARK_SELECT_BEST_NMI"
                    }),
                    stdout,
                    stderr,
                    deadline,
                    timeout_seconds,
                    f"stage {stage}/{len(commands)}",
                )
        remaining_model_time(deadline, timeout_seconds, "output normalization")
        native_labels = newest_native_labels(context.model_dir)
        if model_name in {"densgnn", "deep_densgnn"}:
            native_algorithm_metadata = load_native_algorithm_metadata(
                context.model_dir
            )
        shutil.copy2(native_labels, normalized_labels)
        predictions = load_pred_labels(normalized_labels)
        embeddings = None
        if save_embeddings or embeddings_required or selection_enabled:
            try:
                embeddings = normalize_model_embeddings(
                    context.model_dir,
                    normalized_embeddings,
                    expected_rows=len(predictions),
                )
                embedding_shape = [int(value) for value in embeddings.shape]
                embedding_dtype = str(embeddings.dtype)
            except FileNotFoundError as exc:
                embedding_error = str(exc)
                if embeddings_required or selection_enabled:
                    raise
        if selection_enabled:
            # Include the native final output as a candidate as well. Some
            # methods perform a final inference after the last optimizer step.
            shared_path = str(MODELS_ROOT / "_shared")
            if shared_path not in sys.path:
                sys.path.insert(0, shared_path)
            from benchmark_selection import BestNMISelection

            selector = BestNMISelection(
                as_label_vector(get_attr(data, "y")), selection_dir, context.seed
            )
            had_training_candidates = selector.best is not None
            if model_name != "kmeans_attributes" and not had_training_candidates:
                raise RuntimeError(f"{model_name} did not report any training candidates for NMI selection")
            # Native outputs may come from early stopping, a selected checkpoint,
            # or a separate clustering phase; do not invent an epoch for them.
            selector.consider(predictions, embeddings, None, stage="native_output")
            shutil.copy2(selection_dir / "labels.npy", normalized_labels)
            shutil.copy2(selection_dir / "embeddings.npy", normalized_embeddings)
            predictions = load_pred_labels(normalized_labels)
            embeddings = np.load(normalized_embeddings, allow_pickle=False)
            embedding_shape = [int(value) for value in embeddings.shape]
            embedding_dtype = str(embeddings.dtype)
            selection_info.update(selector.best)
            selection_info["criterion"] = "best_nmi"
            selection_info["supervised"] = True
            selection_info["single_result"] = model_name == "kmeans_attributes"
            print(f"{model_name}: best NMI={selector.best['nmi']:.6f}, "
                  f"epoch={selector.best['epoch']}, stage={selector.best['stage']}", flush=True)
        metric_values = evaluate_model_with_deadline(
            context, data, requested_metrics, embeddings, environment,
            deadline, timeout_seconds,
        )
        remaining_model_time(deadline, timeout_seconds, "evaluation")
    except ModelTimeoutError as exc:
        status = "failed"
        error = str(exc)
        with stderr_path.open("a", encoding="utf-8") as stderr:
            stderr.write(error + "\n")
    except Exception as exc:  # each baseline is an isolated failure boundary
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        stderr_tail = tail_text(stderr_path)
        if stderr_tail:
            error = f"{error} | stderr: {stderr_tail}"

    runtime = time.perf_counter() - started
    end_time = datetime.now(timezone.utc)
    if "runtime_seconds" in requested_metrics:
        metric_values["runtime_seconds"] = runtime

    truth_path.unlink(missing_ok=True)
    if not save_embeddings:
        normalized_embeddings.unlink(missing_ok=True)
        (selection_dir / "embeddings.npy").unlink(missing_ok=True)

    row: dict[str, Any] = {
        "dataset": context.dataset,
        "run_id": context.model_dir.parent.name,
        "model": model_name,
        "seed": context.seed,
        "status": status,
        "error": error,
    }
    for metric in requested_metrics:
        row[metric] = metric_values.get(metric, math.nan)

    metadata = {
        "dataset": context.dataset,
        "data_pt": str(context.data_pt) if context.data_pt else None,
        "model": model_name,
        "seed": context.seed,
        "device": execution_device,
        "cuda_used": False if execution_device == "cpu" else None,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "runtime_seconds": runtime,
        "timeout_seconds": timeout_seconds,
        "error_code": "OOT" if error.startswith("OOT (out of time)") else None,
        "status": status,
        "error": error or None,
        "model_params": dict(model_params),
        "commands": [command.argv for command in commands],
        "git_commit": git_commit,
        "ground_truth_available": get_attr(data, "y") is not None,
        "embeddings_path": str(normalized_embeddings) if normalized_embeddings.is_file() else None,
        "embeddings_saved": normalized_embeddings.is_file(),
        "embeddings_shape": embedding_shape,
        "embeddings_dtype": embedding_dtype,
        "embeddings_error": embedding_error,
        "internal_metrics_space": "embedding",
        "cluster_count": dict(cluster_count) if cluster_count is not None else None,
        "cluster_count_applied": model_name in CLUSTER_COUNT_PARAMETER_PATHS,
        "result_selection": selection_info,
    }
    if model_name in {"densgnn", "deep_densgnn"}:
        metadata["encoder_type"] = native_algorithm_metadata.get(
            "encoder_type", model_params.get("encoder_type", "gcn")
        )
        metadata["loss_history"] = native_algorithm_metadata.get(
            "loss_history",
            [],
        )
        metadata["cluster_support_stats"] = native_algorithm_metadata.get(
            "cluster_support_stats",
            [],
        )
        if model_name == "deep_densgnn":
            metadata["density_scale_info"] = native_algorithm_metadata.get("density_scale_info", {})
            metadata["scale_history"] = native_algorithm_metadata.get("scale_history", [])
    if output_config.get("save_metadata", True):
        write_metadata(metadata_path, metadata)
    if not output_config.get("save_logs", True):
        stdout_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)
    if not output_config.get("save_labels", True):
        normalized_labels.unlink(missing_ok=True)
        (selection_dir / "labels.npy").unlink(missing_ok=True)
        for native in context.model_dir.glob("*_labels.npy"):
            native.unlink(missing_ok=True)
        (context.model_dir / "assignments.npy").unlink(missing_ok=True)
        (context.model_dir / "assignments.csv").unlink(missing_ok=True)
    return row


def dry_run(
    models: list[str],
    config: Mapping[str, Any],
    output_root: Path,
    data_pt: Path | None,
    selected_k: int | str | None,
) -> None:
    experiment = config["experiment"]
    print(f"Model timeout: {experiment.get('timeout_seconds', DEFAULT_TIMEOUT_SECONDS):g}s per model/seed (all stages and evaluation)")
    context_root = output_root / str(experiment["dataset"]) / "runs" / "DRY_RUN"
    for model_name in models:
        context = RunContext(
            dataset=str(experiment["dataset"]),
            data_pt=data_pt,
            device=str(experiment["device"]),
            seed=int(experiment["seed"]),
            model_dir=context_root / model_name,
        )
        params = inject_cluster_count(
            model_name,
            config["models"][model_name]["params"],
            selected_k,
        )
        commands = MODEL_REGISTRY[model_name](context, params)
        print(f"[{model_name}]")
        for command in commands:
            print(f"  cwd={command.cwd}")
            print(f"  {shlex.join(command.argv)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible clustering benchmarks")
    parser.add_argument("--config", default="configs/global_config.yaml")
    parser.add_argument("--model", required=True, choices=[*MODEL_REGISTRY, "all"])
    parser.add_argument("--dataset", default=None, help="Override experiment.dataset")
    parser.add_argument("--data-pt", "--data_pt", dest="data_pt", default=None)
    parser.add_argument("--device", default=None, help="Override experiment.device")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment.seed")
    parser.add_argument("--runs", type=int, default=None, help="Override experiment.runs")
    parser.add_argument("--timeout-seconds", dest="timeout_seconds", type=float, default=None,
                        help="Wall-clock limit per model/seed, including all stages and evaluation (default: 600)")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands without training")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = resolve_repo_path(args.config)
        assert config_path is not None
        config = load_config(config_path)
        apply_overrides(config, args)
        requested_metrics = validate_config(config)
        models = select_models(config, args.model)
        experiment = config["experiment"]
        data_pt = resolve_repo_path(experiment.get("data_pt"))
        output_root = resolve_repo_path(experiment.get("output_root"))
        if output_root is None:
            raise ConfigError("experiment.output_root cannot be empty")
        if data_pt is not None and not data_pt.is_file():
            raise ConfigError(f"experiment.data_pt not found: {data_pt}")

        data = load_data(data_pt) if data_pt else None
        if data is None:
            raise ConfigError("A data_pt is currently required for centralized evaluation")
        dataset = str(experiment["dataset"])
        dataset_root = output_root / dataset
        if experiment.get("ignore_ground_truth", False):
            assert data_pt is not None
            source_data_pt = data_pt
            data, source_label_shape = without_ground_truth(data)
            if args.dry_run:
                data_pt = dataset_root / "inputs" / "data_without_ground_truth.pt"
            else:
                data_pt = materialize_ground_truth_free_input(
                    data,
                    source_data_pt,
                    dataset_root,
                    source_label_shape,
                )
                print(
                    "Ground truth ignored: all models will use "
                    f"{data_pt}",
                    flush=True,
                )
        needs_cluster_count = any(
            model in CLUSTER_COUNT_PARAMETER_PATHS for model in models
        )

        if args.dry_run:
            print(f"Best NMI selection: {experiment.get('select_best_nmi', False)} "
                  "(supervised; falls back to native selection without data.y)")
            method = configured_cluster_count_method(config)
            if not needs_cluster_count:
                preview_k: int | str | None = None
            elif method is None:
                preview_k = ground_truth_cluster_count(data)
                print(f"Cluster count: K={preview_k} from data.y")
            else:
                preview_k = "<newman-riolo-K>"
                print("Cluster count: K will be estimated by Newman-Riolo")
            dry_run(models, config, output_root, data_pt, preview_k)
            return 0

        cluster_count = resolve_cluster_count(config, models, data, dataset_root)
        if cluster_count is not None:
            print(
                f"Cluster count: K={cluster_count['selected_k']} from "
                f"{cluster_count['resolved_method']}",
                flush=True,
            )
            for warning in cluster_count.get("warnings", []):
                warnings.warn(warning, RuntimeWarning, stacklevel=2)
        git_commit = current_git_commit()
        failures = 0

        total_runs = int(experiment["runs"])
        initial_seed = int(experiment["seed"])
        for run_number in range(1, total_runs + 1):
            run_seed = initial_seed + run_number - 1
            run_id = make_run_id(run_seed, run_number, total_runs)
            run_root = dataset_root / "runs" / run_id
            csv_path = run_root / str(config["output"].get("csv_filename", "metrics.csv"))
            for model_name in models:
                print(f"Running {model_name} ({run_id})", flush=True)
                context = RunContext(
                    dataset=dataset,
                    data_pt=data_pt,
                    device=str(experiment["device"]),
                    seed=run_seed,
                    model_dir=run_root / model_name,
                )
                model_params = inject_cluster_count(
                    model_name,
                    config["models"][model_name]["params"],
                    cluster_count["selected_k"] if cluster_count is not None else None,
                )
                row = run_model(
                    model_name,
                    context,
                    model_params,
                    data,
                    requested_metrics,
                    config["output"],
                    git_commit,
                    cluster_count,
                    select_best_nmi=experiment.get("select_best_nmi", False),
                    timeout_seconds=experiment.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
                )
                append_metrics_row(csv_path, requested_metrics, row)
                print(f"{model_name}: {row['status']} ({row.get('runtime_seconds', 0):.3f}s)")
                if row["status"] == "failed":
                    failures += 1
                    if not experiment.get("continue_on_error", True):
                        generate_configured_run_figures(
                            run_root, data_pt, config["output"]
                        )
                        return 1
            generate_configured_run_figures(run_root, data_pt, config["output"])
        return 1 if failures else 0
    except (ConfigError, OSError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
