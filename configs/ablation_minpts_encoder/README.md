# DensGNN: min_pts × encoder

This sweep runs **DensGNN**, with selectable GCN, GAT, and SAGEConv encoders.
There are **270 configurations**: 30 min_pts values × 3 encoders × 3 datasets,
with **one run per configuration**.

```text
configs/ablation_minpts_encoder/
├── experiments.csv
├── cora/{gcn,gat,sageconv}/minpts_{02,04,...,58,60}.yaml
├── citeseer/{gcn,gat,sageconv}/minpts_{02,04,...,58,60}.yaml
└── polblogs/{gcn,gat,sageconv}/minpts_{02,04,...,58,60}.yaml
```

`experiments.csv` lists every configuration, seed, encoder, min_pts value,
and output directory. The `sageconv` directory uses `encoder_type: sage`,
the parameter name accepted by the runner.

## Controlled parameters

Every configuration sets both parameters to the same sweep value:

```yaml
models:
  densgnn:
    enabled: true
    params:
      encoder_type: gcn  # gcn, gat, or sage
      hdbscan_min_samples: 2
      mrd_k: 2
```

Values are **2, 4, 6, ..., 58, 60**. This measures their **joint effect**;
it does not isolate changing `hdbscan_min_samples` from changing `mrd_k`.
The former controls HDBSCAN cluster discovery; the latter controls the
neighbor rank used by DensGNN's MRD and adaptive support weighting.
`hdbscan_min_cluster_size` is held fixed within each dataset.

Other settings are recovered from `configs/ablation_distance/<dataset>/adaptive.yaml`:

| Setting | Cora | CiteSeer | PolBlogs |
| --- | ---: | ---: | ---: |
| Epochs | 200 | 200 | 200 |
| Warm-up epochs | 5 | 5 | 20 |
| Learning rate | 0.005 | 0.005 | 0.005 |
| Hidden dimensions | 128 | 128 | 128 |
| Embedding dimensions | 32 | 32 | 32 |
| Target refresh interval | 1 | 1 | 5 |
| HDBSCAN min cluster size | 5 | 5 | 10 |

All configurations use **seed 2030**, `runs: 1`, `device: cuda`,
`point_selection: adaptive`, `support_distance: mrd`, and
`select_best_nmi: false`, preserving the current ablation settings.
The common seed controls initialization comparisons within each encoder;
different architectures have different parameterizations.
One run per point does not estimate variability across random seeds.

The decoder, reconstruction objective, clustering loss, HDBSCAN logic,
support selection, and distance definitions retain their existing implementations.
GCN remains the default encoder when `encoder_type` is omitted. GAT uses
one attention head per layer; SAGEConv uses mean aggregation. Both follow
the encoder definitions already present in DeepDensGNN, without enabling
DeepDensGNN's learned density-scale mixture.

## Results

Outputs are isolated by dataset, encoder, and min_pts value:

```text
results/ablation_minpts_encoder/<dataset>_<encoder>_minpts_<value>/
└── runs/<run_id>/
    ├── metrics.csv
    └── densgnn/
```

Labels, embeddings, logs, metadata, and figures use the existing runner.
`encoder_type` is recorded in native and benchmark metadata. Existing metric
selection and the **600-second limit per model/run, including evaluation**,
are preserved. A timeout records a failed attempt; it does not automatically retry.

## Validate or run

From the repository root, with the project environment activated:

```bash
# Inspect one command without training.
python scripts/main.py --config configs/ablation_minpts_encoder/cora/gat/minpts_02.yaml --model densgnn --dry-run

# Run one experiment.
python scripts/main.py --config configs/ablation_minpts_encoder/cora/gat/minpts_02.yaml --model densgnn
```

Run all 270 configurations sequentially:

```bash
for config in configs/ablation_minpts_encoder/*/*/minpts_*.yaml; do
  python scripts/main.py --config "$config" --model densgnn
done
```

The configuration files do not start training automatically.
