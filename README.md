# Algorithms Benchmark Runner

This repository is being adapted to run graph clustering algorithms with a shared `data.pt` input format.

## Input Format

The expected input is a PyTorch/PyG-style file containing:

```text
data.x           node features
data.y           node labels/classes
data.edge_index  graph edges with shape [2, num_edges]
```

Example path:

```bash
datasets/custom/scatt_graph.pt
```

## Root Runner

The adapted models are available through the root dispatcher:

```bash
python3 main.py --model MODEL_NAME --data_pt PATH_TO_DATA_PT --output_dir outputs --epochs 300 --device cpu
```

Currently supported in the root `main.py`:

```text
coane
coane_topk
dgc_efr
dgcluster
dsagc
egc_vgae
ns4gc
sanece
soli
ucode
```

## CoANE

Standard CoANE:

```bash
python3 main.py \
  --model coane \
  --data_pt datasets/custom/scatt_graph.pt \
  --output_dir outputs \
  --epochs 300 \
  --device cpu
```

Top-k CoANE:

```bash
python3 main.py \
  --model coane_topk \
  --data_pt datasets/custom/scatt_graph.pt \
  --output_dir outputs \
  --epochs 300 \
  --device cpu
```

Outputs:

```text
outputs/CoANE_<YYYYMMDD_HHMM>_labels.npy
outputs/CoANE_topk_<YYYYMMDD_HHMM>_labels.npy
```

## DGC-EFR

```bash
python3 main.py \
  --model dgc_efr \
  --data_pt datasets/custom/scatt_graph.pt \
  --output_dir outputs \
  --epochs 300 \
  --device cpu
```

The root runner executes:

```text
1. DGC-EFR/preae.py
2. DGC-EFR/pregae.py
3. DGC-EFR/dgc_efr.py
```

Outputs:

```text
outputs/dgc_efr_pretrain/
outputs/DGC_<YYYYMMDD_HHMM>_labels.npy
```

## SOLI

```bash
python3 main.py \
  --model soli \
  --data_pt datasets/custom/scatt_graph.pt \
  --output_dir outputs \
  --epochs 300 \
  --device cpu
```

Output:

```text
outputs/SOLI_<YYYYMMDD_HHMM>_labels.npy
```

## SANECE

```bash
python3 main.py \
  --model sanece \
  --data_pt datasets/custom/scatt_graph.pt \
  --output_dir outputs \
  --epochs 300 \
  --device cpu
```

Output:

```text
outputs/SANECE_<YYYYMMDD_HHMM>_labels.npy
```

## UCODE

```bash
python3 main.py \
  --model ucode \
  --data_pt datasets/custom/scatt_graph.pt \
  --output_dir outputs \
  --epochs 300 \
  --device cpu
```

Output:

```text
outputs/UCODE_<YYYYMMDD_HHMM>_labels.npy
```

## NS4GC

```bash
python3 main.py \
  --model ns4gc \
  --data_pt datasets/custom/scatt_graph.pt \
  --output_dir outputs \
  --epochs 300 \
  --device cpu
```

Output:

```text
outputs/NS4GC_<YYYYMMDD_HHMM>_labels.npy
```

Optional:

```bash
--kmeans_clusters 7
```

This forces the number of KMeans clusters. If omitted, NS4GC infers it from `data.y`.

## DGCluster

```bash
python3 main.py \
  --model dgcluster \
  --data_pt datasets/custom/scatt_graph.pt \
  --output_dir outputs \
  --epochs 300 \
  --device cpu
```

Output:

```text
outputs/DGCluster_<YYYYMMDD_HHMM>_labels.npy
```

DGCluster also keeps its original metrics/results file under:

```text
results/
```

## EGC-VGAE

```bash
python3 main.py \
  --model egc_vgae \
  --data_pt datasets/custom/scatt_graph.pt \
  --output_dir outputs \
  --epochs 300 \
  --device cpu
```

Output:

```text
outputs/EGC_VGAE_<YYYYMMDD_HHMM>_labels.npy
```

## DSAGC

```bash
python3 main.py \
  --model dsagc \
  --data_pt datasets/custom/scatt_graph.pt \
  --output_dir outputs \
  --epochs 300 \
  --device cpu
```

Output:

```text
outputs/DSAGC_<YYYYMMDD_HHMM>_labels.npy
```

## Common Options

```text
--data_pt           Path to the data.pt file
--output_dir        Directory where labels are saved
--epochs            Number of training epochs
--device            cpu or cuda
--kmeans_clusters   Optional number of clusters for models that use KMeans
```

## Adapted Models So Far

```text
CoANE
CoANE top-k
DGC-EFR
SOLI
SANECE
UCODE
NS4GC
DGCluster
EGC-VGAE
DSAGC
```
