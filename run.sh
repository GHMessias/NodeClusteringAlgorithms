#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${1:-$ROOT_DIR/datasets/custom}"
OUTPUT_BASE_DIR="${2:-$ROOT_DIR/outputs}"
EPOCHS="${EPOCHS:-300}"
DEVICE="${DEVICE:-cuda}"
DRY_RUN="${DRY_RUN:-0}"
FAIL_FAST="${FAIL_FAST:-0}"

if [[ -x "$ROOT_DIR/venv/bin/python3" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python3}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory not found: $INPUT_DIR" >&2
  exit 1
fi

mapfile -t PT_FILES < <(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.pt' | sort)

if [[ "${#PT_FILES[@]}" -eq 0 ]]; then
  echo "No .pt files found in: $INPUT_DIR" >&2
  exit 1
fi

FAILED_DATASETS=()

for data_pt in "${PT_FILES[@]}"; do
  data_name="$(basename "$data_pt" .pt)"
  output_dir="$OUTPUT_BASE_DIR/$data_name"

  mkdir -p "$output_dir"

  cmd=(
    "$PYTHON_BIN"
    "$ROOT_DIR/main.py"
    --model all
    --data_pt "$data_pt"
    --output_dir "$output_dir"
    --epochs "$EPOCHS"
    --device "$DEVICE"
    --continue_on_error
  )

  echo
  echo "=== Running dataset: $data_name ==="
  echo "Data: $data_pt"
  echo "Output: $output_dir"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  if ! "${cmd[@]}"; then
    FAILED_DATASETS+=("$data_name")
    echo "Dataset failed: $data_name" >&2
    if [[ "$FAIL_FAST" == "1" ]]; then
      exit 1
    fi
  fi
done

echo
if [[ "${#FAILED_DATASETS[@]}" -eq 0 ]]; then
  echo "All datasets finished."
else
  echo "Finished with failures in ${#FAILED_DATASETS[@]} dataset(s):" >&2
  for data_name in "${FAILED_DATASETS[@]}"; do
    echo "  - $data_name" >&2
  done
  exit 1
fi
