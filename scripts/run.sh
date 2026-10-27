#!/usr/bin/env bash

set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

config_target=${1:-configs/densgnn_support_set_test}
model=${2:-densgnn}

if [[ -d "$config_target" ]]; then
    shopt -s nullglob
    configs=("$config_target"/*.yaml "$config_target"/*.yml)
    shopt -u nullglob
elif [[ -f "$config_target" ]]; then
    configs=("$config_target")
else
    echo "Configuração ou pasta não encontrada: $config_target" >&2
    exit 2
fi

if [[ ${#configs[@]} -eq 0 ]]; then
    echo "Nenhum arquivo .yaml ou .yml encontrado em: $config_target" >&2
    exit 2
fi

entries=()

for config in "${configs[@]}"; do
    nome=$(basename "$config" .yaml)
    nome=${nome%.yml}
    nome=${nome#densgnn_}
    dataset=$(awk '$1 == "dataset:" {print $2; exit}' "$config")
    if [[ -z "$dataset" ]]; then
        echo "experiment.dataset não encontrado em: $config" >&2
        exit 2
    fi

    echo "Executando: $config"
    python scripts/main.py --config "$config" --model "$model" || exit 1

    if [[ "$model" == "densgnn" ]]; then
        metrics=$(find "results/$dataset/runs" \
            -path "*/densgnn/metrics.json" \
            -printf '%T@ %p\n' |
            sort -nr |
            head -n 1 |
            cut -d' ' -f2-)
        if [[ -z "$metrics" ]]; then
            echo "metrics.json do DensGNN não encontrado em results/$dataset/runs" >&2
            exit 1
        fi

        entries+=("    {\"nome\": \"$nome\", \"caminho\": \"$metrics\"},")
    fi
done

if [[ ${#entries[@]} -gt 0 ]]; then
    echo
    echo "experimentos = ["
    printf '%s\n' "${entries[@]}"
    echo "]"
fi
