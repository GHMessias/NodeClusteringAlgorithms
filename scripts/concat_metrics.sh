#!/bin/bash

# Verifica se o diretório raiz foi fornecido como argumento
ROOT_DIR="${1:-.}"
OUTPUT_FILE="metrics_combinados.csv"

if [ ! -d "$ROOT_DIR" ]; then
  echo "Erro: O diretório '$ROOT_DIR' não existe."
  exit 1
fi

# Remove o arquivo de saída anterior se ele já existir para evitar duplicar dados
rm -f "$OUTPUT_FILE"

# Flag para controlar o cabeçalho
HEADER_SAVED=false

# Busca recursivamente todos os arquivos chamados 'metrics.csv'
find "$ROOT_DIR" -type f -name "metrics.csv" | while read -r file; do
  if [ "$HEADER_SAVED" = false ]; then
    # Na primeira vez, copia o arquivo inteiro (incluindo o cabeçalho)
    cat "$file" >> "$OUTPUT_FILE"
    HEADER_SAVED=true
  else
    # Nas próximas vezes, ignora a primeira linha e concatena o restante
    tail -n +2 "$file" >> "$OUTPUT_FILE"
  fi
done

echo "Processo concluído! Arquivos concatenados em: $OUTPUT_FILE"