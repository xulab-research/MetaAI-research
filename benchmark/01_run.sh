#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="../data/benchmark/data"
OUTPUT_ROOT="../data/benchmark/raw_prediction_metaai"

mkdir -p "${OUTPUT_ROOT}"

for protein_dir in "${DATA_ROOT}"/*; do
    protein=$(basename "${protein_dir}")
    output_csv="${OUTPUT_ROOT}/${protein}.csv"
    CUDA_VISIBLE_DEVICES=3 python train.py \
        --protein "${protein}" \
        --output-csv "${output_csv}"
done

echo "All benchmark proteins finished."
