#!/usr/bin/env bash

set -Eeuo pipefail
mkdir -p log
CUDA_VISIBLE_DEVICES=1 nohup python inference_muts.py --mut_counts 2 > log/mut_counts_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python inference_muts.py --mut_counts 3 > log/mut_counts_3.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python inference_muts.py --mut_counts 4 > log/mut_counts_4.log 2>&1 &
