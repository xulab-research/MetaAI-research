# MetaAI

MetaAI is a PyTorch implementation of a low-N, structure-informed model for ranking combinatorial protein mutations. In the accompanying study, MetaAI learns from CoEvo-derived anchor variants to reconstruct cellular sequence-metabolite landscapes and prioritize unsampled mutants.

## Layout

```text
benchmark/   training code for the public low-N benchmark
data/        benchmark datasets, precomputed features, raw predictions, example inputs 
examples/    inference scripts and checkpoints for 4CL and CHS
```

## Install

```bash
conda create -n metaai python=3.10 -y
conda activate metaai
pip install torch numpy pandas scipy numba click
```

Mutation strings use comma-separated substitutions, for example `F109R,N396K`. Residue numbers are **0-based indices** matched to the supplied WT FASTA/features.

## Data

```text
data/benchmark/
  data/                      17 public low-N datasets
  checkpoints_metaai/         trained MetaAI checkpoints for the 17 datasets
  raw_prediction_metaai/      MetaAI predictions 
  raw_prediction_supervised/  supervised baseline predictions
  raw_prediction_unsupervised/ zero-shot/unsupervised baseline scores
```

The 17 public benchmark datasets used for low-N evaluation comprise 10 compatible ProteinGym datasets and 7 curated epistasis/combinatorial-mutation datasets.
Each folder in `data/benchmark/data/<protein>/` contains:

```text
data.csv                    # mutation_name,label,fold_id
wt.fasta                    # wild-type sequence
esmc_600m_embedding.pt       # precomputed sequence representation
spired_fitness_embedding.pt  # precomputed structure-related representation
```

`fold_id = 0` is used for training and `fold_id = 1` for testing. The raw-prediction folders store per-variant predictions/scores used for benchmark comparison. 

```text
data/example_protein/4CL/
data/example_protein/CHS/
  data/train.csv             # CoEvo anchor variants used for training
  data/test.csv              # blind experimental test variants
  candidates/                # 10,000 candidates for 2-, 3-, and 4-mutant design
  features/wt/               # WT FASTA and precomputed ESM C/SPIRED-Fitness features
```

For the cellular sequence-metabolite examples, 4CL contains 211 training variants and 54 blind-test variants; CHS contains 149 training variants and 60 blind-test variants. 

## Benchmark training

```bash
cd benchmark
python train.py --protein <protein> --output-csv ../data/benchmark/raw_prediction_metaai/<protein>.csv
# or run all benchmark folders
bash 01_run.sh
```

## Example inference

```bash
cd examples/4CL        # or examples/CHS
python inference_muts.py --mut_counts 2   # use 2, 3, or 4
```

