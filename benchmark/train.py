import math
import random
import os
import click
import numpy as np
import pandas as pd
import torch

from model import AA_TO_IDX, FusionModel, build_mut_tensors, load_tensor_maybe_dict, predict_fitness, spearman_corr, spearman_loss

DEVICE = 0

DATA_ROOT = "../data/benchmark/data"
OUTPUT_ROOT = "../data/benchmark/raw_prediction_metaai"
CHECKPOINT_ROOT = "../data/benchmark/checkpoints_metaai"

SEED = 0

TOTAL_EPOCHS = 100
INITIAL_LR = 1e-6
MAX_LR = 4e-5
MIN_LR = 1e-6
WARMUP_FRAC = 0.05
WEIGHT_DECAY = 2e-4
GRAD_CLIP_MAX_NORM = 1.0

SPEARMAN_REG_STRENGTH = 0.2
SPEARMAN_REG = "l2"


def to_gpu(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.to(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)

    if isinstance(obj, list):
        return [to_gpu(i, device=device) for i in obj]

    if isinstance(obj, tuple):
        return tuple(to_gpu(i, device=device) for i in obj)

    if isinstance(obj, dict):
        return {k: to_gpu(v, device=device) for k, v in obj.items()}

    return obj


def cosine_lr(epoch, total_epochs, warmup_frac, initial_lr, max_lr, min_lr):
    warmup_epochs = int(total_epochs * warmup_frac)

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        alpha = (epoch - 1) / max(1, warmup_epochs - 1)
        return initial_lr + alpha * (max_lr - initial_lr)

    if warmup_epochs > 0:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    else:
        progress = (epoch - 1) / max(1, total_epochs - 1)

    return min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


@click.command()
@click.option("--protein", required=True, help="Benchmark protein folder name under ../data/benchmark/data.")
@click.option("--output-csv", required=True, type=click.Path(dir_okay=False), help="Output CSV path for fold_id == 1 predictions.")
def main(protein, output_csv):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu")

    data_dir = f"{DATA_ROOT}/{protein}"

    checkpoint_dir = CHECKPOINT_ROOT
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/{protein}.pt"

    output_dir = "/".join(output_csv.split("/")[:-1])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_csv = pd.read_csv(f"{data_dir}/data.csv").copy()
    all_csv["mutation_name"] = all_csv["mutation_name"].astype(str).str.strip()

    train_index = all_csv.index[all_csv["fold_id"] == 0].to_numpy(dtype=np.int64)
    test_index = all_csv.index[all_csv["fold_id"] == 1].to_numpy(dtype=np.int64)

    mut_info_list = all_csv["mutation_name"].astype(str).tolist()
    pos_all, aa_all, mask_all = build_mut_tensors(mut_info_list, device=device)

    esm_raw = load_tensor_maybe_dict(f"{data_dir}/esmc_600m_embedding.pt")
    spired_raw = load_tensor_maybe_dict(f"{data_dir}/spired_fitness_embedding.pt")

    with open(f"{data_dir}/wt.fasta", "r", encoding="utf-8") as f:
        wt_seq = "".join(line.strip() for line in f if line.strip() and not line.startswith(">")).upper()

    wt_idx = to_gpu(torch.tensor([AA_TO_IDX[c] for c in wt_seq], dtype=torch.long), device)

    esm_emb = to_gpu(esm_raw.to(torch.float32).unsqueeze(0), device)
    spired_emb = to_gpu(spired_raw.to(torch.float32).unsqueeze(0), device)

    y = torch.tensor(all_csv["label"].values, dtype=torch.float32, device=device)

    train_idx = torch.tensor(train_index, dtype=torch.long, device=device)
    test_idx = torch.tensor(test_index, dtype=torch.long, device=device)

    model = FusionModel(int(esm_raw.shape[1]), r=20, spired_dim=32, esm_out=64, hidden_dim1=384, hidden_dim2=160, dropout=0.03).to(device)

    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=float(INITIAL_LR), weight_decay=float(WEIGHT_DECAY))

    for epoch in range(1, int(TOTAL_EPOCHS) + 1):
        lr = cosine_lr(epoch=epoch, total_epochs=int(TOTAL_EPOCHS), warmup_frac=float(WARMUP_FRAC), initial_lr=float(INITIAL_LR), max_lr=float(MAX_LR), min_lr=float(MIN_LR))

        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        optimizer.zero_grad()

        single_mut, high_delta = model(esm_emb, spired_emb, wt_idx)

        y_hat = predict_fitness(single_mut, high_delta, pos_all[train_idx], aa_all[train_idx], mask_all[train_idx])

        label = y[train_idx]
        y_hat_norm = (y_hat - y_hat.mean()) / (y_hat.std() + 1e-8)

        loss = spearman_loss(y_hat_norm.unsqueeze(0), label.unsqueeze(0), regularization_strength=float(SPEARMAN_REG_STRENGTH), regularization=SPEARMAN_REG)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), norm_type=2, max_norm=float(GRAD_CLIP_MAX_NORM), error_if_nonfinite=True)

        optimizer.step()

        print(f"[{protein}] " f"epoch={epoch:03d}/{TOTAL_EPOCHS} " f"loss={float(loss.item()):.6f} " f"lr={lr:.8g}")

    torch.save(model, checkpoint_path)

    model.eval()

    with torch.no_grad():
        single_mut, high_delta = model(esm_emb, spired_emb, wt_idx)

        pred = predict_fitness(single_mut, high_delta, pos_all[test_idx], aa_all[test_idx], mask_all[test_idx])

        true = y[test_idx]

        corr = spearman_corr(pred.detach().float().cpu(), true.detach().float().cpu())

    pd.DataFrame({"mutation_name": all_csv.iloc[test_index]["mutation_name"].tolist(), "label": true.detach().float().cpu().numpy(), "pred": pred.detach().float().cpu().numpy()}).to_csv(output_csv, index=False)

    print(f"[{protein}] test_spearman={corr:.2f}")
    print(f"[{protein}] saved_prediction={output_csv}")
    print(f"[{protein}] saved_checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
