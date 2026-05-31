import click
import torch
import pandas as pd

from model import AA_TO_IDX, load_tensor_maybe_dict, build_mut_tensors, predict_fitness

FEATURE_NAME = "esmc_600m"


@click.command()
@click.option("--mut_counts", required=True, type=int)
def main(mut_counts: int):
    out_csv = f"pred_sorted_mut_counts_{mut_counts}.csv"
    ckpt = "4CL.pt"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open("../../data/example_protein/4CL/features/wt/result.fasta", "r") as f:
        wt_seq = "".join(line.strip() for line in f if line.strip() and not line.startswith(">")).upper()
    wt_idx = torch.tensor([AA_TO_IDX[c] for c in wt_seq], dtype=torch.long, device=device)

    esm_path = "../../data/example_protein/4CL/features/wt/esmc_600m_embedding.pt" if FEATURE_NAME == "esmc_600m" else "../../data/example_protein/4CL/features/wt/esmc_300m_embedding.pt"
    esm = load_tensor_maybe_dict(esm_path).to(torch.float32).unsqueeze(0).to(device)
    spired = load_tensor_maybe_dict("../../data/example_protein/4CL/features/wt/spired_fitness_embedding.pt").to(torch.float32).unsqueeze(0).to(device)

    df = pd.read_csv(f"../../data/example_protein/4CL/candidates/sorted_mut_counts_{mut_counts}.csv")
    mut = df["mut_name"].astype(str).tolist()
    pos, aa, mask = build_mut_tensors(mut, device)

    model = torch.load(ckpt, map_location="cpu", weights_only=False).to(device).eval()
    with torch.no_grad():
        single_mut, high_delta = model(esm, spired, wt_idx)
        pred = predict_fitness(single_mut, high_delta, pos, aa, mask).detach().float().cpu().numpy()

    out_df = pd.DataFrame({"mut_name": mut, "pred": pred}).sort_values("pred", ascending=False)
    out_df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
