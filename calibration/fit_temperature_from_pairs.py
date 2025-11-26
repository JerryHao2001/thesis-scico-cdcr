import json, argparse
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset

def load_pair_scores(path):
    by_tid = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            by_tid[int(obj["topic_id"])] = obj
    return by_tid

class BinaryTemperatureScaler(nn.Module):
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        # optimize log_T; softplus keeps T > 0
        self.log_T = nn.Parameter(torch.log(torch.tensor(init_temp)))

    @property
    def T(self):
        return torch.nn.functional.softplus(self.log_T) + 1e-6

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T

def fit_temperature(dev_logits, dev_labels, pos_weight=None, max_iter=200, tol=1e-7):
    if not torch.is_tensor(dev_logits):
        dev_logits = torch.tensor(dev_logits, dtype=torch.float32)
    if not torch.is_tensor(dev_labels):
        dev_labels = torch.tensor(dev_labels, dtype=torch.float32)

    scaler = BinaryTemperatureScaler()
    scaler.train()

    criterion = (nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
                 if pos_weight is not None else nn.BCEWithLogitsLoss())
    opt = torch.optim.LBFGS([scaler.log_T], lr=0.1, max_iter=max_iter,
                            tolerance_grad=tol, tolerance_change=tol,
                            line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = criterion(scaler(dev_logits), dev_labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(scaler.T.detach().item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_path", required=True, help="pair_scores_dev.jsonl (from dumper)")
    ap.add_argument("--split", default="validation", choices=["train","validation","test"])
    ap.add_argument("--out_json", default="temperature_dev.json")
    ap.add_argument("--use_pos_weight", action="store_true",
                    help="optional: weight positives like in training")
    args = ap.parse_args()

    # 1) load dumped logits per topic
    by_tid = load_pair_scores(args.scores_path)

    # 2) build pair labels from gold clusters for the same split
    ds = load_dataset("allenai/scico")[args.split]
    logits, labels = [], []
    skipped = 0

    for r in ds:
        tid = int(r["id"])
        if tid not in by_tid:
            skipped += 1
            continue
        obj = by_tid[tid]
        n = int(obj["n"])
        gold_mentions = r["mentions"]               # [ [pid, s, e, cluster_id], ... ]
        if n != len(gold_mentions):
            # Inconsistent topic; skip conservatively
            skipped += 1
            continue
        gold_cids = [int(m[3]) for m in gold_mentions]
        # make a quick set for positive pairs? Not needed; compute on the fly
        for e in obj["edges"]:
            i, j, z = int(e["i"]), int(e["j"]), float(e["logit"])
            y = 1.0 if gold_cids[i] == gold_cids[j] else 0.0
            logits.append(z); labels.append(y)

    logits = np.array(logits, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    if logits.size == 0:
        raise RuntimeError("No pairs found to fit temperature. Check scores_path/split alignment.")

    # 3) fit temperature on dev pairs
    pos_rate = float(labels.mean()) if labels.size else 0.0
    pos_w = None
    if args.use_pos_weight and 0 < pos_rate < 1:
        pos_w = (1 - pos_rate) / pos_rate  # optional; usually leave off for TS

    T = fit_temperature(logits, labels, pos_weight=pos_w)

    # 4) save to JSON (with a couple of stats)
    out = {
        "temperature": T,
        "split": args.split,
        "num_pairs": int(labels.size),
        "pos_rate": pos_rate,
        "scores_path": args.scores_path
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Fitted temperature T={T:.4f} on {labels.size} pairs (pos_rate={pos_rate:.3f}).")
    print(f"Wrote {args.out_json}")

if __name__ == "__main__":
    main()
