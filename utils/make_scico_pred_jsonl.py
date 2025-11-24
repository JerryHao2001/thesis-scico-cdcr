# make_system_coref_jsonl.py
import json, argparse
from datasets import load_dataset

def load_pred_clusters(path):
    pred = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pred[int(rec["topic_id"])] = rec["clusters"]
    return pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train","validation","test"])
    ap.add_argument("--predicted_clusters", required=True)
    ap.add_argument("--out_path", default="system_pred.jsonl")
    args = ap.parse_args()

    ds = load_dataset("allenai/scico")[args.split]
    pred = load_pred_clusters(args.predicted_clusters)

    # Build topic-id -> row map
    by_id = {int(row["id"]): row for row in ds}

    n_written, n_skipped = 0, 0
    with open(args.out_path, "w", encoding="utf-8") as out:
        for tid, row in by_id.items():
            if tid not in pred:
                n_skipped += 1
                continue
            gold_mentions = row["mentions"]  # [pid, s, e, gold_cluster]
            clusters = pred[tid]
            if len(clusters) != len(gold_mentions):
                raise ValueError(f"Topic {tid}: #pred={len(clusters)} != #gold_mentions={len(gold_mentions)}")
            sys_mentions = []
            for i, (pid, s, e, _gold_c) in enumerate(gold_mentions):
                sys_mentions.append([int(pid), int(s), int(e), int(clusters[i])])

            rec = {
                "id": int(tid),
                "tokens": row["tokens"],
                "doc_ids": row.get("doc_ids", []),
                "mentions": sys_mentions,
                "relations": []  # empty; we’re only scoring coref
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} topics to {args.out_path} (skipped {n_skipped} missing preds)")

if __name__ == "__main__":
    main()