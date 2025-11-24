# eval/run_coref_eval.py
import json
import argparse
from collections import defaultdict
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support

def labels_from_clusters(cluster_ids):
    # returns pairwise labels upper-triangle (i<j) for metric sanity
    pairs = {}
    n = len(cluster_ids)
    for i in range(n):
        for j in range(i+1, n):
            pairs[(i,j)] = 1 if cluster_ids[i] == cluster_ids[j] else 0
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--predicted_clusters", required=True, help="JSONL from predict_signature_coref.py")
    args = ap.parse_args()

    ds = load_dataset("allenai/scico")[args.split]
    gold_pairs = {}

    # build gold pairwise labels by topic from dataset clusters
    for i in range(len(ds)):
        topic = ds[i]
        tid = int(topic["id"])
        mentions = topic["mentions"]
        # Map gold cluster id per mention index (in dataset order)
        gold_cluster_ids = [m[3] for m in mentions]
        gold_pairs[tid] = labels_from_clusters(gold_cluster_ids)

    # load predictions
    preds_pairs = {}
    with open(args.predicted_clusters, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            preds_pairs[r["topic_id"]] = labels_from_clusters(r["clusters"])

    # align topics that are present in predictions
    y_true, y_pred = [], []
    for tid, gp in gold_pairs.items():
        pp = preds_pairs.get(tid)
        if pp is None:
            continue
        # intersect pairs
        keys = sorted(list(set(gp.keys()) & set(pp.keys())))
        y_true.extend([gp[k] for k in keys])
        y_pred.extend([pp[k] for k in keys])

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    print({"pairwise_precision": p, "pairwise_recall": r, "pairwise_f1": f1})

if __name__ == "__main__":
    main()
