# scripts/sweep_thresholds.py
import os, json, argparse, subprocess, tempfile
import numpy as np
from collections import defaultdict
from datasets import load_dataset
from sklearn.cluster import AgglomerativeClustering

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def load_pair_scores(path):
    """JSONL with {topic_id, n, edges:[{i,j,logit}]}"""
    by_tid = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            by_tid[int(obj["topic_id"])] = obj
    return by_tid

def export_gold_jsonl(split, out_path, gold_end_inclusive=False):
    ds = load_dataset("allenai/scico")[split]
    with open(out_path, "w", encoding="utf-8") as out:
        for r in ds:
            rec = {
                "id": int(r["id"]),
                "tokens": r["tokens"],
                "doc_ids": r.get("doc_ids", []),
                "relations": r.get("relations", []),
                "mentions": []
            }
            for pid, s, e, cid in r["mentions"]:
                s, e = int(s), int(e)
                if gold_end_inclusive:
                    # the dataset spans are inclusive already; keep as-is to mirror gold file convention
                    pass
                rec["mentions"].append([int(pid), s, e, int(cid)])
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

def connected_components_from_prob(P, thr):
    """P: [n,n] probs; edge if P>=thr. Return cluster labels [n]."""
    n = P.shape[0]
    adj = (P >= thr).astype(np.bool_)
    # zero diagonal
    np.fill_diagonal(adj, False)
    # BFS
    seen = np.zeros(n, dtype=bool)
    labels = -np.ones(n, dtype=int)
    c = 0
    for i in range(n):
        if seen[i]: continue
        # start new component
        stack = [i]
        seen[i] = True
        labels[i] = c
        while stack:
            u = stack.pop()
            nbrs = np.where(adj[u])[0]
            for v in nbrs:
                if not seen[v]:
                    seen[v] = True
                    labels[v] = c
                    stack.append(v)
        c += 1
    return labels

def cluster_topic(P, method, linkage, distance_threshold):
    """Return labels for a topic given P (probs)."""
    n = P.shape[0]
    if n <= 1:
        return np.arange(n, dtype=int)
    if method == "cc":
        thr = 1.0 - distance_threshold  # P threshold
        return connected_components_from_prob(P, thr)
    # agglomerative
    D = 1.0 - P
    np.fill_diagonal(D, 0.0)
    clus = AgglomerativeClustering(
        metric="precomputed", linkage=linkage,
        distance_threshold=distance_threshold, n_clusters=None
    )
    return clus.fit_predict(D)

def write_system_jsonl(split, scores_by_tid, labels_by_tid, out_path, gold_end_inclusive=False):
    """Rewrite gold structure but replace cluster ids with predicted labels (matching mention order)."""
    ds = load_dataset("allenai/scico")[split]
    by_id = {int(r["id"]): r for r in ds}
    with open(out_path, "w", encoding="utf-8") as out:
        for tid, r in by_id.items():
            if tid not in labels_by_tid:
                continue
            labels = labels_by_tid[tid]
            mentions = r["mentions"]
            assert len(labels) == len(mentions), f"Topic {tid}: labels({len(labels)}) != mentions({len(mentions)})"
            sys_mentions = []
            for i, (pid, s, e, _gold) in enumerate(mentions):
                s, e = int(s), int(e)
                # keep spans as-is (they’re gold); only swap cluster id
                sys_mentions.append([int(pid), s, e, int(labels[i])])
            rec = {
                "id": int(tid),
                "tokens": r["tokens"],
                "doc_ids": r.get("doc_ids", []),
                "relations": [],        # empty for coref-only eval
                "mentions": sys_mentions
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

def run_evaluator(eval_script, gold_jsonl, sys_jsonl):
    """Call your evaluate.py and parse its stdout for 'CoNLL score' (last token)."""
    proc = subprocess.run(
        ["python", eval_script, gold_jsonl, sys_jsonl],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    out = proc.stdout.strip().splitlines()
    # Assumes evaluate.py prints a line like: "CoNLL score: 73.21"
    # Robust parse: find last number in output
    import re
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", out[-1] if out else "")
    if not nums:
        # fallback: scan all lines for the phrase
        for line in reversed(out):
            if "Conll" in line.lower() or "conll" in line:
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if nums: break
    if not nums:
        print("\n".join(out))
        raise RuntimeError("Could not parse CoNLL score from evaluator output.")
    return float(nums[-1]), out  # score, raw output

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_path", required=True, help="pair_scores_dev.jsonl from dump step")
    ap.add_argument("--split", default="validation", choices=["train","validation","test"])
    ap.add_argument("--eval_script", required=True, help="path to evaluate.py")
    ap.add_argument("--gold_jsonl", default="", help="optional prebuilt gold jsonl; if empty, will export one")
    ap.add_argument("--temperature", type=float, default=1.0, help="T for p = sigmoid(logit/T)")
    ap.add_argument("--method", default="agglomerative", choices=["agglomerative","cc"])
    ap.add_argument("--linkage", default="average", choices=["average","complete"], help="for agglomerative")
    ap.add_argument("--t_min", type=float, default=0.05)
    ap.add_argument("--t_max", type=float, default=0.95)
    ap.add_argument("--t_step", type=float, default=0.02)
    ap.add_argument("--gold_end_inclusive", action="store_true")
    ap.add_argument("--work_dir", default="sweep_outputs")
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    scores_by_tid = load_pair_scores(args.scores_path)

    # Export gold if needed
    gold_path = args.gold_jsonl
    if not gold_path:
        gold_path = os.path.join(args.work_dir, f"gold_{args.split}.jsonl")
        export_gold_jsonl(args.split, gold_path, gold_end_inclusive=args.gold_end_inclusive)

    # Build per-topic probability matrices from logits
    probs_by_tid = {}
    for tid, obj in scores_by_tid.items():
        n = int(obj["n"])
        P = np.zeros((n, n), dtype=np.float32)
        for e in obj["edges"]:
            i, j, z = int(e["i"]), int(e["j"]), float(e["logit"])
            p = sigmoid(z / args.temperature)
            P[i, j] = p; P[j, i] = p
        np.fill_diagonal(P, 1.0)
        probs_by_tid[tid] = P

    # Sweep thresholds
    best = {"score": -1.0, "t": None, "sys_path": None}
    t = args.t_min
    results = []
    while t <= args.t_max + 1e-9:
        labels_by_tid = {}
        for tid, P in probs_by_tid.items():
            labels_by_tid[tid] = cluster_topic(P, args.method, args.linkage, t)

        sys_path = os.path.join(args.work_dir, f"system_{args.method}_{args.linkage}_t{t:.2f}.jsonl")
        write_system_jsonl(args.split, scores_by_tid, labels_by_tid, sys_path, gold_end_inclusive=args.gold_end_inclusive)

        score, _out = run_evaluator(args.eval_script, gold_path, sys_path)
        results.append((t, score))
        if score > best["score"]:
            best.update({"score": score, "t": t, "sys_path": sys_path})
        t += args.t_step

    # Print table and winner
    print("Threshold sweep (t, CoNLL):")
    for t, s in results:
        print(f"{t:.2f}\t{s:.2f}")
    print(f"\nBest: t={best['t']:.2f}, CoNLL={best['score']:.2f}")
    print(f"Best system file: {best['sys_path']}")

if __name__ == "__main__":
    main()
