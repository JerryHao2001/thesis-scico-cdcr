# scripts/sweep_thresholds_inprocess.py
import os, json, argparse, importlib.util
from collections import defaultdict
import numpy as np
from datasets import load_dataset
from sklearn.cluster import AgglomerativeClustering

# ---------- utils ----------
def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def dynamic_import(path, module_name="eval_sigcoref"):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_pair_scores(path):
    """Read JSONL with: {topic_id, n, edges:[{i,j,logit}]} per line"""
    by_tid = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            by_tid[int(obj["topic_id"])] = obj
    return by_tid

def build_gold_topics(split, gold_end_inclusive=False):
    """Return list[dict] in the structure expected by evaluate_signature_coref.get_coref_scores"""
    ds = load_dataset("allenai/scico")[split]
    gold = []
    for r in ds:
        rec = {
            "id": int(r["id"]),
            "tokens": r["tokens"],
            "doc_ids": r.get("doc_ids", []),
            "relations": r.get("relations", []),
            "mentions": []
        }
        for pid, s, e, cid in r["mentions"]:
            # spans in SciCO are inclusive; keep as-is to match gold convention
            rec["mentions"].append([int(pid), int(s), int(e), int(cid)])
        gold.append(rec)
    return gold

def cc_from_prob(P, thr):
    """Connected Components on thresholded probability graph."""
    n = P.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    adj = (P >= thr).astype(np.bool_)
    np.fill_diagonal(adj, False)
    labels = -np.ones(n, dtype=int)
    seen = np.zeros(n, dtype=bool)
    c = 0
    for i in range(n):
        if seen[i]: continue
        stack = [i]
        seen[i] = True
        labels[i] = c
        while stack:
            u = stack.pop()
            for v in np.where(adj[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    labels[v] = c
                    stack.append(v)
        c += 1
    return labels

def cluster_topic(P, method, linkage, distance_threshold):
    """P: [n,n] probs. Return labels [n]."""
    n = P.shape[0]
    if n <= 1:
        return np.arange(n, dtype=int)
    if method == "cc":
        thr = 1.0 - distance_threshold  # convert distance cut to prob cut
        return cc_from_prob(P, thr)
    # agglomerative on D = 1 - P
    D = 1.0 - P
    np.fill_diagonal(D, 0.0)
    clus = AgglomerativeClustering(
        metric="precomputed", linkage=linkage,
        distance_threshold=distance_threshold, n_clusters=None
    )
    return clus.fit_predict(D)

def make_system_from_labels(split, labels_by_tid):
    """Construct 'system' list mirroring gold structure but with predicted cluster ids."""
    ds = load_dataset("allenai/scico")[split]
    by_id = {int(r["id"]): r for r in ds}
    system = []
    for tid, row in by_id.items():
        if tid not in labels_by_tid:
            continue
        labels = labels_by_tid[tid]
        mentions = row["mentions"]
        assert len(labels) == len(mentions), f"Topic {tid}: labels({len(labels)}) != mentions({len(mentions)})"
        sys_mentions = []
        for i, (pid, s, e, _gold) in enumerate(mentions):
            sys_mentions.append([int(pid), int(s), int(e), int(labels[i])])
        system.append({
            "id": int(tid),
            "tokens": row["tokens"],
            "doc_ids": row.get("doc_ids", []),
            "relations": [],
            "mentions": sys_mentions
        })
    return system

# ---------- main sweep ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_path", required=True, help="pair_scores_*.jsonl from dump step")
    ap.add_argument("--split", default="validation", choices=["train","validation","test"])
    ap.add_argument("--eval_module_path", required=True, help="Path to evaluate_signature_coref.py")
    ap.add_argument("--temperature_json", default="", help="JSON produced by fit_temperature_from_pairs.py")
    ap.add_argument("--method", default="agglomerative", choices=["agglomerative","cc"])
    ap.add_argument("--linkage", default="average", choices=["average","complete"])
    ap.add_argument("--t_min", type=float, default=0.10)
    ap.add_argument("--t_max", type=float, default=0.90)
    ap.add_argument("--t_step", type=float, default=0.02)
    args = ap.parse_args()

    # then, after parsing args:
    if args.temperature_json:
        with open(args.temperature_json, "r", encoding="utf-8") as f:
            Tobj = json.load(f)
        args.temperature = float(Tobj["temperature"])
        print(f"[sweep] Loaded temperature T={args.temperature:.4f} from {args.temperature_json}")


    # Import evaluator in-process
    eval_mod = dynamic_import(args.eval_module_path)
    get_coref_scores = eval_mod.get_coref_scores  # returns dict with 'conll' = sum of F1s

    # Build gold topics (Python objects)
    gold = build_gold_topics(args.split)

    # Map topic_id -> probability matrix
    scores_by_tid = load_pair_scores(args.scores_path)
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
    t = args.t_min
    best = {"t": None, "score": -1.0, "system": None}
    results = []
    while t <= args.t_max + 1e-9:
        labels_by_tid = {tid: cluster_topic(P, args.method, args.linkage, t)
                         for tid, P in probs_by_tid.items()}
        system = make_system_from_labels(args.split, labels_by_tid)
        scores = get_coref_scores(gold, system)
        conll = (scores["conll"] / 3.0) * 100.0
        results.append((t, conll))
        if conll > best["score"]:
            best.update({"t": t, "score": conll, "system": system})
        t += args.t_step

    # Report
    print("Threshold sweep (t, CoNLL):")
    for t, s in results:
        print(f"{t:.2f}\t{s:.2f}")
    print(f"\nBest: t={best['t']:.2f}, CoNLL={best['score']:.2f}")

    # Optional: write the best system JSONL to disk for record keeping
    out_path = f"best_system_{args.method}_{args.linkage}_t{best['t']:.2f}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in best["system"]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved best system to {out_path}")

if __name__ == "__main__":
    main()
