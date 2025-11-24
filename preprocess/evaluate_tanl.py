import json, re, os, argparse
from collections import defaultdict
from typing import List, Tuple, Dict

from datasets import load_dataset
from tqdm import tqdm

# --- token join & normalization must match what you used when generating TANL inputs ---
_PUNCT = re.compile(r"\s+([,.;:%)\]])")
_OPENP = re.compile(r"([\[(])\s+")

def detok(tokens: List[str]) -> str:
    s = " ".join(tokens)
    s = _PUNCT.sub(r"\1", s)
    s = _OPENP.sub(r"\1", s)
    return s

def norm(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("’","'").replace("“",'"').replace("”",'"')
    s = re.sub(r"\s+", " ", s)
    return s

# bracket pattern: [ mention | type | ... ]  (we take the *mention* before the first '|')
BRACKET_SPAN_RE = re.compile(r"\[\s*(?P<mention>.+?)\s*\|\s*[^|\]]+?(?:\s*\|[^\]]+)?\]")

def extract_pred_mentions(tanl_output: str) -> List[str]:
    out = []
    if not tanl_output:
        return out
    for m in BRACKET_SPAN_RE.finditer(tanl_output):
        raw = m.group("mention")
        out.append(raw.split("|")[0].strip())
    return out

def find_token_spans_for_string(par_tokens: List[str], mention: str, max_window: int = 20) -> List[Tuple[int,int]]:
    """
    Map a predicted mention string to token spans in the paragraph by sliding window
    and comparing detok-normalized strings. Returns list of (start,end) [end exclusive].
    """
    tgt = norm(mention)
    N = len(par_tokens)
    hits = []
    for i in range(N):
        acc = []
        for j in range(i+1, min(N, i+max_window)+1):
            acc.append(par_tokens[j-1])
            cand = norm(detok(acc))
            if cand == tgt:
                hits.append((i, j))
                break
            if len(cand) > len(tgt) + 10:
                break
    return hits

def fuzzy_hit(gold_span: Tuple[int,int], pred_span: Tuple[int,int], delta_pos=2, delta_len=2) -> bool:
    gs, ge = gold_span
    ps, pe = pred_span
    len_g = ge - gs
    len_p = pe - ps
    return abs(ps - gs) <= delta_pos and abs(len_p - len_g) <= delta_len

def load_scico_gold(split: str):
    ds = load_dataset("allenai/scico")[split]
    gold = {}           # (topic_id, para_idx) -> dict(tokens, doc_id, gold_spans, gold_strings)
    topic_order = []    # dataset order of topic ids
    for i in range(len(ds)):
        row = ds[i]
        tid = int(row["id"])
        topic_order.append(tid)
        for pid, toks in enumerate(row["tokens"]):
            spans, strings = [], []
            for (ppid, s, e, _cid) in row["mentions"]:
                if ppid == pid:
                    spans.append((int(s), int(e+1)))
                    strings.append(detok(row["tokens"][pid][s:e+1]))
            gold[(tid, pid)] = {
                "tokens": row["tokens"][pid],
                "doc_id": int(row["doc_ids"][pid]),
                "gold_spans": spans,
                "gold_strings": strings,
            }
    return ds, gold, topic_order

def load_predictions(pred_path: str):
    assert os.path.exists(pred_path), f"Missing predictions JSONL at {pred_path}"
    pred = {}  # (tid, pid) -> record dict
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = (int(rec["topic_id"]), int(rec["para_idx"]))
            pred[key] = rec
    return pred

def eval_topic(tid: int, gold, pred, delta_pos: int, delta_len: int, log_fh=None):
    # find paragraph count for topic id
    pids = sorted([pid for (t,pid) in gold.keys() if t == tid])
    topic_gold = 0
    topic_hits = 0
    for pid in pids:
        g = gold.get((tid, pid))
        p = pred.get((tid, pid))
        if g is None or p is None:
            # skip paragraphs without prediction or gold
            continue

        tokens = g["tokens"]
        gold_spans = g["gold_spans"]
        gold_strings = g["gold_strings"]

        # predicted raw strings and their spans
        pred_strings = extract_pred_mentions(p.get("tanl_output",""))
        pred_spans = []
        first_span_for_string = []

        seen = set()
        for s in pred_strings:
            spans = find_token_spans_for_string(tokens, s, max_window=20)
            pred_spans.extend(spans)
            if spans:
                if spans[0] not in seen:
                    first_span_for_string.append((s, spans[0]))
                    seen.add(spans[0])

        # greedy match: each gold can match at most one predicted span
        hits = 0
        misses = []
        matched_pred = set()
        for gidx, gs in enumerate(gold_spans):
            found = False
            for ps in pred_spans:
                if ps in matched_pred:
                    continue
                if fuzzy_hit(gs, ps, delta_pos=delta_pos, delta_len=delta_len):
                    hits += 1
                    matched_pred.add(ps)
                    found = True
                    break
            if not found:
                gtext = gold_strings[gidx] if gidx < len(gold_strings) else ""
                misses.append({"gold_idx": gidx, "span": gs, "text": gtext})

        topic_gold += len(gold_spans)
        topic_hits += hits

        # verbose log for paragraphs with misses
        if log_fh is not None and misses:
            log_fh.write(json.dumps({
                "topic_id": tid,
                "para_idx": pid,
                "doc_id": g["doc_id"],
                "text": detok(tokens),
                "gold_spans": gold_spans,
                "gold_strings": gold_strings,
                "pred_strings": pred_strings,
                "pred_spans_display": [ [s, list(sp)] for s, sp in first_span_for_string ],
                "hits": hits,
                "misses": misses,
                "delta_pos": delta_pos,
                "delta_len": delta_len,
            }, ensure_ascii=False) + "\n")

    recall = (topic_hits / topic_gold) if topic_gold else 0.0
    return topic_hits, topic_gold, recall

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_path", required=True, help="JSONL from batch TANL extraction")
    ap.add_argument("--split", default="test", choices=["train","validation","test"])
    ap.add_argument("--k", type=int, default=-1, help="evaluate only first K topics in dataset order; -1 = all")
    ap.add_argument("--delta_pos", type=int, default=2, help="tolerance for start-position difference (tokens)")
    ap.add_argument("--delta_len", type=int, default=2, help="tolerance for span-length difference (tokens)")
    ap.add_argument("--log_path", default="scico_eval_misses.jsonl", help="JSONL file for verbose missed cases")
    args = ap.parse_args()

    _, gold, topic_order = load_scico_gold(args.split)
    pred = load_predictions(args.pred_path)

    # choose topic subset
    topic_ids = topic_order if args.k < 0 else topic_order[:args.k]

    # progress & logging
    total_gold = total_hits = 0
    macro_parts = []
    with open(args.log_path, "w", encoding="utf-8") as log_fh:
        for idx, tid in enumerate(tqdm(topic_ids, desc="Evaluating topics")):
            th, tg, rec = eval_topic(
                tid, gold, pred, args.delta_pos, args.delta_len, log_fh=log_fh
            )
            total_hits += th
            total_gold += tg
            if tg > 0:
                macro_parts.append(th / tg)

    micro_recall = (total_hits / total_gold) if total_gold else 0.0
    macro_recall = (sum(macro_parts) / len(macro_parts)) if macro_parts else 0.0

    print("=" * 80)
    print(f"SciCo split: {args.split}")
    print(f"Topics evaluated: {len(topic_ids)}  |  Total gold mentions: {total_gold}")
    print(f"[MICRO] recall: {micro_recall:.4f}  ({total_hits}/{total_gold})")
    print(f"[MACRO] recall: {macro_recall:.4f}")
    print(f"Missed cases log: {args.log_path}")

if __name__ == "__main__":
    main()