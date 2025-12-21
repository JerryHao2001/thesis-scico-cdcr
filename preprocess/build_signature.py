#!/usr/bin/env python3
"""
Build YAML-style signatures for cross-encoder input from SciCo gold mentions and TANL outputs.

Input:
  - SciCo split (train/validation/test) via HF datasets
  - scico_tanl_extraction.jsonl (from your batch extractor), per paragraph:
        { topic_id, para_idx, doc_id, text, tanl_output, sentences? }

Output (JSONL):
  one line per GOLD mention:
    {
      "topic_id": int,
      "doc_id": int,
      "para_idx": int,
      "sent_idx": int,
      "gold_span": [s, e],          # normalized to right-open indices
      "gold_text": str,
      "cluster_id": int,
      "signature": "<yaml string>"  # top-level key is <m> ... </m>
    }

Signature YAML shape (concise):
  "<m>{gold_text}</m>":
    type: <pred_type or "unknown">
    relations:
      outgoing: [ {predicate, object_text, object_type?}, ... ]
      incoming: [ {predicate, subject_text, subject_type?}, ... ]
    co_mentions: [ {text, type?}, ... ]
    context: "<sentence with <m>...</m>>"
"""

import json, re, os, argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import yaml
from datasets import load_dataset
from tqdm import tqdm

# --------------------------- tokenization helpers ---------------------------

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

# --------------------------- TANL parsing -----------------------------------

# [ mention | TYPE | r1 = obj ; r2 = obj2 ]  ; (the rel section can be separated by ';' or '|')
BRACKET_RE = re.compile(
    r"\[\s*(?P<mention>[^|\]]+?)\s*\|\s*(?P<etype>[^|\]]+?)\s*(?:\|\s*(?P<rels>.+?))?\s*\]"
)

REL_SEP_RE = re.compile(r"\s*\|\s*")

def parse_relations(rels_str: str):
    """
    Parse relations like:
      "used for = hybrid model | evaluated on = CNN/DailyMail"
    allowing spaces in predicate names and arguments.
    """
    rels = []
    rels_str = (rels_str or "").strip()
    if not rels_str:
        return rels

    for chunk in REL_SEP_RE.split(rels_str):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue

        pred, arg = chunk.split("=", 1)  # split once: keep '=' inside arg if it ever occurs
        pred = re.sub(r"\s+", " ", pred).strip()
        arg  = re.sub(r"\s+", " ", arg).strip()
        if pred and arg:
            rels.append((pred, arg))

    return rels

def parse_tanl(tanl_text: str):
    entities = []
    for m in BRACKET_RE.finditer(tanl_text or ""):
        mention = (m.group("mention") or "").strip()
        etype = (m.group("etype") or "").strip()
        rels_str = m.group("rels") or ""

        rels = parse_relations(rels_str)

        entities.append({
            "text": mention,
            "type": etype if etype else None,
            "rels_out": rels,
        })
    return entities

def find_token_spans_for_string(par_tokens: List[str], mention: str, max_window: int = 20) -> List[Tuple[int,int]]:
    """Map a mention string back to paragraph token spans by sliding-window detok equality."""
    tgt = norm(mention)
    N = len(par_tokens)
    hits = []
    for i in range(N):
        acc = []
        for j in range(i+1, min(N, i+max_window)+1):
            acc.append(par_tokens[j-1])
            if norm(detok(acc)) == tgt:
                hits.append((i, j))
                break
            if len(" ".join(acc)) > len(mention) + 15:
                break
    return hits

def fuzzy_hit(gold_span: Tuple[int,int], pred_span: Tuple[int,int], delta_pos=2, delta_len=2) -> bool:
    gs, ge = gold_span
    ps, pe = pred_span
    len_g = ge - gs
    len_p = pe - ps
    return abs(ps - gs) <= delta_pos and abs(len_p - len_g) <= delta_len

# --------------------------- sentence & co-mentions -------------------------

def find_sentence_idx(sent_spans: List[Tuple[int,int]], token_span: Tuple[int,int]) -> int:
    """Choose the sentence index whose span overlaps the gold span the most."""
    if not sent_spans:
        return 0
    gs, ge = token_span
    best_i, best_overlap = 0, -1
    for i, (ss,se) in enumerate(sent_spans):
        inter = max(0, min(ge,se) - max(gs,ss))
        if inter > best_overlap:
            best_overlap = inter
            best_i = i
    return best_i

# --------------------------- YAML signature builder -------------------------

def build_signature_yaml(
    gold_text: str,
    sentence_text: str,
    target_type: Optional[str],
    outgoing: List[Dict],
    incoming: List[Dict],
    co_mentions: List[Dict],
    m_open: str = "<m>",
    m_close: str = "</m>",
) -> str:
    top_key = f'{m_open}{gold_text}{m_close}'
    # sort deterministically & cap sizes here if desired
    outgoing_sorted = sorted(outgoing, key=lambda x: (x["predicate"], x.get("object_text","")))
    incoming_sorted = sorted(incoming, key=lambda x: (x["predicate"], x.get("subject_text","")))
    co_sorted = sorted(co_mentions, key=lambda x: (x.get("text",""), x.get("type","")))

    card = {
        top_key: {
            "type": target_type if target_type else "unknown",
            "relations": {
                "outgoing": outgoing_sorted,
                "incoming": incoming_sorted,
            },
            "co_mentions": co_sorted,
            "context": sentence_text,
        }
    }
    # Keep YAML compact (no anchors)
    return yaml.safe_dump(card, sort_keys=False, allow_unicode=True).strip()

# --------------------------- main ------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_path", required=True, help="JSONL from batch TANL extraction")
    ap.add_argument("--split", default="test", choices=["train","validation","test"])
    ap.add_argument("--out_path", default="scico_signatures.jsonl")
    ap.add_argument("--k", type=int, default=-1, help="Only process first K topics; -1 = all")

    # Fuzzy match thresholds
    ap.add_argument("--delta_pos", type=int, default=3)
    ap.add_argument("--delta_len", type=int, default=3)

    # Signature content controls
    ap.add_argument("--max_rel", type=int, default=6, help="cap number of incoming/outgoing relations")
    ap.add_argument("--max_com", type=int, default=6, help="cap number of co-mentions")
    ap.add_argument("--m_open", default="<m>")
    ap.add_argument("--m_close", default="</m>")
    args = ap.parse_args()

    # Load SciCo
    ds = load_dataset("allenai/scico")[args.split]
    topic_order = [int(ds[i]["id"]) for i in range(len(ds))]
    if args.k > 0:
        topic_order = topic_order[:args.k]

    # Load predictions
    assert os.path.exists(args.pred_path), f"Missing: {args.pred_path}"
    pred_by_para = {}  # (tid, pid) -> record
    with open(args.pred_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pred_by_para[(int(rec["topic_id"]), int(rec["para_idx"]))] = rec

    # Process topics
    out_fh = open(args.out_path, "w", encoding="utf-8")
    for tid in tqdm(topic_order, desc="Building signatures"):
        row = next(r for r in ds if int(r["id"]) == tid)

        tokens_by_para = row["tokens"]
        doc_ids = row["doc_ids"]
        sent_spans_by_para = row.get("sentences", None)  # list of list of spans
        mentions = row["mentions"]  # list of [pid, s, e, cluster_id]

        # normalize gold spans (right-open) and group per paragraph
        gold_by_para = defaultdict(list)
        for (pid, s, e, cid) in mentions:
            s = int(s); e = int(e)
            # e = min(e + 1, len(tokens_by_para[pid]))
            gold_by_para[int(pid)].append((s, e, int(cid)))

        for pid, toks in enumerate(tokens_by_para):
            key = (tid, pid)
            pred_rec = pred_by_para.get(key)
            if pred_rec is None:
                continue

            tanl_text = pred_rec.get("tanl_output", "")
            tanl_entities = parse_tanl(tanl_text)

            # map each TANL entity to its token spans (first match used for indexing & sentence)
            tanl_spans = []
            for ent in tanl_entities:
                spans = find_token_spans_for_string(toks, ent["text"])
                tanl_spans.append(spans[0] if spans else None)  # may be None

            # sentence indexer
            para_sent_spans = []
            if sent_spans_by_para:
                para_sent_spans = [(int(a), int(b)) for (a,b) in sent_spans_by_para[pid]]
            else:
                # fallback single "sentence" = whole paragraph
                para_sent_spans = [(0, len(toks))]

            # For incoming relations, build a quick map: object_text(norm) -> list of (subj_idx, pred)
            obj_to_incoming = defaultdict(list)
            for i, ent in enumerate(tanl_entities):
                for (pred_name, obj_text) in ent["rels_out"]:
                    obj_to_incoming[norm(obj_text)].append((i, pred_name))

            # For each gold mention in this paragraph: build YAML signature
            for (gs, ge, cid) in gold_by_para.get(pid, []):
                ge_ro = int(ge) + 1

                gold_text = detok(toks[gs:ge_ro])

                # find sentence
                sent_idx = find_sentence_idx(para_sent_spans, (gs,ge))
                ss, se = para_sent_spans[sent_idx]
                sent_text = detok(toks[ss:se])

                # try to match gold to one TANL entity (fuzzy rule on token spans)
                match_idx = None
                for i, sp in enumerate(tanl_spans):
                    if sp is None:
                        continue
                    if fuzzy_hit((gs,ge_ro), sp, delta_pos=args.delta_pos, delta_len=args.delta_len):
                        match_idx = i
                        break

                # target type & outgoing (if matched)
                target_type = None
                outgoing = []
                if match_idx is not None:
                    target_type = tanl_entities[match_idx]["type"]
                    for (pred_name, obj_text) in tanl_entities[match_idx]["rels_out"]:
                        outgoing.append({
                            "predicate": pred_name,
                            "object_text": obj_text,
                            # It’s optional to add object_type; we could scan entities to find a type if you want.
                        })

                # incoming: any other TANL entity that points to this target by arg string
                incoming = []
                for (subj_idx, pred_name) in obj_to_incoming.get(norm(gold_text), []):
                    subj_ent = tanl_entities[subj_idx]
                    incoming.append({
                        "predicate": pred_name,
                        "subject_text": subj_ent["text"],
                        "subject_type": subj_ent["type"],
                    })

                # co-mentions: TANL entities whose (first) span falls within the same sentence and is not the target
                co_mentions = []
                for i, sp in enumerate(tanl_spans):
                    if sp is None:
                        continue
                    st, en = sp
                    # inside sentence?
                    if not (st >= ss and en <= se):
                        continue
                    # skip if this is the matched target span (loosely check overlap with gold)
                    if abs(st - gs) <= args.delta_pos and abs((en - st) - (ge_ro - gs)) <= args.delta_len:
                        continue
                    item = {"text": tanl_entities[i]["text"]}
                    if tanl_entities[i]["type"]:
                        item["type"] = tanl_entities[i]["type"]
                    co_mentions.append(item)

                # cap lists deterministically
                outgoing = sorted(outgoing, key=lambda x: (x["predicate"], x.get("object_text","")))[:args.max_rel]
                incoming = sorted(incoming, key=lambda x: (x["predicate"], x.get("subject_text","")))[:args.max_rel]
                co_mentions = sorted(co_mentions, key=lambda x: (x.get("text",""), x.get("type","")))[:args.max_com]

                # highlight target in sentence
                # simple token-level insertion using offsets relative to sentence
                pre = detok(toks[ss:gs])
                mid = gold_text
                post = detok(toks[ge_ro:se])
                highlighted = (pre + (" " if pre and not pre.endswith(" ") else "")
                               + args.m_open + mid + args.m_close
                               + ("" if (not post or post.startswith(" ")) else " ") + post).strip()
                # normalize stray spaces near punctuation
                highlighted = _PUNCT.sub(r"\1", highlighted)
                highlighted = _OPENP.sub(r"\1", highlighted)

                signature_yaml = build_signature_yaml(
                    gold_text=gold_text,
                    sentence_text=highlighted,
                    target_type=target_type,
                    outgoing=outgoing,
                    incoming=incoming,
                    co_mentions=co_mentions,
                    m_open=args.m_open,
                    m_close=args.m_close,
                )

                out_fh.write(json.dumps({
                    "topic_id": int(tid),
                    "doc_id": int(doc_ids[pid]),
                    "para_idx": int(pid),
                    "sent_idx": int(sent_idx),
                    "gold_span": [int(gs), int(ge)],
                    "gold_text": gold_text,
                    "cluster_id": int(cid),
                    "signature": signature_yaml
                }, ensure_ascii=False) + "\n")

    out_fh.close()
    print(f"Wrote signatures to {args.out_path}")

if __name__ == "__main__":
    main()
