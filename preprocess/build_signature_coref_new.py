#!/usr/bin/env python3
"""
Stage 2: Build YAML-style signatures for cross-encoder input from SciCo gold mentions
and TANL-based KG clusters (stage-1 output that merges entity & coref).

Compared to the original build_signature.py which worked directly on flat TANL entities,
this script:

  * Loads a KG JSONL where each line corresponds to a (topic_id, doc_id) pair and contains:
        {
          "topic_id": int,
          "doc_id": int,
          "mentions": [
              {
                  "id": int,
                  "surface": str,
                  "canonical": str,          # may equal surface
                  "source": "entity"|"coref",
                  "topic_id": int,
                  "doc_id": int,
                  "para_idx": int,
                  "span": [int, int],        # TANL-internal span (not used in matching)
                  "type": str or null
              },
              ...
          ],
          "clusters": [
              {
                  "key": str,                # normalized canonical name
                  "canonical": str,
                  "names": [str, ...],
                  "types": [str, ...],
                  "mention_ids": [int, ...],
                  "rel_out": [
                      {"predicate": str,
                       "object": str,        # cluster key of object
                       "object_text": str},  # surface form in TANL
                      ...
                  ],
                  "rel_in": [
                      {"predicate": str,
                       "subject": str,       # cluster key of subject
                       "subject_text": str},
                      ...
                  ],
              },
              ...
          ]
        }

  * For each SciCo gold mention (topic, paragraph, span, cluster_id), it:
        - finds candidate mentions from the KG in the same topic & paragraph,
        - maps each mention surface back to SciCo token spans (possibly multiple hits),
        - scores candidates by a combination of:
              text similarity between gold_text and (surface / canonical),
              positional closeness in SciCo token indices,
              preference for source="entity" over "coref",
        - chooses the best mention (if any), then the corresponding cluster,
        - builds a signature from cluster-level relations (outgoing/incoming) and co-mentions.

  * Tracks provenance via match_source:
        - "entity"  -> matched via a TANL entity mention
        - "coref"   -> matched via a TANL coref-only mention
        - "none"    -> no KG match; signature has empty relations, only context.

Input:
  - SciCo split (train/validation/test) via HF datasets "allenai/scico"
  - Stage-1 KG JSONL as described above

Output (JSONL):
  one line per GOLD mention:
    {
      "topic_id": int,
      "doc_id": int,
      "para_idx": int,
      "sent_idx": int,
      "gold_span": [int, int],
      "gold_text": str,
      "cluster_id": int,         # SciCo cluster id of the gold mention
      "match_source": "entity" | "coref" | "none",
      "signature": str           # YAML card as in build_signature_yaml
    }
"""

import argparse
import json
import os
import re
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

# --------------------------- SciCo span finder ------------------------------

def find_token_spans_for_string(par_tokens: List[str], mention: str,
                                max_window: int = 20) -> List[Tuple[int,int]]:
    """
    Map a mention string back to paragraph token spans by sliding-window detok equality.

    This is identical to the helper in build_signature.py, but we keep *all*
    hits instead of collapsing to a single first span at precompute time.
    """
    tgt = norm(mention)
    N = len(par_tokens)
    hits: List[Tuple[int,int]] = []
    for i in range(N):
        acc: List[str] = []
        for j in range(i+1, min(N, i+max_window)+1):
            acc.append(par_tokens[j-1])
            if norm(detok(acc)) == tgt:
                hits.append((i, j))
                break
            if len(" ".join(acc)) > len(mention) + 15:
                break
    return hits

def find_sentence_idx(sent_spans: List[Tuple[int,int]],
                      token_span: Tuple[int,int]) -> int:
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

# --------------------------- text similarity --------------------------------

def text_sim(a: str, b: str) -> float:
    """
    Simple token-overlap similarity in [0,1].
    Uses normalized, lowercased tokens and ignores empty strings.
    """
    ta = [t for t in norm(a).split() if t]
    tb = [t for t in norm(b).split() if t]
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    inter = len(sa & sb)
    denom = max(len(sa), len(sb))
    return inter / denom

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
    top_key = f"{m_open}{gold_text}{m_close}"
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
    return yaml.safe_dump(card, sort_keys=False, allow_unicode=True).strip()

# --------------------------- matching helpers -------------------------------

def classify_text_sim(sim: float) -> int:
    """
    Bucket similarity into 0..3 (lower = better).
      0: >= 0.9   (near-identical)
      1: >= 0.6
      2: >= 0.3
      3: otherwise
    """
    if sim >= 0.9:
        return 0
    if sim >= 0.6:
        return 1
    if sim >= 0.3:
        return 2
    return 3

def best_pos_for_mention(scico_spans: List[Tuple[int,int]],
                         gs: int, ge: int,
                         delta_pos: int, delta_len: int) -> Tuple[int,int]:
    """
    Given all SciCo spans for a mention and a gold span (gs,ge),
    return (pos_cls, pos_dist) where:
      - pos_cls in {0,2} (0 = within fuzz window, 2 = outside)
      - pos_dist = |start_diff| + |len_diff| for the best span
    """
    if not scico_spans:
        return 2, 10**9
    len_g = ge - gs
    best_cls = 2
    best_dist = 10**9
    for (ps, pe) in scico_spans:
        len_p = pe - ps
        pos_ok = (abs(ps - gs) <= delta_pos and
                  abs(len_p - len_g) <= delta_len)
        pos_dist = abs(ps - gs) + abs(len_p - len_g)
        cls = 0 if pos_ok else 2
        if (cls, pos_dist) < (best_cls, best_dist):
            best_cls, best_dist = cls, pos_dist
    return best_cls, best_dist

def score_mention_candidate(m: Dict,
                            gs: int, ge: int,
                            gold_text: str,
                            delta_pos: int, delta_len: int) -> Tuple[int,int,int,int]:
    """
    Return a 4-tuple used for ranking candidates (lower is better):
      (text_cls, pos_cls, source_priority, pos_dist)
    where:
      - text_cls: 0..3 from classify_text_sim
      - pos_cls:  0 or 2 from best_pos_for_mention
      - source_priority: 0 for entity, 1 for coref/other
      - pos_dist: integer tie-break inside same bucket
    """
    # Text similarity against surface and canonical if present
    surf = m.get("surface") or ""
    canon = m.get("canonical") or surf
    sim_surf  = text_sim(gold_text, surf)
    sim_canon = text_sim(gold_text, canon)
    sim = max(sim_surf, sim_canon)
    text_cls = classify_text_sim(sim)

    # Positional similarity
    scico_spans: List[Tuple[int,int]] = m.get("scico_spans") or []
    pos_cls, pos_dist = best_pos_for_mention(scico_spans, gs, ge, delta_pos, delta_len)

    # Prefer entities over coref
    source = m.get("source") or ""
    source_priority = 0 if source == "entity" else 1

    return (text_cls, pos_cls, source_priority, pos_dist)

def best_match(mentions: List[Dict],
               gs: int, ge: int,
               gold_text: str,
               delta_pos: int, delta_len: int) -> Optional[Dict]:
    """Return the single best mention from a list, or None if list is empty."""
    best = None
    best_m = None
    for m in mentions:
        scico_spans = m.get("scico_spans") or []
        if not scico_spans:
            continue
        tup = score_mention_candidate(m, gs, ge, gold_text, delta_pos, delta_len)
        if best is None or tup < best:
            best = tup
            best_m = m
    return best_m

# --------------------------- main ------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg_path", required=True,
                    help="JSONL with KG clusters and mentions (stage-1 output)")
    ap.add_argument("--split", default="test",
                    choices=["train","validation","test"])
    ap.add_argument("--out_path", default="scico_signatures_stage2.jsonl")
    ap.add_argument("--k", type=int, default=-1,
                    help="Only process first K topics; -1 = all")

    # Fuzzy match thresholds for spans
    ap.add_argument("--delta_pos", type=int, default=3)
    ap.add_argument("--delta_len", type=int, default=3)

    # Signature content controls
    ap.add_argument("--max_rel", type=int, default=6,
                    help="cap number of incoming/outgoing relations")
    ap.add_argument("--max_com", type=int, default=6,
                    help="cap number of co-mentions")
    ap.add_argument("--m_open", default="<m>")
    ap.add_argument("--m_close", default="</m>")

    args = ap.parse_args()

    # Load SciCo
    ds = load_dataset("allenai/scico")[args.split]
    topic_order = [int(ds[i]["id"]) for i in range(len(ds))]
    if args.k > 0:
        topic_order = topic_order[:args.k]

    # Load KG (stage-1 output)
    assert os.path.exists(args.kg_path), f"Missing: {args.kg_path}"
    kg_by_doc: Dict[Tuple[int,int], Dict] = {}
    with open(args.kg_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            tid = int(rec["topic_id"])
            did = int(rec["doc_id"])
            kg_by_doc[(tid, did)] = rec

    out_fh = open(args.out_path, "w", encoding="utf-8")

    # Map (topic_id, doc_id) -> mention_id -> cluster_key
    mention_cluster_maps: Dict[Tuple[int,int], Dict[int,str]] = {}

    for tid in tqdm(topic_order, desc="Building signatures (stage 2)"):
        row = next(r for r in ds if int(r["id"]) == tid)

        tokens_by_para: List[List[str]] = row["tokens"]
        doc_ids: List[int] = row["doc_ids"]
        sent_spans_by_para = row.get("sentences", None)
        mentions = row["mentions"]  # list of [pid, s, e, cluster_id]

        # normalize gold spans (right-open) and group per paragraph
        gold_by_para: Dict[int, List[Tuple[int,int,int]]] = defaultdict(list)
        for (pid, s, e, cid) in mentions:
            s = int(s); e = int(e)
            gold_by_para[int(pid)].append((s, e, int(cid)))

        # For each paragraph in this topic
        for pid, toks in enumerate(tokens_by_para):
            did = int(doc_ids[pid])
            kg_doc = kg_by_doc.get((tid, did))
            if kg_doc is None:
                kg_mentions = []
                clusters = {}
                mention_to_cluster = {}
            else:
                kg_mentions = kg_doc.get("mentions", [])
                clusters_list = kg_doc.get("clusters", [])
                clusters = {c["key"]: c for c in clusters_list}

                key_doc = (tid, did)
                if key_doc not in mention_cluster_maps:
                    m2c = {}
                    for c in clusters_list:
                        ck = c["key"]
                        for mid in c.get("mention_ids", []):
                            m2c[int(mid)] = ck
                    mention_cluster_maps[key_doc] = m2c
                mention_to_cluster = mention_cluster_maps.get((tid, did), {})

                # For each mention in this KG doc, compute SciCo spans lazily
                for m in kg_mentions:
                    if int(m.get("para_idx", -1)) != pid:
                        continue
                    if "scico_spans" in m:
                        continue
                    surface = m.get("surface") or ""
                    spans = find_token_spans_for_string(toks, surface)
                    if not spans:
                        canon = m.get("canonical")
                        if canon and canon != surface:
                            spans = find_token_spans_for_string(toks, canon)
                    m["scico_spans"] = spans

            # sentence indexer for this paragraph
            if sent_spans_by_para:
                para_sent_spans = [
                    (int(a), int(b)) for (a,b) in sent_spans_by_para[pid]
                ]
            else:
                para_sent_spans = [(0, len(toks))]

            # For each gold mention in this paragraph: build YAML signature
            for (gs, ge, cid) in gold_by_para.get(pid, []):
                gold_text = detok(toks[gs:ge+1])

                # choose sentence
                sent_idx = find_sentence_idx(para_sent_spans, (gs, ge))
                ss, se = para_sent_spans[sent_idx]
                sent_text = detok(toks[ss:se])

                # KG mentions in this paragraph
                kg_mentions_para = [
                    m for m in kg_mentions
                    if int(m.get("para_idx", -1)) == pid
                ]

                match_source = "none"
                match_mention: Optional[Dict] = None
                match_cluster = None

                # ----- Pass 1: entities only -----
                entity_mentions = [
                    m for m in kg_mentions_para
                    if m.get("source") == "entity" and m.get("scico_spans")
                ]
                if entity_mentions:
                    mm = best_match(
                        entity_mentions, gs, ge, gold_text,
                        delta_pos=args.delta_pos,
                        delta_len=args.delta_len,
                    )
                    if mm is not None:
                        match_mention = mm
                        match_source = "entity"

                # ----- Pass 2: allow coref if no entity match -----
                if match_mention is None and kg_mentions_para:
                    mm = best_match(
                        [m for m in kg_mentions_para if m.get("scico_spans")],
                        gs, ge, gold_text,
                        delta_pos=args.delta_pos,
                        delta_len=args.delta_len,
                    )
                    if mm is not None:
                        match_mention = mm
                        match_source = mm.get("source") or "coref"

                # Build outgoing / incoming / co_mentions from the matched cluster
                target_type = None
                outgoing: List[Dict] = []
                incoming: List[Dict] = []
                co_mentions: List[Dict] = []

                if match_mention is not None:
                    mid = int(match_mention["id"])
                    ck = mention_to_cluster.get(mid)
                    if ck is not None and ck in clusters:
                        match_cluster = clusters[ck]

                        # choose a target type from cluster types if available
                        types = match_cluster.get("types") or []
                        if types:
                            target_type = types[0]

                        # Outgoing relations
                        for r in match_cluster.get("rel_out", []):
                            outgoing.append({
                                "predicate": r.get("predicate"),
                                "object_text": r.get("object_text"),
                                "object_cluster": r.get("object"),
                            })

                        # Incoming relations
                        for r in match_cluster.get("rel_in", []):
                            incoming.append({
                                "predicate": r.get("predicate"),
                                "subject_text": r.get("subject_text"),
                                "subject_cluster": r.get("subject"),
                            })

                        # Co-mentions: other mentions in same cluster, in same sentence
                        m_scico_spans = match_mention.get("scico_spans") or []
                        if m_scico_spans:
                            best_m_span = max(
                                m_scico_spans,
                                key=lambda sp: max(0, min(sp[1],se) - max(sp[0],ss)),
                            )
                        else:
                            best_m_span = None

                        for other_mid in match_cluster.get("mention_ids", []):
                            other_mid = int(other_mid)
                            if other_mid == mid:
                                continue
                            other_m = next((m for m in kg_mentions
                                            if int(m["id"]) == other_mid), None)
                            if other_m is None:
                                continue
                            if int(other_m.get("para_idx", -1)) != pid:
                                continue
                            spans_o = other_m.get("scico_spans")
                            if not spans_o:
                                surface_o = other_m.get("surface") or ""
                                spans_o = find_token_spans_for_string(toks, surface_o)
                                if not spans_o:
                                    canon_o = other_m.get("canonical")
                                    if canon_o and canon_o != surface_o:
                                        spans_o = find_token_spans_for_string(toks, canon_o)
                                other_m["scico_spans"] = spans_o
                            if not spans_o:
                                continue
                            inside_sentence = any(
                                (ps >= ss and pe <= se) for (ps,pe) in spans_o
                            )
                            if not inside_sentence:
                                continue
                            item = {"text": other_m.get("surface") or ""}
                            if other_m.get("type"):
                                item["type"] = other_m["type"]
                            co_mentions.append(item)

                # cap lists deterministically
                outgoing = sorted(outgoing, key=lambda x: (x["predicate"], x.get("object_text","")))[:args.max_rel]
                incoming = sorted(incoming, key=lambda x: (x["predicate"], x.get("subject_text","")))[:args.max_rel]
                co_mentions = sorted(co_mentions, key=lambda x: (x.get("text",""), x.get("type","")))[:args.max_com]

                # highlight target in sentence
                pre = detok(toks[ss:gs])
                mid = gold_text
                post = detok(toks[ge:se])
                highlighted = (
                    pre
                    + (" " if pre and not pre.endswith(" ") else "")
                    + args.m_open + mid + args.m_close
                    + ("" if (not post or post.startswith(" ")) else " ")
                    + post
                ).strip()

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
                    "doc_id": int(did),
                    "para_idx": int(pid),
                    "sent_idx": int(sent_idx),
                    "gold_span": [int(gs), int(ge)],
                    "gold_text": gold_text,
                    "cluster_id": int(cid),
                    "match_source": match_source,
                    "signature": signature_yaml,
                }, ensure_ascii=False) + "\n")

    out_fh.close()
    print(f"Wrote signatures to {args.out_path}")

if __name__ == "__main__":
    main()
