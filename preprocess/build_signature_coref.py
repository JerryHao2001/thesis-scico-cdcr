#!/usr/bin/env python3
"""
Build YAML signatures for SciCo gold mentions using TANL + WDCR (within-doc coref).

Pipeline
  1) Load SciCo rows, TANL paragraphs, and WDCR paragraphs
  2) For each paragraph:
       a) Parse TANL brackets -> predicted mentions with type + outgoing rels
       b) Parse WDCR brackets -> coref links (mention1 ~ mention2)
       c) Map all bracket strings to token spans
       d) Union-find mentions into WDCR clusters
  3) For each gold mention:
       a) fuzzy-match to ANY mention span; if matched, pick its cluster
       b) aggregate type/relations across that cluster
       c) derive incoming edges from other mentions pointing to any alias in the cluster
       d) pick co-mentions from the same sentence
       e) output concise YAML signature with <m>…</m> at top-level

Output JSONL: one line per gold mention:
  {
    topic_id, doc_id, para_idx, sent_idx,
    gold_span, gold_text, cluster_id,
    signature: "<yaml string>"
  }
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

# [ mention | TYPE | r1 = obj ; r2 = obj2 ]
BRACKET_TANL_RE = re.compile(
    r"\[\s*(?P<mention>.+?)\s*\|\s*(?P<etype>[^|\]]+?)\s*(?:\|\s*(?P<rels>.+?))?\s*\]"
)
REL_PAIR_RE = re.compile(r"(?P<rname>[a-zA-Z_]+)\s*=\s*(?P<arg>[^;|]+)")

def parse_tanl(tanl_text: str):
    ents = []
    for m in BRACKET_TANL_RE.finditer(tanl_text or ""):
        mention = m.group("mention").split("|")[0].strip()
        etype = (m.group("etype") or "").strip()
        rels_str = m.group("rels") or ""
        rels = []
        for r in REL_PAIR_RE.finditer(rels_str):
            pred = r.group("rname").strip()
            obj = r.group("arg").strip()
            rels.append((pred, obj))
        ents.append({
            "text": mention,
            "type": etype if etype else None,
            "rels_out": rels,   # subject = this mention
        })
    return ents

# --------------------------- WDCR parsing -----------------------------------
# WDCR brackets: "[ m1 | m2 ]" meaning m1 corefers to m2 (strings only, no types)
BRACKET_WDCR_RE = re.compile(r"\[\s*(?P<m1>.+?)\s*\|\s*(?P<m2>.+?)\s*\]")

def parse_wdcr(wdcr_text: str):
    links = []  # list of (m1_text, m2_text)
    for m in BRACKET_WDCR_RE.finditer(wdcr_text or ""):
        m1 = m.group("m1").split("|")[0].strip()
        m2 = m.group("m2").split("|")[0].strip()
        links.append((m1, m2))
    return links

# --------------------------- span mapping -----------------------------------

def find_token_spans_for_string(par_tokens: List[str], mention: str, max_window: int = 20) -> List[Tuple[int,int]]:
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

# --------------------------- sentences & context ----------------------------

def find_sentence_idx(sent_spans: List[Tuple[int,int]], token_span: Tuple[int,int]) -> int:
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

# --------------------------- union-find -------------------------------------

class DSU:
    def __init__(self, n): self.p=list(range(n)); self.r=[0]*n
    def find(self,x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]
            x=self.p[x]
        return x
    def union(self,a,b):
        ra, rb = self.find(a), self.find(b)
        if ra==rb: return
        if self.r[ra]<self.r[rb]: ra,rb=rb,ra
        self.p[rb]=ra
        if self.r[ra]==self.r[rb]: self.r[ra]+=1

# --------------------------- YAML builder -----------------------------------

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

# --------------------------- main ------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_path", required=True, help="TANL JSONL (per paragraph)",)
    ap.add_argument("--coref_path", required=True, help="WDCR JSONL (per paragraph)",)
    ap.add_argument("--split", default="test", choices=["train","validation","test"])
    ap.add_argument("--out_path", default="scico_signatures_coref.jsonl")
    ap.add_argument("--k", type=int, default=-1, help="first K topics only; -1 = all")

    # Matching & gold span normalization
    ap.add_argument("--delta_pos", type=int, default=2)
    ap.add_argument("--delta_len", type=int, default=2)

    # Caps & markers
    ap.add_argument("--max_rel", type=int, default=6)
    ap.add_argument("--max_com", type=int, default=6)
    ap.add_argument("--m_open", default="<m>")
    ap.add_argument("--m_close", default="</m>")
    args = ap.parse_args()

    # Load SciCo
    ds = load_dataset("allenai/scico")[args.split]
    topic_order = [int(ds[i]["id"]) for i in range(len(ds))]
    if args.k > 0:
        topic_order = topic_order[:args.k]

    # Load TANL predictions
    assert os.path.exists(args.pred_path), f"Missing TANL: {args.pred_path}"
    tanl_by_para = {}
    with open(args.pred_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tanl_by_para[(int(rec["topic_id"]), int(rec["para_idx"]))] = rec

    # Load WDCR predictions
    assert os.path.exists(args.coref_path), f"Missing WDCR: {args.coref_path}"
    coref_by_para = {}
    with open(args.coref_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            coref_by_para[(int(rec["topic_id"]), int(rec["para_idx"]))] = rec

    out_fh = open(args.out_path, "w", encoding="utf-8")

    for tid in tqdm(topic_order, desc="Signatures with WDCR"):
        row = next(r for r in ds if int(r["id"]) == tid)

        tokens_by_para = row["tokens"]
        doc_ids = row["doc_ids"]
        sent_spans_by_para = row.get("sentences", None)
        mentions = row["mentions"]  # [pid, s, e, cluster_id]

        # group gold by para; normalize to right-open
        gold_by_para = defaultdict(list)
        for (pid, s, e, cid) in mentions:
            s = int(s); e = int(e)

            e = min(e + 1, len(tokens_by_para[pid]))
            gold_by_para[int(pid)].append((s, e, int(cid)))

        for pid, toks in enumerate(tokens_by_para):
            key = (tid, pid)
            tanl_rec = tanl_by_para.get(key)
            coref_rec = coref_by_para.get(key)
            if tanl_rec is None and coref_rec is None:
                continue

            tanl_text = tanl_rec.get("tanl_output", "") if tanl_rec else ""
            coref_text = coref_rec.get("tanl_output", "") if coref_rec else ""  # same field name

            # Parse TANL & WDCR
            tanl_ents = parse_tanl(tanl_text)  # list of dicts: text,type,rels_out
            wdcr_links = parse_wdcr(coref_text)  # list of (m1, m2)

            # Map TANL mentions to spans (take first match)
            tanl_spans = []
            for ent in tanl_ents:
                spans = find_token_spans_for_string(toks, ent["text"])
                tanl_spans.append(spans[0] if spans else None)

            # Build mention inventory for clustering:
            #   Start with all TANL mentions
            mention_texts = [e["text"] for e in tanl_ents]
            mention_types = [e["type"] for e in tanl_ents]
            mention_rels  = [e["rels_out"] for e in tanl_ents]
            mention_spans = tanl_spans

            #   Add WDCR-only mentions if they do not appear in TANL list
            #   (collect strings from wdcr links; we'll try find spans for them)
            wdcr_strings = set()
            for a,b in wdcr_links:
                wdcr_strings.add(a.strip())
                wdcr_strings.add(b.strip())
            for s in sorted(wdcr_strings):
                if any(norm(s) == norm(x) for x in mention_texts):
                    continue
                spans = find_token_spans_for_string(toks, s)
                sp0 = spans[0] if spans else None
                mention_texts.append(s)
                mention_types.append(None)
                mention_rels.append([])   # no relations from WDCR-only nodes
                mention_spans.append(sp0)

            n = len(mention_texts)
            dsu = DSU(n)

            # Union by WDCR coref links (string-normalized)
            # Link *all* indices whose text matches either side
            norm_to_ids = defaultdict(list)
            for i, t in enumerate(mention_texts):
                norm_to_ids[norm(t)].append(i)
            for a,b in wdcr_links:
                la = norm(a)
                lb = norm(b)
                if la in norm_to_ids and lb in norm_to_ids:
                    for i in norm_to_ids[la]:
                        for j in norm_to_ids[lb]:
                            dsu.union(i,j)

            # Clusters: root -> member indices
            root_to_members = defaultdict(list)
            for i in range(n):
                root_to_members[dsu.find(i)].append(i)

            # Precompute object_text -> type (majority by cluster of that object, from TANL typed members)
            obj_text_to_type = {}
            for root, mems in root_to_members.items():
                # gather aliases
                aliases = [mention_texts[i] for i in mems]
                # majority type from TANL members only
                types = [mention_types[i] for i in mems if mention_types[i]]
                maj = None
                if types:
                    # majority vote
                    from collections import Counter
                    maj = Counter(types).most_common(1)[0][0]
                for a in aliases:
                    obj_text_to_type[norm(a)] = maj

            # Sentence spans for the paragraph
            if sent_spans_by_para:
                para_sent_spans = [(int(a), int(b)) for (a,b) in sent_spans_by_para[pid]]
            else:
                para_sent_spans = [(0, len(toks))]

            # Build incoming-lookup: for each mention, its rels_out (subject = mention_i)
            # Later we’ll collect incoming for the TARGET CLUSTER by matching object_text to any alias
            all_outgoing = []
            for i in range(len(mention_texts)):
                for (pred_name, obj_text) in mention_rels[i]:
                    all_outgoing.append((i, pred_name, obj_text))

            # For each GOLD mention in this paragraph -> build signature
            for (gs, ge, cid) in gold_by_para.get(pid, []):
                gold_text = detok(toks[gs:ge])

                # sentence for context
                sent_idx = 0
                if para_sent_spans:
                    sent_idx = find_sentence_idx(para_sent_spans, (gs, ge))
                ss, se = para_sent_spans[sent_idx]
                sent_text = detok(toks[ss:se])

                # fuzzy match to ANY mention span, prefer IoU then start distance
                match_idx = None
                best_score = (-1, 10**9)  # (IoU*1000, start_dist)
                for i, sp in enumerate(mention_spans):
                    if sp is None:
                        continue
                    ps, pe = sp
                    # IoU on tokens
                    inter = max(0, min(ge,pe) - max(gs,ps))
                    union = (ge-gs) + (pe-ps) - inter
                    iou = inter/union if union>0 else 0.0
                    if fuzzy_hit((gs,ge), sp, delta_pos=args.delta_pos, delta_len=args.delta_len):
                        score = (int(round(iou*1000)), abs(ps-gs))
                        if score > best_score:
                            best_score = score
                            match_idx = i

                # Aggregate from the matched cluster (if any)
                target_type = None
                outgoing = []
                incoming = []
                co_mentions = []

                if match_idx is not None:
                    root = dsu.find(match_idx)
                    members = root_to_members[root]
                    aliases = [mention_texts[i] for i in members]

                    # majority type from TANL-typed members
                    from collections import Counter
                    typed = [mention_types[i] for i in members if mention_types[i]]
                    target_type = Counter(typed).most_common(1)[0][0] if typed else None

                    # Outgoing = union of rels_out from all members
                    seen_out = set()
                    for i in members:
                        for (pred_name, obj_text) in mention_rels[i]:
                            key = (pred_name, norm(obj_text))
                            if key in seen_out: 
                                continue
                            seen_out.add(key)
                            out_item = {
                                "predicate": pred_name,
                                "object_text": obj_text
                            }
                            obj_t = obj_text_to_type.get(norm(obj_text))
                            if obj_t:
                                out_item["object_type"] = obj_t
                            outgoing.append(out_item)

                    # Incoming = any mention (subject) whose rels_out points to ANY alias in this cluster
                    alias_norms = {norm(a) for a in aliases}
                    seen_in = set()
                    for (subj_idx, pred_name, obj_text) in all_outgoing:
                        if norm(obj_text) in alias_norms:
                            key = (pred_name, norm(mention_texts[subj_idx]))
                            if key in seen_in:
                                continue
                            seen_in.add(key)
                            incoming.append({
                                "predicate": pred_name,
                                "subject_text": mention_texts[subj_idx],
                                "subject_type": mention_types[subj_idx]
                            })

                    # Co-mentions from the same sentence (TANL mentions preferred)
                    gs_sent, ge_sent = ss, se
                    for i, sp in enumerate(mention_spans):
                        if sp is None: 
                            continue
                        st, en = sp
                        if not (st >= gs_sent and en <= ge_sent):
                            continue
                        # skip if it's overlapping like the target
                        if match_idx == i:
                            continue
                        item = {"text": mention_texts[i]}
                        if mention_types[i]:
                            item["type"] = mention_types[i]
                        co_mentions.append(item)

                # Cap lists
                outgoing = sorted(outgoing, key=lambda x: (x["predicate"], x.get("object_text","")))[:args.max_rel]
                incoming = sorted(incoming, key=lambda x: (x["predicate"], x.get("subject_text","")))[:args.max_rel]
                co_mentions = sorted(co_mentions, key=lambda x: (x.get("text",""), x.get("type","")))[:args.max_com]

                # Highlight target in the sentence
                pre = detok(toks[ss:gs])
                mid = gold_text
                post = detok(toks[ge:se])
                highlighted = (pre + (" " if pre and not pre.endswith(" ") else "")
                               + args.m_open + mid + args.m_close
                               + ("" if (not post or post.startswith(" ")) else " ") + post).strip()
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
