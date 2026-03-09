#!/usr/bin/env python3
"""
Build signatures for SciCo gold mentions by combining:
  (1) TANL relation/type extraction per paragraph, and
  (2) TANL within-paragraph coreference (WD-coref) extraction per paragraph (optional).

This script outputs one JSONL record per GOLD mention. Each record contains a 'signature'
string intended for cross-encoder / antecedent-style coreference models.

Key ideas (soft union):
  - Use WD-coref to derive a within-paragraph canonical mention + aliases.
  - Propagate TANL type and relations across the WD-coref cluster (when matched).
  - Canonicalize relation arguments when they match a known alias in the paragraph.
  - Keep the original gold mention surface/context (do NOT overwrite gold_text).

Signature formats:
  - flat (recommended): compact tag-based linearization (good for BERT token budget)
  - yaml: human-readable YAML card (useful for debugging)

Inputs:
  --pred_path: JSONL with TANL relation/type extraction per paragraph.
      Expected keys include: topic_id, para_idx, doc_id, tanl_output, sentences (optional)
  --coref_pred_path: JSONL with TANL within-paragraph coref extraction per paragraph (optional).
      Expected keys include: topic_id, para_idx, tanl_output

SciCo spans:
  SciCo mentions are [pid, s, e, cluster_id] where e is inclusive.
  Internally, we frequently convert to right-open spans (s, e+1) for slicing and overlap.

Output JSONL (one per gold mention):
  {
    "topic_id": int,
    "doc_id": int,
    "para_idx": int,
    "sent_idx": int,
    "gold_span": [s, e],   # inclusive end (SciCo convention)
    "gold_text": str,
    "cluster_id": int,
    "signature": str
  }

Conservative defaults:
  - signature_format=flat
  - include_comentions=False
  - include_canonical_context=False
  - max_aliases=4, max_rel=6

Additional features added in this version:
  - Bidirectional relations: for symmetric predicates (default: compare, conjunction),
    add the relation to both endpoints when the object resolves to another node.
  - Extended context fallback: if a gold mention cannot be matched to any TANL node
    (or if the constructed signature is an outlier/invalid), fall back to a multi-sentence
    context window (still highlighted with <m>...</m>).
  - JSONL debug flags: emit match/fallback metadata fields to support later error analysis.
  - Outlier guard: if a signature exceeds --max_signature_chars, rebuild it as a context-only
    fallback (treat as no-match).
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import yaml
from datasets import load_dataset
from tqdm import tqdm


# Relations whose semantics are symmetric (two-sided). If the object resolves to another
# paragraph node, we add a reverse edge so both mentions can see the relation.
SYMMETRIC_REL_PRED_KEYS = {"compare", "conjunction"}


def normalize_predicate(pred: str) -> str:
    """Normalize a predicate for key comparisons / deduping."""
    return re.sub(r"\s+", " ", (pred or "").strip().lower())

# --------------------------- tokenization helpers ---------------------------

_PUNCT = re.compile(r"\s+([,.;:%)\]])")
_OPENP = re.compile(r"([\[(])\s+")

def detok(tokens: List[str]) -> str:
    s = " ".join(tokens)
    s = _PUNCT.sub(r"\1", s)
    s = _OPENP.sub(r"\1", s)
    return s

def norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s

def token_set(s: str) -> set:
    return set([t for t in norm(s).split() if t])

def is_acronym_of(short: str, long: str) -> bool:
    """
    Heuristic: short is an acronym of long.
    Examples: "cnn" vs "Cable News Network" -> True
    """
    a = re.sub(r"[^a-zA-Z]", "", short or "")
    if len(a) < 2 or len(a) > 10:
        return False
    if not a.isupper() and not a.islower():
        # mixed-case acronym is rare; allow but keep strict
        pass
    toks = [t for t in re.split(r"[^a-zA-Z0-9]+", long or "") if t]
    if len(toks) < 2:
        return False
    initials = "".join(t[0] for t in toks if t[0].isalpha())
    return norm(a) == norm(initials)

# --------------------------- span utils -------------------------------------

Span = Tuple[int, int]  # right-open [start, end)

def span_overlap(a: Span, b: Span) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def fuzzy_hit(gold_span: Span, pred_span: Span, delta_pos: int = 2, delta_len: int = 2) -> bool:
    gs, ge = gold_span
    ps, pe = pred_span
    len_g = ge - gs
    len_p = pe - ps
    return abs(ps - gs) <= delta_pos and abs(len_p - len_g) <= delta_len

def find_sentence_idx(sent_spans: List[Span], token_span: Span) -> int:
    """Choose the sentence index whose span overlaps the target span the most."""
    if not sent_spans:
        return 0
    gs, ge = token_span
    best_i, best_overlap = 0, -1
    for i, (ss, se) in enumerate(sent_spans):
        inter = max(0, min(ge, se) - max(gs, ss))
        if inter > best_overlap:
            best_overlap = inter
            best_i = i
    return best_i

def highlight_in_sentence(
    toks: List[str],
    sent_span: Span,
    target_span: Span,
    m_open: str = "<m>",
    m_close: str = "</m>",
) -> str:
    """Detok sentence tokens and insert <m>...</m> around target_span."""
    ss, se = sent_span
    gs, ge = target_span
    gs = max(gs, ss)
    ge = min(ge, se)
    pre = detok(toks[ss:gs])
    mid = detok(toks[gs:ge])
    post = detok(toks[ge:se])
    s = (pre + (" " if pre and not pre.endswith(" ") else "")
         + m_open + mid + m_close
         + ("" if (not post or post.startswith(" ")) else " ") + post).strip()
    # normalize stray spaces near punctuation
    s = _PUNCT.sub(r"\1", s)
    s = _OPENP.sub(r"\1", s)
    return s


def _sentence_window_indices(n_sents: int, center: int, window_sents: int) -> Tuple[int, int]:
    """Return inclusive [lo, hi] sentence indices for a centered window."""
    window_sents = max(1, int(window_sents))
    # distribute remainder to the right
    left = (window_sents - 1) // 2
    right = (window_sents - 1) - left
    lo = max(0, center - left)
    hi = min(n_sents - 1, center + right)
    return lo, hi


def build_highlighted_context_window(
    toks: List[str],
    sent_spans: List[Span],
    sent_idx: int,
    target_span: Span,
    window_sents: int,
    max_tokens: int,
    m_open: str,
    m_close: str,
) -> str:
    """Build a multi-sentence context window (token span) with <m>..</m> highlighting.

    Strategy:
      1) Try a centered sentence window of size `window_sents`.
      2) If it exceeds `max_tokens`, shrink by removing one sentence from each side.
      3) If still too large (e.g., single long sentence), clip to a token window around the target.

    Note: max_tokens is in *word tokens* (SciCo tokens), not WordPiece tokens.
    """
    if not sent_spans:
        sent_spans = [(0, len(toks))]
    n_sents = len(sent_spans)
    sent_idx = max(0, min(sent_idx, n_sents - 1))
    target_span = (max(0, target_span[0]), min(len(toks), target_span[1]))

    # 1) shrink sentence window until within max_tokens (or down to 1 sentence)
    w = max(1, int(window_sents))
    while True:
        lo, hi = _sentence_window_indices(n_sents, sent_idx, w)
        win_span = (sent_spans[lo][0], sent_spans[hi][1])
        if max_tokens <= 0 or (win_span[1] - win_span[0]) <= max_tokens or w <= 1:
            break
        # remove one sentence from each side (preserve centering)
        w = max(1, w - 2)

    lo, hi = _sentence_window_indices(n_sents, sent_idx, w)
    win_span = (sent_spans[lo][0], sent_spans[hi][1])

    # 2) if still too large, clip to token window around the target
    if max_tokens > 0 and (win_span[1] - win_span[0]) > max_tokens:
        gs, ge = target_span
        # try to center around the mention
        half = max_tokens // 2
        start = max(0, min(gs, (gs + ge) // 2 - half))
        end = start + max_tokens
        if end < ge:
            end = ge
            start = max(0, end - max_tokens)
        end = min(len(toks), end)
        win_span = (start, end)

    return highlight_in_sentence(toks, win_span, target_span, m_open=m_open, m_close=m_close)

# --------------------------- TANL parsing: relations/types -------------------

# [ mention | TYPE | r1 = obj | r2 = obj2 ]  (relations section optional)
BRACKET_RE = re.compile(
    r"\[\s*(?P<mention>[^|\]]+?)\s*\|\s*(?P<etype>[^|\]]+?)\s*(?:\|\s*(?P<rels>.+?))?\s*\]"
)
REL_SEP_RE = re.compile(r"\s*\|\s*")

def parse_relations(rels_str: str) -> List[Tuple[str, str]]:
    """
    Parse relations like:
      "used for = hybrid model | evaluated on = CNN/DailyMail"
    allowing spaces in predicate names and arguments.
    """
    rels: List[Tuple[str, str]] = []
    rels_str = (rels_str or "").strip()
    if not rels_str:
        return rels

    for chunk in REL_SEP_RE.split(rels_str):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        pred, arg = chunk.split("=", 1)
        pred = re.sub(r"\s+", " ", pred).strip()
        arg = re.sub(r"\s+", " ", arg).strip()
        if pred and arg:
            rels.append((pred, arg))
    return rels

def parse_tanl_rel(tanl_text: str) -> List[Dict]:
    entities: List[Dict] = []
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


def filter_tanl_entities(
    entities: List[Dict],
    max_entity_chars: int,
    max_rel_arg_chars: int,
) -> List[Dict]:
    """Filter out obvious TANL hallucination artifacts.

    We apply conservative character-length guards:
      - drop an entity if its mention surface is extremely long
      - drop individual relations whose argument is extremely long
    """
    out: List[Dict] = []
    for e in entities:
        txt = (e.get("text") or "").strip()
        if max_entity_chars > 0 and len(txt) > max_entity_chars:
            continue
        rels = []
        for pred, arg in e.get("rels_out", []) or []:
            if max_rel_arg_chars > 0 and len(arg) > max_rel_arg_chars:
                continue
            rels.append((pred, arg))
        out.append({"text": txt, "type": e.get("type"), "rels_out": rels})
    return out

# --------------------------- TANL parsing: within-doc coref ------------------

# [ mention | antecedent ] (coref model output)
COREF_BRACKET_RE = re.compile(
    r"\[\s*(?P<mention>[^|\]]+?)\s*\|\s*(?P<ante>[^|\]]+?)\s*\]"
)

def parse_tanl_coref(tanl_text: str) -> List[Dict]:
    """
    Parse within-paragraph coref output:
      [ new surface | antecedent surface ]
    First occurrences are often encoded as [ X | X ].
    """
    occ: List[Dict] = []
    for m in COREF_BRACKET_RE.finditer(tanl_text or ""):
        mention = (m.group("mention") or "").strip()
        ante = (m.group("ante") or "").strip()
        if mention:
            occ.append({"text": mention, "ante": ante if ante else mention})
    return occ

# --------------------------- string->span mapping ----------------------------

def find_token_spans_for_string(par_tokens: List[str], mention: str, max_window: int = 20) -> List[Span]:
    """Map a mention string back to paragraph token spans by sliding-window detok equality."""
    tgt = norm(mention)
    N = len(par_tokens)
    hits: List[Span] = []
    for i in range(N):
        acc = []
        for j in range(i + 1, min(N, i + max_window) + 1):
            acc.append(par_tokens[j - 1])
            if norm(detok(acc)) == tgt:
                hits.append((i, j))
                break
            if len(" ".join(acc)) > len(mention) + 15:
                break
    return hits

def align_mentions_to_spans_greedy(
    par_tokens: List[str],
    mentions_in_order: List[str],
    max_window: int = 20,
) -> List[Optional[Span]]:
    """
    Occurrence-aware alignment: for each mention string in text order, pick the first token span
    whose start >= cursor. Fallback to the earliest hit if none meet cursor.
    """
    cursor = 0
    spans: List[Optional[Span]] = []
    for mention in mentions_in_order:
        hits = find_token_spans_for_string(par_tokens, mention, max_window=max_window)
        chosen: Optional[Span] = None
        for (s, e) in hits:
            if s >= cursor:
                chosen = (s, e)
                break
        if chosen is None and hits:
            chosen = hits[0]
        spans.append(chosen)
        if chosen is not None:
            cursor = max(cursor, chosen[1])
    return spans

# --------------------------- WD-coref clustering -----------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

def coref_link_ok(mention: str, antecedent: str, jaccard_thr: float = 0.5) -> bool:
    """Accept all WD-coref links (coref safety filter disabled).

    Rationale: TANL within-paragraph coref can link pronouns / generic anaphors
    (e.g., "it", "this method") to a long antecedent; lexical overlap checks
    like Jaccard will systematically reject those. We therefore trust the WD-coref
    link signal (subject to antecedent resolution to a prior mention occurrence).
    """
    a = norm(mention)
    b = norm(antecedent)
    return bool(a and b)

def build_wd_clusters(
    occ: List[Dict],
    occ_spans: List[Optional[Span]],
    jaccard_thr: float = 0.5,
) -> Tuple[List[int], Dict[int, List[int]], Dict[int, int]]:
    """
    Build WD-coref clusters over occurrences (in one paragraph).

    Returns:
      occ2cluster: list[int] length n, cluster id (0..C-1)
      clusters: dict cluster_id -> member occurrence indices
      cluster2canon: dict cluster_id -> canonical occurrence index (earliest span start, fallback lowest idx)
    """
    n = len(occ)
    uf = UnionFind(n)
    norm_texts = [norm(o["text"]) for o in occ]

    for i in range(n):
        ante = occ[i].get("ante", occ[i]["text"])
        ante_norm = norm(ante)
        if not ante_norm:
            continue
        # resolve to a previous occurrence by surface match (closest previous match)
        j = None
        for k in range(i - 1, -1, -1):
            if norm_texts[k] == ante_norm:
                j = k
                break
        if j is None:
            continue
        if coref_link_ok(occ[i]["text"], ante, jaccard_thr=jaccard_thr):
            uf.union(i, j)

    # compress to ids
    root_to_cid: Dict[int, int] = {}
    occ2cluster: List[int] = []
    for i in range(n):
        r = uf.find(i)
        if r not in root_to_cid:
            root_to_cid[r] = len(root_to_cid)
        occ2cluster.append(root_to_cid[r])

    clusters: Dict[int, List[int]] = defaultdict(list)
    for i, cid in enumerate(occ2cluster):
        clusters[cid].append(i)

    cluster2canon: Dict[int, int] = {}
    for cid, members in clusters.items():
        # pick earliest span start if available; else smallest index
        best = None
        for i in members:
            sp = occ_spans[i]
            if sp is None:
                continue
            if best is None or sp[0] < best[0]:
                best = (sp[0], i)
        if best is not None:
            cluster2canon[cid] = best[1]
        else:
            cluster2canon[cid] = min(members)

    return occ2cluster, clusters, cluster2canon

# --------------------------- paragraph entity nodes --------------------------

@dataclass
class Node:
    node_id: str
    canonical_text: str
    canonical_span: Optional[Span] = None
    spans: List[Span] = field(default_factory=list)
    aliases: set = field(default_factory=set)

    # aggregation buffers
    _type_counts: Counter = field(default_factory=Counter)
    _type_first: List[str] = field(default_factory=list)
    outgoing_raw: List[Tuple[str, str]] = field(default_factory=list)  # (pred, obj_text)

    # finalized
    final_type: str = "unknown"
    outgoing: List[Tuple[str, str]] = field(default_factory=list)  # (pred, obj_canon_or_raw)
    incoming: List[Tuple[str, str, str]] = field(default_factory=list)  # (subj_canon, subj_type, pred)

    def add_type(self, t: Optional[str]):
        if t is None:
            return
        t = t.strip()
        if not t:
            return
        self._type_counts[t] += 1
        if t not in self._type_first:
            self._type_first.append(t)

    def add_outgoing(self, rels: List[Tuple[str, str]]):
        for pred, obj in rels:
            pred = re.sub(r"\s+", " ", (pred or "").strip())
            obj = re.sub(r"\s+", " ", (obj or "").strip())
            if pred and obj:
                self.outgoing_raw.append((pred, obj))

    def finalize_type(self):
        if not self._type_counts:
            self.final_type = "unknown"
            return
        # majority vote; tie -> first seen
        best_count = max(self._type_counts.values())
        best_types = [t for t, c in self._type_counts.items() if c == best_count]
        if len(best_types) == 1:
            self.final_type = best_types[0]
            return
        for t in self._type_first:
            if t in best_types:
                self.final_type = t
                return
        self.final_type = best_types[0]

# --------------------------- signature builders -----------------------------

DEFAULT_TAGS = ("<M>", "<CANON>", "<TYPE>", "<CTX>", "<ALIASES>", "<OUT>", "<IN>", "<CAN_CTX>", "<CO>")

def build_signature_flat(
    gold_text: str,
    highlighted_context: str,
    canonical_text: str,
    typ: str,
    aliases: List[str],
    outgoing_strs: List[str],
    incoming_strs: List[str],
    co_mentions_strs: List[str],
    canonical_context: Optional[str] = None,
    m_open: str = "<m>",
    m_close: str = "</m>",
) -> str:
    parts: List[str] = []
    parts.append(f"<M> {m_open}{gold_text}{m_close}")
    parts.append(f"<CANON> {canonical_text}")
    parts.append(f"<TYPE> {typ}")
    parts.append(f"<CTX> {highlighted_context}")

    if aliases:
        parts.append("<ALIASES> " + " ; ".join(aliases))
    if outgoing_strs:
        parts.append("<OUT> " + " ; ".join(outgoing_strs))
    if incoming_strs:
        parts.append("<IN> " + " ; ".join(incoming_strs))
    if canonical_context:
        parts.append("<CAN_CTX> " + canonical_context)
    if co_mentions_strs:
        parts.append("<CO> " + " ; ".join(co_mentions_strs))

    return " ".join([p for p in parts if p]).strip()

def build_signature_yaml(
    gold_text: str,
    highlighted_context: str,
    canonical_text: str,
    typ: str,
    aliases: List[str],
    outgoing: List[Dict],
    incoming: List[Dict],
    co_mentions: List[Dict],
    canonical_context: Optional[str] = None,
    m_open: str = "<m>",
    m_close: str = "</m>",
) -> str:
    top_key = f"{m_open}{gold_text}{m_close}"
    card = {
        top_key: {
            "canonical": canonical_text,
            "aliases": aliases,
            "type": typ if typ else "unknown",
            "context": highlighted_context,
            "canonical_context": canonical_context if canonical_context else None,
            "relations": {
                "outgoing": outgoing,
                "incoming": incoming,
            },
            "co_mentions": co_mentions,
        }
    }
    return yaml.safe_dump(card, sort_keys=False, allow_unicode=True).strip()

# --------------------------- matching helpers --------------------------------

def best_match_span(
    target: Span,
    cand_spans: List[Optional[Span]],
    delta_pos: int,
    delta_len: int,
) -> Optional[int]:
    """
    Pick the first candidate span that fuzzy-matches the target; tie-break by highest overlap then earliest start.
    """
    best_idx = None
    best_score = None
    for i, sp in enumerate(cand_spans):
        if sp is None:
            continue
        if not fuzzy_hit(target, sp, delta_pos=delta_pos, delta_len=delta_len):
            continue
        ov = span_overlap(target, sp)
        score = (ov, -abs(sp[0] - target[0]))
        if best_score is None or score > best_score:
            best_score = score
            best_idx = i
    return best_idx

# --------------------------- span matching (extended) ---------------------------

def span_covers(big: Span, small: Span) -> bool:
    """Return True if span `big` fully covers span `small`."""
    return big[0] <= small[0] and small[1] <= big[1]

def best_match_span_fuzzy_then_cover(
    target: Span,
    cand_spans: List[Optional[Span]],
    delta_pos: int,
    delta_len: int,
) -> Tuple[Optional[int], str]:
    """Two-pass span match used for gold->TANL alignment.

    Pass 1: existing fuzzy match (delta_pos/delta_len).
    Pass 2: containment fallback: accept if candidate span fully covers the gold span.

    Returns (best_idx, mode) where mode in {"fuzzy", "contains", "none"}.
    """
    idx = best_match_span(target, cand_spans, delta_pos=delta_pos, delta_len=delta_len)
    if idx is not None:
        return idx, "fuzzy"

    # containment fallback
    best = None  # (extra_len, start_diff, -overlap, idx)
    len_t = target[1] - target[0]
    for i, sp in enumerate(cand_spans):
        if sp is None:
            continue
        if not span_covers(sp, target):
            continue
        extra = (sp[1] - sp[0]) - len_t  # >= 0
        start_diff = abs(sp[0] - target[0])
        ov = span_overlap(target, sp)
        cand = (extra, start_diff, -ov, i)
        if best is None or cand < best:
            best = cand
    if best is None:
        return None, "none"
    return best[-1], "contains"


# --------------------------- main -------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_path", required=True, help="JSONL from TANL relation/type extraction (per paragraph).")
    ap.add_argument("--coref_pred_path", default="", help="Optional JSONL from TANL within-paragraph coref extraction.")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--out_path", default="scico_signatures.jsonl")
    ap.add_argument("--k", type=int, default=-1, help="Only process first K topics; -1 = all")

    # Fuzzy match thresholds
    ap.add_argument("--delta_pos", type=int, default=3)
    ap.add_argument("--delta_len", type=int, default=3)

    # WD-coref safety filter
    ap.add_argument("--coref_jaccard", type=float, default=0.5, help="(Ignored) Kept for backward compatibility; WD-coref safety filter is disabled.")

    # Signature content controls
    ap.add_argument("--signature_format", default="flat", choices=["flat", "yaml"])
    ap.add_argument("--max_rel", type=int, default=6, help="Cap number of incoming/outgoing relations.")
    ap.add_argument("--max_aliases", type=int, default=6, help="Cap number of aliases (excluding canonical).")
    ap.add_argument("--include_comentions", action="store_true", help="Include co-mentions field (can be noisy).")
    ap.add_argument("--max_com", type=int, default=6, help="Cap number of co-mentions (only if include_comentions).")
    ap.add_argument("--include_canonical_context", action="store_true", help="Include canonical mention's sentence context (optional).")
    ap.add_argument(
        "--fallback_context_sents",
        type=int,
        default=10,
        help="If a gold mention has no TANL match (or is forced to fallback), use this many sentences of context (centered).",
    )
    ap.add_argument(
        "--fallback_max_tokens",
        type=int,
        default=0,
        help="Max *word tokens* in the fallback context window (0 = no limit).",
    )
    ap.add_argument(
        "--max_signature_chars",
        type=int,
        default=2000,
        help="Outlier guard: if a built signature exceeds this many characters, rebuild as context-only fallback.",
    )
    ap.add_argument(
        "--max_entity_chars",
        type=int,
        default=250,
        help="Drop TANL entities with mention surface longer than this many characters (guards hallucinations).",
    )
    ap.add_argument(
        "--max_rel_arg_chars",
        type=int,
        default=300,
        help="Drop TANL relations whose argument string exceeds this many characters (guards hallucinations).",
    )
    ap.add_argument("--m_open", default="<m>")
    ap.add_argument("--m_close", default="</m>")

    # Diagnostics
    ap.add_argument("--diag_tokenizer_model", default="", help="If set, compute token-length diagnostics with this HF tokenizer model.")
    ap.add_argument("--diag_max_lengths", default="256,384,512", help="Comma-separated max_length values for truncation-rate diagnostics.")
    ap.add_argument("--write_yaml_debug", action="store_true", help="Also write 'signature_yaml' field for debugging even if format=flat.")

    args = ap.parse_args()

    # Load SciCo split
    ds = load_dataset("allenai/scico")[args.split]
    rows_by_tid: Dict[int, Dict] = {}
    for i in range(len(ds)):
        r = ds[i]
        rows_by_tid[int(r["id"])] = r
    topic_order = list(rows_by_tid.keys())
    topic_order.sort()
    if args.k > 0:
        topic_order = topic_order[: args.k]

    # Load relation/type predictions
    assert os.path.exists(args.pred_path), f"Missing: {args.pred_path}"
    pred_by_para: Dict[Tuple[int, int], Dict] = {}
    with open(args.pred_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pred_by_para[(int(rec["topic_id"]), int(rec["para_idx"]))] = rec

    # Load coref predictions (optional)
    coref_by_para: Dict[Tuple[int, int], Dict] = {}
    if args.coref_pred_path:
        assert os.path.exists(args.coref_pred_path), f"Missing: {args.coref_pred_path}"
        with open(args.coref_pred_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                coref_by_para[(int(rec["topic_id"]), int(rec["para_idx"]))] = rec

    # diagnostics counters
    diag = Counter()
    alias_counts: List[int] = []
    out_counts: List[int] = []
    in_counts: List[int] = []
    sig_lengths_chars: List[int] = []

    # optional tokenizer diagnostics
    tok = None
    diag_maxlens = []
    if args.diag_tokenizer_model:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.diag_tokenizer_model, use_fast=True)
            # Add tags to approximate your eventual training tokenizer setup
            tok.add_special_tokens({"additional_special_tokens": list(DEFAULT_TAGS) + [args.m_open, args.m_close]})
            diag_maxlens = [int(x) for x in args.diag_max_lengths.split(",") if x.strip()]
        except Exception as e:
            print(f"[warn] failed to init tokenizer for diagnostics: {e}")
            tok = None

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_fh = open(args.out_path, "w", encoding="utf-8")

    # If we want token-length diagnostics, collect signatures to tokenize in batches
    diag_sigs_for_tok: List[str] = []

    for tid in tqdm(topic_order, desc="Building signatures"):
        row = rows_by_tid[tid]
        tokens_by_para = row["tokens"]
        doc_ids = row["doc_ids"]
        sent_spans_by_para = row.get("sentences", None)  # list of list of spans
        mentions = row["mentions"]  # list of [pid, s, e, cluster_id]

        # group gold mentions per paragraph (SciCo end is inclusive)
        gold_by_para: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        for (pid, s, e, cid) in mentions:
            gold_by_para[int(pid)].append((int(s), int(e), int(cid)))

        for pid, toks in enumerate(tokens_by_para):
            key = (tid, pid)
            pred_rec = pred_by_para.get(key)
            if pred_rec is None:
                continue

            # sentence spans for paragraph
            if sent_spans_by_para:
                para_sent_spans = [(int(a), int(b)) for (a, b) in sent_spans_by_para[pid]]
            else:
                para_sent_spans = [(0, len(toks))]

            # ---- Parse + align WD-coref occurrences (optional) ----
            occ: List[Dict] = []
            occ_spans: List[Optional[Span]] = []
            occ2cluster: List[int] = []
            clusters: Dict[int, List[int]] = {}
            cluster2canon: Dict[int, int] = {}

            coref_rec = coref_by_para.get(key) if coref_by_para else None
            if coref_rec is not None:
                occ = parse_tanl_coref(coref_rec.get("tanl_output", ""))
                occ_texts = [o["text"] for o in occ]
                occ_spans = align_mentions_to_spans_greedy(toks, occ_texts, max_window=20)
                if occ:
                    occ2cluster, clusters, cluster2canon = build_wd_clusters(
                        occ, occ_spans, jaccard_thr=args.coref_jaccard
                    )

            # ---- Build nodes from WD clusters (soft union base) ----
            nodes: Dict[str, Node] = {}
            occ_idx_to_node: Dict[int, str] = {}

            if occ:
                for cid, members in clusters.items():
                    canon_idx = cluster2canon[cid]
                    canon_text = occ[canon_idx]["text"]
                    canon_span = occ_spans[canon_idx]
                    aliases = set([occ[i]["text"] for i in members if occ[i]["text"]])
                    spans = [sp for i in members for sp in ([occ_spans[i]] if occ_spans[i] is not None else [])]
                    nid = f"W{cid}"
                    node = Node(node_id=nid, canonical_text=canon_text, canonical_span=canon_span,
                                spans=spans, aliases=aliases)
                    nodes[nid] = node
                    for i in members:
                        occ_idx_to_node[i] = nid

            # ---- Parse + align relation/type entities ----
            tanl_entities = parse_tanl_rel(pred_rec.get("tanl_output", ""))
            tanl_entities = filter_tanl_entities(
                tanl_entities,
                max_entity_chars=args.max_entity_chars,
                max_rel_arg_chars=args.max_rel_arg_chars,
            )
            rel_texts = [e["text"] for e in tanl_entities]
            rel_spans = align_mentions_to_spans_greedy(toks, rel_texts, max_window=20)

            rel_idx_to_node: Dict[int, str] = {}

            # helper: match rel span to a WD occurrence (if available)
            for i, ent in enumerate(tanl_entities):
                sp = rel_spans[i]
                nid: Optional[str] = None
                if sp is not None and occ:
                    j = best_match_span(sp, occ_spans, delta_pos=args.delta_pos, delta_len=args.delta_len)
                    if j is not None:
                        nid = occ_idx_to_node.get(j)

                if nid is None:
                    # create rel-only node
                    nid = f"R{i}"
                    canon_text = ent["text"]
                    canon_span = sp
                    node = Node(node_id=nid, canonical_text=canon_text, canonical_span=canon_span,
                                spans=[canon_span] if canon_span is not None else [],
                                aliases=set([canon_text]) if canon_text else set())
                    nodes[nid] = node

                rel_idx_to_node[i] = nid
                nodes[nid].add_type(ent.get("type"))
                nodes[nid].add_outgoing(ent.get("rels_out", []))

            # finalize node types
            for node in nodes.values():
                node.finalize_type()

            # alias map: norm(surface) -> node_id (first wins)
            alias_to_node: Dict[str, str] = {}
            for nid, node in nodes.items():
                # canonical first to bias map toward canonical name
                for s in [node.canonical_text] + sorted(list(node.aliases)):
                    k = norm(s)
                    if not k:
                        continue
                    if k not in alias_to_node:
                        alias_to_node[k] = nid

            # canonicalize outgoing + build incoming edges.
            # Additionally, for symmetric predicates, add reverse edges so both endpoints see the relation.
            raw_edges: List[Tuple[str, str, str]] = []  # (subj_nid, pred_clean, obj_text_raw)
            for nid, node in nodes.items():
                seen = set()
                for pred, obj in node.outgoing_raw:
                    pred_clean = re.sub(r"\s+", " ", (pred or "").strip())
                    obj = re.sub(r"\s+", " ", (obj or "").strip())
                    if not pred_clean or not obj:
                        continue
                    key = (pred_clean, obj)
                    if key in seen:
                        continue
                    seen.add(key)
                    raw_edges.append((nid, pred_clean, obj))

            # Canonicalize edges + generate symmetric reverse edges when possible
            canon_edges: List[Tuple[str, str, Optional[str], str]] = []  # (subj_nid, pred_clean, obj_nid, obj_text)
            for subj_nid, pred_clean, obj_raw in raw_edges:
                obj_nid = alias_to_node.get(norm(obj_raw))
                obj_text = nodes[obj_nid].canonical_text if obj_nid is not None else obj_raw
                canon_edges.append((subj_nid, pred_clean, obj_nid, obj_text))

                # symmetric handling
                if obj_nid is not None and normalize_predicate(pred_clean) in SYMMETRIC_REL_PRED_KEYS:
                    subj_text = nodes[subj_nid].canonical_text
                    canon_edges.append((obj_nid, pred_clean, subj_nid, subj_text))

            # Build outgoing lists
            outgoing_by_node: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
            for subj_nid, pred_clean, _obj_nid, obj_text in canon_edges:
                outgoing_by_node[subj_nid].append((pred_clean, obj_text))

            for nid, node in nodes.items():
                # dedup, sort, cap
                dedup = sorted(set(outgoing_by_node.get(nid, [])), key=lambda x: (x[0], x[1]))
                node.outgoing = dedup[: args.max_rel]

            # Build incoming lists by reversing canonicalized edges that resolve to a node
            incoming_edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # obj_node -> list[(subj_node, pred)]
            for subj_nid, pred_clean, obj_nid, _obj_text in canon_edges:
                if obj_nid is None:
                    continue
                incoming_edges[obj_nid].append((subj_nid, pred_clean))

            for obj_nid, inc_list in incoming_edges.items():
                seen = set()
                out = []
                for subj_nid, pred_clean in inc_list:
                    subj = nodes[subj_nid]
                    key = (subj.node_id, pred_clean)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append((subj.canonical_text, subj.final_type, pred_clean))
                nodes[obj_nid].incoming = sorted(out, key=lambda x: (x[2], x[0]))[: args.max_rel]

            # helper for co-mentions
            def get_comentions_for_sentence(sent_span: Span, target_nid: str) -> List[Tuple[str, str]]:
                ss, se = sent_span
                found = []
                for nid2, node2 in nodes.items():
                    if nid2 == target_nid:
                        continue
                    # any span inside sentence?
                    ok = False
                    for sp2 in node2.spans or ([] if node2.canonical_span is None else [node2.canonical_span]):
                        if sp2 is None:
                            continue
                        if sp2[0] >= ss and sp2[1] <= se:
                            ok = True
                            break
                    if not ok:
                        continue
                    if node2.final_type == "unknown":
                        continue
                    found.append((node2.canonical_text, node2.final_type))
                found = sorted(set(found), key=lambda x: (x[1], x[0]))[: args.max_com]
                return found

            # ---- Build signatures for each gold mention in this paragraph ----
            for (gs, ge_incl, cid) in gold_by_para.get(pid, []):
                gold_span_ro: Span = (int(gs), int(ge_incl) + 1)
                gold_text = detok(toks[gold_span_ro[0] : gold_span_ro[1]])

                # sentence for gold mention
                sent_idx = find_sentence_idx(para_sent_spans, gold_span_ro)
                ss, se = para_sent_spans[sent_idx]
                highlighted_sent = highlight_in_sentence(
                    toks, (ss, se), gold_span_ro, m_open=args.m_open, m_close=args.m_close
                )

                # match gold to a node: compute both hits for analysis, but prefer WD when available
                wd_hit = None
                wd_match_mode = "none"
                if occ:
                    wd_hit = best_match_span(gold_span_ro, occ_spans, delta_pos=args.delta_pos, delta_len=args.delta_len)
                    if wd_hit is not None:
                        wd_match_mode = "fuzzy"
                rel_hit = None
                rel_match_mode = "none"
                if tanl_entities:
                    rel_hit, rel_match_mode = best_match_span_fuzzy_then_cover(
                        gold_span_ro, rel_spans, delta_pos=args.delta_pos, delta_len=args.delta_len
                    )

                target_nid: Optional[str] = None
                matched_kind_raw = "none"
                if wd_hit is not None:
                    target_nid = occ_idx_to_node.get(wd_hit)
                    matched_kind_raw = "wd"
                elif rel_hit is not None:
                    target_nid = rel_idx_to_node.get(rel_hit)
                    matched_kind_raw = "rel"

                # default highlighted context is the single best sentence
                highlighted = highlighted_sent
                used_extended_context = False
                fallback_reason = ""
                matched_kind = matched_kind_raw

                if target_nid is None:
                    # singleton node (not added to nodes dict to avoid affecting other mentions)
                    matched_kind = "none"
                    fallback_reason = "no_match"
                    highlighted = build_highlighted_context_window(
                        toks=toks,
                        sent_spans=para_sent_spans,
                        sent_idx=sent_idx,
                        target_span=gold_span_ro,
                        window_sents=args.fallback_context_sents,
                        max_tokens=args.fallback_max_tokens,
                        m_open=args.m_open,
                        m_close=args.m_close,
                    )
                    used_extended_context = True
                    tmp_node = Node(node_id="G", canonical_text=gold_text, canonical_span=gold_span_ro,
                                    spans=[gold_span_ro], aliases=set([gold_text]))
                    tmp_node.finalize_type()
                    node = tmp_node
                else:
                    node = nodes[target_nid]

                    # If we matched to a node but it carries no useful TANL facts (no type and no relations),
                    # expand the context window to compensate.
                    if node.final_type == "unknown" and (not node.outgoing) and (not node.incoming):
                        highlighted = build_highlighted_context_window(
                            toks=toks,
                            sent_spans=para_sent_spans,
                            sent_idx=sent_idx,
                            target_span=gold_span_ro,
                            window_sents=args.fallback_context_sents,
                            max_tokens=args.fallback_max_tokens,
                            m_open=args.m_open,
                            m_close=args.m_close,
                        )
                        used_extended_context = True
                        if not fallback_reason:
                            fallback_reason = "no_structured_info"

                # aliases list (exclude canonical)
                aliases = sorted(
                    [a for a in node.aliases if norm(a) != norm(node.canonical_text)],
                    key=lambda x: (-len(norm(x)), norm(x)),
                )[: args.max_aliases]

                # outgoing/incoming strings
                outgoing_strs = [f"{pred} -> {obj}" for (pred, obj) in node.outgoing]
                incoming_strs = []
                for subj_text, subj_type, pred in node.incoming:
                    if subj_type and subj_type != "unknown":
                        incoming_strs.append(f"{subj_text}({subj_type}) -> {pred}")
                    else:
                        incoming_strs.append(f"{subj_text} -> {pred}")

                # canonical context (optional)
                can_ctx = None
                if args.include_canonical_context and node.canonical_span is not None:
                    if node.canonical_span != gold_span_ro:
                        can_sent_idx = find_sentence_idx(para_sent_spans, node.canonical_span)
                        css, cse = para_sent_spans[can_sent_idx]
                        can_ctx = highlight_in_sentence(
                            toks, (css, cse), node.canonical_span, m_open=args.m_open, m_close=args.m_close
                        )

                # co-mentions (optional)
                co_mentions_strs: List[str] = []
                co_mentions_yaml: List[Dict] = []
                if args.include_comentions and target_nid is not None:
                    com = get_comentions_for_sentence((ss, se), target_nid)
                    co_mentions_strs = [f"{t}({ty})" for (t, ty) in com]
                    co_mentions_yaml = [{"text": t, "type": ty} for (t, ty) in com]

                # build signature
                if args.signature_format == "flat":
                    signature = build_signature_flat(
                        gold_text=gold_text,
                        highlighted_context=highlighted,
                        canonical_text=node.canonical_text,
                        typ=node.final_type,
                        aliases=aliases,
                        outgoing_strs=outgoing_strs,
                        incoming_strs=incoming_strs,
                        co_mentions_strs=co_mentions_strs,
                        canonical_context=can_ctx,
                        m_open=args.m_open,
                        m_close=args.m_close,
                    )
                else:
                    outgoing_yaml = [{"predicate": pred, "object_text": obj} for (pred, obj) in node.outgoing]
                    incoming_yaml = [
                        {"predicate": pred, "subject_text": subj_text, "subject_type": subj_type}
                        for (subj_text, subj_type, pred) in node.incoming
                    ]
                    signature = build_signature_yaml(
                        gold_text=gold_text,
                        highlighted_context=highlighted,
                        canonical_text=node.canonical_text,
                        typ=node.final_type,
                        aliases=aliases,
                        outgoing=outgoing_yaml,
                        incoming=incoming_yaml,
                        co_mentions=co_mentions_yaml,
                        canonical_context=can_ctx,
                        m_open=args.m_open,
                        m_close=args.m_close,
                    )

                # Outlier guard: if signature is suspiciously long, treat as no-match and fall back to extended context.
                if args.max_signature_chars > 0 and len(signature) > args.max_signature_chars:
                    fallback_reason = "signature_too_long"
                    used_extended_context = True
                    matched_kind = "none"

                    # override structured fields (treat as no-match)
                    aliases = []
                    outgoing_strs = []
                    incoming_strs = []
                    co_mentions_strs = []
                    co_mentions_yaml = []
                    can_ctx = None

                    # Rebuild as context-only signature, but ensure we don't keep generating very long fallbacks.
                    # If the extended context is still too long, shrink its token cap progressively.
                    max_tok = args.fallback_max_tokens
                    for _ in range(6):
                        highlighted = build_highlighted_context_window(
                            toks=toks,
                            sent_spans=para_sent_spans,
                            sent_idx=sent_idx,
                            target_span=gold_span_ro,
                            window_sents=args.fallback_context_sents,
                            max_tokens=max_tok,
                            m_open=args.m_open,
                            m_close=args.m_close,
                        )
                        if args.signature_format == "flat":
                            signature = build_signature_flat(
                                gold_text=gold_text,
                                highlighted_context=highlighted,
                                canonical_text=gold_text,
                                typ="unknown",
                                aliases=[],
                                outgoing_strs=[],
                                incoming_strs=[],
                                co_mentions_strs=[],
                                canonical_context=None,
                                m_open=args.m_open,
                                m_close=args.m_close,
                            )
                        else:
                            signature = build_signature_yaml(
                                gold_text=gold_text,
                                highlighted_context=highlighted,
                                canonical_text=gold_text,
                                typ="unknown",
                                aliases=[],
                                outgoing=[],
                                incoming=[],
                                co_mentions=[],
                                canonical_context=None,
                                m_open=args.m_open,
                                m_close=args.m_close,
                            )
                        if args.max_signature_chars <= 0 or len(signature) <= args.max_signature_chars:
                            break
                        if max_tok <= 32:
                            # last resort: single sentence
                            highlighted = highlighted_sent
                            if args.signature_format == "flat":
                                signature = build_signature_flat(
                                    gold_text=gold_text,
                                    highlighted_context=highlighted,
                                    canonical_text=gold_text,
                                    typ="unknown",
                                    aliases=[],
                                    outgoing_strs=[],
                                    incoming_strs=[],
                                    co_mentions_strs=[],
                                    canonical_context=None,
                                    m_open=args.m_open,
                                    m_close=args.m_close,
                                )
                            else:
                                signature = build_signature_yaml(
                                    gold_text=gold_text,
                                    highlighted_context=highlighted,
                                    canonical_text=gold_text,
                                    typ="unknown",
                                    aliases=[],
                                    outgoing=[],
                                    incoming=[],
                                    co_mentions=[],
                                    canonical_context=None,
                                    m_open=args.m_open,
                                    m_close=args.m_close,
                                )
                            break
                        max_tok = max(32, int(max_tok * 0.7))

                # ---- diagnostics counters (final, after any fallback) ----
                diag["signatures"] += 1
                diag[f"matched_raw_{matched_kind_raw}"] += 1
                diag[f"matched_{matched_kind}"] += 1
                if fallback_reason:
                    diag[f"fallback_{fallback_reason}"] += 1

                # type/canonical stats reflect what will go into the signature
                type_used = "unknown" if matched_kind == "none" else node.final_type
                canon_used = gold_text if matched_kind == "none" else node.canonical_text

                if type_used == "unknown":
                    diag["type_unknown"] += 1
                if norm(canon_used) != norm(gold_text):
                    diag["canonical_diff"] += 1

                alias_counts.append(len(aliases))
                out_counts.append(len(outgoing_strs))
                in_counts.append(len(incoming_strs))

                sig_lengths_chars.append(len(signature))
                if tok is not None:
                    diag_sigs_for_tok.append(signature)

                out_obj = {
                    "topic_id": int(tid),
                    "doc_id": int(doc_ids[pid]),
                    "para_idx": int(pid),
                    "sent_idx": int(sent_idx),
                    "gold_span": [int(gs), int(ge_incl)],  # inclusive end
                    "gold_text": gold_text,
                    "cluster_id": int(cid),
                    # match / fallback metadata (for later error analysis)
                    "match_kind": matched_kind,
                    "match_kind_raw": matched_kind_raw,
                    "matched_wd_span": bool(wd_hit is not None),
                    "matched_rel_span": bool(rel_hit is not None),
                    "rel_span_match_mode": str(rel_match_mode),
                    "wd_span_match_mode": str(wd_match_mode),
                    "target_match_mode": (str(wd_match_mode) if matched_kind_raw == "wd" else (str(rel_match_mode) if matched_kind_raw == "rel" else "none")),
                    "used_extended_context": bool(used_extended_context),
                    "fallback_reason": fallback_reason,
                    # lightweight signature attributes (avoid re-parsing signature text later)
                    "sig_canonical": canon_used,
                    "sig_type": type_used,
                    "sig_n_aliases": int(len(aliases)),
                    "sig_n_out": int(len(outgoing_strs)),
                    "sig_n_in": int(len(incoming_strs)),
                    "signature": signature,
                }

                if args.write_yaml_debug and args.signature_format != "yaml":
                    # also output YAML signature for spot-checking; keep it consistent with the *final* signature
                    if matched_kind == "none":
                        out_obj["signature_yaml"] = build_signature_yaml(
                            gold_text=gold_text,
                            highlighted_context=highlighted,
                            canonical_text=gold_text,
                            typ="unknown",
                            aliases=[],
                            outgoing=[],
                            incoming=[],
                            co_mentions=[],
                            canonical_context=None,
                            m_open=args.m_open,
                            m_close=args.m_close,
                        )
                    else:
                        outgoing_yaml = [{"predicate": pred, "object_text": obj} for (pred, obj) in node.outgoing]
                        incoming_yaml = [
                            {"predicate": pred, "subject_text": subj_text, "subject_type": subj_type}
                            for (subj_text, subj_type, pred) in node.incoming
                        ]
                        out_obj["signature_yaml"] = build_signature_yaml(
                            gold_text=gold_text,
                            highlighted_context=highlighted,
                            canonical_text=node.canonical_text,
                            typ=node.final_type,
                            aliases=aliases,
                            outgoing=outgoing_yaml,
                            incoming=incoming_yaml,
                            co_mentions=co_mentions_yaml,
                            canonical_context=can_ctx,
                            m_open=args.m_open,
                            m_close=args.m_close,
                        )

                out_fh.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    out_fh.close()
    print(f"Wrote signatures to {args.out_path}")

    # -------- diagnostics --------
    n = diag["signatures"]
    if n == 0:
        print("[diag] no signatures written (check inputs).")
        return

    def pct(x: int) -> float:
        return 100.0 * x / max(1, n)

    print("\n[diag] Signature build summary")
    print(f"  signatures: {n:,}")
    print(f"  matched_wd: {diag['matched_wd']:,} ({pct(diag['matched_wd']):.2f}%)")
    print(f"  matched_rel: {diag['matched_rel']:,} ({pct(diag['matched_rel']):.2f}%)")
    print(f"  matched_none: {diag['matched_none']:,} ({pct(diag['matched_none']):.2f}%)")
    # raw match kind (before any outlier fallback)
    if diag["matched_raw_wd"] + diag["matched_raw_rel"] + diag["matched_raw_none"]:
        print(f"  matched_raw_wd: {diag['matched_raw_wd']:,} ({pct(diag['matched_raw_wd']):.2f}%)")
        print(f"  matched_raw_rel: {diag['matched_raw_rel']:,} ({pct(diag['matched_raw_rel']):.2f}%)")
        print(f"  matched_raw_none: {diag['matched_raw_none']:,} ({pct(diag['matched_raw_none']):.2f}%)")
    print(f"  canonical!=surface: {diag['canonical_diff']:,} ({pct(diag['canonical_diff']):.2f}%)")
    print(f"  type_unknown: {diag['type_unknown']:,} ({pct(diag['type_unknown']):.2f}%)")

    # fallback breakdown (empty if feature not used)
    fb_keys = [
        ("no_match", diag.get("fallback_no_match", 0)),
        ("no_structured_info", diag.get("fallback_no_structured_info", 0)),
        ("signature_too_long", diag.get("fallback_signature_too_long", 0)),
    ]
    fb_total = sum(v for _, v in fb_keys)
    if fb_total:
        parts = ", ".join([f"{k}={v}" for k, v in fb_keys if v])
        print(f"  fallbacks: {fb_total:,} ({parts})")

    def quantiles(xs: List[int]) -> Dict[str, float]:
        if not xs:
            return {}
        xs_sorted = sorted(xs)
        def q(p: float) -> float:
            idx = int(round(p * (len(xs_sorted) - 1)))
            return float(xs_sorted[idx])
        return {"p50": q(0.50), "p75": q(0.75), "p90": q(0.90), "p95": q(0.95), "max": float(xs_sorted[-1])}

    print(f"  aliases (count) quantiles: {quantiles(alias_counts)}")
    print(f"  outgoing rels quantiles: {quantiles(out_counts)}")
    print(f"  incoming rels quantiles: {quantiles(in_counts)}")
    if sig_lengths_chars:
        print(f"  signature length (chars) quantiles: {quantiles(sig_lengths_chars)}")

    if tok is not None and diag_sigs_for_tok:
        # tokenize in batches for speed
        lens = []
        bs = 128
        for i in range(0, len(diag_sigs_for_tok), bs):
            batch = diag_sigs_for_tok[i:i+bs]
            enc = tok(batch, add_special_tokens=True, padding=False, truncation=False)
            lens.extend([len(ids) for ids in enc["input_ids"]])
        print(f"  signature length (tokens) quantiles: {quantiles(lens)}")
        for ml in diag_maxlens:
            frac = sum(1 for L in lens if L > ml) / max(1, len(lens))
            print(f"  > max_length {ml}: {frac*100:.2f}%")

if __name__ == "__main__":
    main()
