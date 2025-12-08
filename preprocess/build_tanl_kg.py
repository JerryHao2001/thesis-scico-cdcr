#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Any

def norm(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize(text: str) -> List[str]:
    # Simple whitespace tokenizer; spans are w.r.t this tokenization
    return [t for t in text.split() if t]

# ------------------ Entity TANL parsing with spans ------------------ #

BRACKET_RE = re.compile(
    r"\[\s*(?P<mention>.+?)\s*\|\s*(?P<etype>[^|\]]+?)\s*(?:\|\s*(?P<rels>.+?))?\s*\]"
)
REL_PAIR_RE = re.compile(r"(?P<rname>[a-zA-Z_]+)\s*=\s*(?P<arg>[^;|]+)")

def parse_entities_with_spans(tanl_text: str) -> List[Dict[str, Any]]:
    """
    Parse TANL entity+relation output into entities with token spans
    relative to the text obtained by replacing each [...] with the mention surface.
    """
    entities = []
    token_idx = 0
    pos = 0
    for m in BRACKET_RE.finditer(tanl_text or ""):
        # Tokens before this bracket
        before = tanl_text[pos:m.start()]
        token_idx += len(tokenize(before))

        # Inside the bracket
        inside_mention = m.group("mention") or ""
        mention_str = inside_mention.split("|")[0].strip()
        etype = (m.group("etype") or "").strip()
        rels_str = m.group("rels") or ""
        rels = []
        for r in REL_PAIR_RE.finditer(rels_str):
            pred = r.group("rname").strip()
            obj = r.group("arg").strip()
            rels.append((pred, obj))

        # Tokens for the mention surface
        mention_tokens = tokenize(mention_str)
        if mention_tokens:
            start = token_idx
            end = start + len(mention_tokens) - 1
            token_idx = end + 1
        else:
            start = end = None

        entities.append(
            {
                "text": mention_str,
                "type": etype if etype else None,
                "rels_out": rels,
                "span": (start, end),
            }
        )
        pos = m.end()

    # Trailing text doesn't affect any entity span
    return entities

# ------------------ Coref TANL parsing with spans ------------------ #

COREF_BRACKET_RE = re.compile(r"\[(.+?)\]")

def parse_coref_with_spans(tanl_text: str) -> List[Dict[str, Any]]:
    """
    Parse TANL coref output of the form [A] or [B | A] into mentions
    with spans in the same tokenization scheme as above.
    """
    coref_mentions = []
    token_idx = 0
    pos = 0
    for m in COREF_BRACKET_RE.finditer(tanl_text or ""):
        before = tanl_text[pos:m.start()]
        token_idx += len(tokenize(before))

        inside = m.group(1).strip()
        parts = [p.strip() for p in inside.split("|")]
        if len(parts) == 1:
            surface = parts[0]
            canonical = parts[0]
            is_coref = False
        else:
            surface, canonical = parts[0], parts[1]
            is_coref = True

        mention_tokens = tokenize(surface)
        if mention_tokens:
            start = token_idx
            end = start + len(mention_tokens) - 1
            token_idx = end + 1
        else:
            start = end = None

        coref_mentions.append(
            {
                "surface": surface,
                "canonical": canonical,
                "is_coref": is_coref,
                "span": (start, end),
            }
        )
        pos = m.end()
    return coref_mentions

# ------------------ KG construction ------------------ #

def get_doc_struct(
    docs: Dict[Tuple[int, int], Dict[str, Any]],
    topic_id: int,
    doc_id: int,
) -> Dict[str, Any]:
    key = (topic_id, doc_id)
    if key not in docs:
        docs[key] = {
            "topic_id": topic_id,
            "doc_id": doc_id,
            "mentions": [],     # list of mention dicts (entity + coref)
            "clusters": {},     # cluster_key -> cluster dict
        }
    return docs[key]

def get_cluster(
    doc: Dict[str, Any],
    cluster_key: str,
    canonical_str: str,
) -> Dict[str, Any]:
    clusters = doc["clusters"]
    if cluster_key not in clusters:
        clusters[cluster_key] = {
            "key": cluster_key,         # normalized key
            "canonical": canonical_str, # first-seen canonical form
            "names": set(),             # set of surface forms
            "types": set(),             # aggregated entity types
            "mention_ids": [],          # ids into doc["mentions"]
            "rel_out": [],              # cluster-level outgoing edges
            "rel_in": [],               # cluster-level incoming edges
        }
    return clusters[cluster_key]

def build_kg(
    ent_path: str,
    coref_path: str,
    out_path: str,
) -> None:
    docs: Dict[Tuple[int, int], Dict[str, Any]] = {}
    # For measuring how often canonical A from [B|A] isn't seen as an entity name
    entity_names_by_doc: Dict[Tuple[int, int], set] = defaultdict(set)

    # ------------------ Pass 1: entity+relation TANL ------------------ #
    with open(ent_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            topic_id = int(rec["topic_id"])
            doc_id = int(rec.get("doc_id", 0))
            para_idx = int(rec.get("para_idx", 0))
            tanl_text = rec.get("tanl_output", "") or ""

            doc = get_doc_struct(docs, topic_id, doc_id)
            entities = parse_entities_with_spans(tanl_text)

            # First add entity mentions and build clusters
            for ent in entities:
                surface = ent["text"]
                etype = ent["type"]
                rels_out = ent.get("rels_out", [])
                span = ent.get("span", (None, None))
                start, end = span

                m_id = len(doc["mentions"])
                canonical = surface  # for entity mentions, canonical = surface
                cluster_key = norm(canonical)

                # Track entity names for later canonical-A stats
                entity_names_by_doc[(topic_id, doc_id)].add(cluster_key)

                cluster = get_cluster(doc, cluster_key, canonical)
                cluster["names"].add(surface)
                if etype:
                    cluster["types"].add(etype)

                mention = {
                    "id": m_id,
                    "surface": surface,
                    "canonical": canonical,
                    "source": "entity",
                    "topic_id": topic_id,
                    "para_idx": para_idx,
                    "span": [start, end],
                    "type": etype,
                    "is_coref": False,
                }
                doc["mentions"].append(mention)
                cluster["mention_ids"].append(m_id)

            # Then lift relations to cluster level
            for ent in entities:
                surface = ent["text"]
                rels_out = ent.get("rels_out", [])
                subj_key = norm(surface)
                subj_cluster = get_cluster(doc, subj_key, surface)
                for (pred, obj_text) in rels_out:
                    obj_key = norm(obj_text)
                    obj_cluster = get_cluster(doc, obj_key, obj_text)
                    subj_cluster["rel_out"].append(
                        {"pred": pred, "obj": obj_key, "obj_text": obj_text}
                    )
                    obj_cluster["rel_in"].append(
                        {"pred": pred, "subj": subj_key, "subj_text": surface}
                    )

    # ------------------ Pass 2: coref TANL ------------------ #

    total_coref_links = 0
    missing_canonical = 0

    with open(coref_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            topic_id = int(rec["topic_id"])
            doc_id = int(rec.get("doc_id", 0))
            para_idx = int(rec.get("para_idx", 0))
            tanl_text = rec.get("tanl_output", "") or ""

            doc = get_doc_struct(docs, topic_id, doc_id)
            coref_mentions = parse_coref_with_spans(tanl_text)

            for cm in coref_mentions:
                surface = cm["surface"]
                canonical = cm["canonical"]
                is_coref = cm["is_coref"]
                start, end = cm["span"]

                m_id = len(doc["mentions"])
                cluster_key = norm(canonical)
                cluster = get_cluster(doc, cluster_key, canonical)
                cluster["names"].add(surface)

                mention = {
                    "id": m_id,
                    "surface": surface,
                    "canonical": canonical,
                    "source": "coref",
                    "topic_id": topic_id,
                    "para_idx": para_idx,
                    "span": [start, end],
                    "type": None,
                    "is_coref": is_coref,
                }
                doc["mentions"].append(mention)
                cluster["mention_ids"].append(m_id)

                # stats: [B|A] where A doesn't match any entity by name
                if is_coref:
                    total_coref_links += 1
                    doc_key = (topic_id, doc_id)
                    if cluster_key not in entity_names_by_doc.get(doc_key, set()):
                        missing_canonical += 1

    # ------------------ Write output ------------------ #

    with open(out_path, "w") as out_f:
        for (topic_id, doc_id), doc in docs.items():
            clusters_out = []
            for cl_key, cl in doc["clusters"].items():
                clusters_out.append(
                    {
                        "key": cl_key,
                        "canonical": cl["canonical"],
                        "names": sorted(cl["names"]),
                        "types": sorted(cl["types"]),
                        "mention_ids": cl["mention_ids"],
                        "rel_out": cl["rel_out"],
                        "rel_in": cl["rel_in"],
                    }
                )
            doc_out = {
                "topic_id": topic_id,
                "doc_id": doc_id,
                "clusters": clusters_out,
                "mentions": doc["mentions"],
            }
            out_f.write(json.dumps(doc_out) + "\n")

    print(f"Total coref links [B|A] seen: {total_coref_links}")
    print(
        "Coref links whose canonical A not found among entity mentions "
        f"(by pure norm(name) match): {missing_canonical}"
    )
    if total_coref_links > 0:
        frac = missing_canonical / total_coref_links
        print(f"Fraction missing: {frac:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity_path", type=str, required=True)
    ap.add_argument("--coref_path", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    args = ap.parse_args()
    build_kg(args.entity_path, args.coref_path, args.out_path)

if __name__ == "__main__":
    main()
