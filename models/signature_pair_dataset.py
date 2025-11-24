# models/signature_pair_dataset.py
import json
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

def add_special_tokens(tokenizer, specials=("<m>", "</m>")):
    add = {"additional_special_tokens": list(specials)}
    tokenizer.add_special_tokens(add)
    return tokenizer

def load_signatures_jsonl(path: str) -> Dict[Tuple[int,int,Tuple[int,int]], Dict]:
    """
    Returns a mapping (topic_id, para_idx, (s,e)) -> record
    record has keys: topic_id, doc_id, para_idx, sent_idx, gold_span, gold_text, cluster_id, signature
    """
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (int(r["topic_id"]), int(r["para_idx"]), tuple(r["gold_span"]))
            out[key] = r
    return out

def build_topic_mentions(ds_topic, sig_map):
    """
    For a single SciCO topic row, return:
      - mentions list (order same as dataset['mentions'])
        each item: {span:(s,e), para_idx, cluster_id, signature:str}
    """
    mentions = []
    for (pid, s, e, cid) in ds_topic["mentions"]:
        s = int(s); e = int(e)
        
        # e = min(e + 1, len(ds_topic["tokens"][pid]))
        key = (int(ds_topic["id"]), int(pid), (s, e))
        rec = sig_map.get(key)
        if rec is None:
            raise KeyError(f"Missing signature for topic={ds_topic['id']} para={pid} span={(s,e)}")
        mentions.append({
            "span": (s, e),
            "para_idx": int(pid),
            "cluster_id": int(rec["cluster_id"]),
            "signature": rec["signature"],
        })
    return mentions

def pairs_from_topic(mentions, neg_pos_ratio=1.0, seed=13) -> List[Tuple[int,int,int]]:
    """
    Produce pair indices and labels for training from a topic's mentions.
      - positives: all combinations within a cluster (i<j)
      - negatives: sampled to reach neg_pos_ratio * num_positives
    Returns list of (i, j, label) with i<j
    """
    random.seed(seed)
    # group by cluster
    by_c = {}
    for idx, m in enumerate(mentions):
        by_c.setdefault(m["cluster_id"], []).append(idx)
    pos = []
    for c, idxs in by_c.items():
        if len(idxs) < 2: 
            continue
        idxs = sorted(idxs)
        for a in range(len(idxs)):
            for b in range(a+1, len(idxs)):
                pos.append((idxs[a], idxs[b], 1))
    n_pos = len(pos)
    # negatives: sample from different clusters
    all_idxs = list(range(len(mentions)))
    neg = []
    if n_pos == 0:
        # fallback: sample some negatives anyway
        candidates = []
        for a in range(len(all_idxs)):
            for b in range(a+1, len(all_idxs)):
                if mentions[a]["cluster_id"] != mentions[b]["cluster_id"]:
                    candidates.append((a,b,0))
        neg = random.sample(candidates, min(32, len(candidates)))
    else:
        target_neg = int(neg_pos_ratio * n_pos)
        candidates = []
        for a in range(len(all_idxs)):
            for b in range(a+1, len(all_idxs)):
                if mentions[a]["cluster_id"] != mentions[b]["cluster_id"]:
                    candidates.append((a,b,0))
        random.shuffle(candidates)
        neg = candidates[:target_neg]
    return pos + neg

class SignaturePairDataset(Dataset):
    """
    Training dataset that yields pair (sig_i, sig_j, label)
    """
    def __init__(self,
                 split: str,
                 signatures_path: str,
                 bert_model: str = "allenai/scibert_scivocab_uncased",
                 max_length: int = 384,
                 neg_pos_ratio: float = 1.0,
                 topics_limit: Optional[int] = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model, use_fast=True)
        add_special_tokens(self.tokenizer, ("<m>", "</m>"))
        self.max_length = max_length

        ds = load_dataset("allenai/scico")[split]
        sig_map = load_signatures_jsonl(signatures_path)

        # build pairs across topics
        self.examples = []
        for i in range(len(ds) if topics_limit is None else min(topics_limit, len(ds))):
            topic = ds[i]
            mentions = build_topic_mentions(topic, sig_map)
            pairs = pairs_from_topic(mentions, neg_pos_ratio=neg_pos_ratio)
            for a, b, y in pairs:
                self.examples.append({
                    "text_a": mentions[a]["signature"],
                    "text_b": mentions[b]["signature"],
                    "label": y
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex

@dataclass
class PairCollator:
    tokenizer: AutoTokenizer
    max_length: int = 384

    def __call__(self, batch: List[Dict]):
        texts_a = [b["text_a"] for b in batch]
        texts_b = [b["text_b"] for b in batch]
        enc = self.tokenizer(
            texts_a,
            text_pair=texts_b,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc.get("token_type_ids", None),
            "labels": labels
        }
