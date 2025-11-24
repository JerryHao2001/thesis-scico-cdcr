# predict_signature_coref.py
import os
import json
import math
import argparse
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from datasets import load_dataset
from sklearn.cluster import AgglomerativeClustering

from models.signature_crossencoder import SignatureCorefCrossEncoder
from models.signature_pair_dataset import add_special_tokens, load_signatures_jsonl, build_topic_mentions

class TopicPairScoringDataset(Dataset):
    """
    For a given topic: produce all unordered pairs (i<j) to score.
    """
    def __init__(self, signatures: List[str], tokenizer, max_length=384):
        super().__init__()
        self.sigs = signatures
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = []
        n = len(self.sigs)
        for i in range(n):
            for j in range(i+1, n):
                self.pairs.append((i,j))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i,j = self.pairs[idx]
        return {"i": i, "j": j, "a": self.sigs[i], "b": self.sigs[j]}

def collate_pairs(batch, tokenizer, max_length):
    texts_a = [b["a"] for b in batch]
    texts_b = [b["b"] for b in batch]
    enc = tokenizer(
        texts_a, text_pair=texts_b,
        padding=True, truncation=True, max_length=max_length,
        return_tensors="pt"
    )
    meta = [(b["i"], b["j"]) for b in batch]
    return enc, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--signatures_path", required=True)
    ap.add_argument("--bert_model", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--checkpoint", required=True, help="path to .pt checkpoint from training")
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--distance_threshold", type=float, default=0.5,
                    help="clustering threshold on distance=1-sigmoid(logit)")
    ap.add_argument("--out_path", default="predicted_clusters.jsonl")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and signatures
    ds = load_dataset("allenai/scico")[args.split]
    sig_map = load_signatures_jsonl(args.signatures_path)

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=True)
    add_special_tokens(tokenizer, ("<m>", "</m>"))

    model = SignatureCorefCrossEncoder(bert_model=args.bert_model)
    model.bert.resize_token_embeddings(len(tokenizer))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    out_fh = open(args.out_path, "w", encoding="utf-8")

    for i in range(len(ds)):
        topic = ds[i]
        tid = int(topic["id"])
        mentions = build_topic_mentions(topic, sig_map)
        signatures = [m["signature"] for m in mentions]

        if len(signatures) <= 1:
            # trivial clustering
            clusters = list(range(len(signatures)))
            out_fh.write(json.dumps({"topic_id": tid, "clusters": clusters}) + "\n")
            continue

        # score all unordered pairs
        topic_ds = TopicPairScoringDataset(signatures, tokenizer, max_length=args.max_length)
        dl = DataLoader(topic_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate_pairs(b, tokenizer, args.max_length))

        n = len(signatures)
        scores = np.zeros((n, n), dtype=np.float32)
        with torch.no_grad():
            for enc, meta in dl:
                input_ids = enc["input_ids"].to(device)
                attn = enc["attention_mask"].to(device)
                tti = enc.get("token_type_ids")
                if tti is not None:
                    tti = tti.to(device)
                out = model(input_ids, attn, token_type_ids=tti)
                logits = out["logits"].cpu().numpy()
                probs = 1.0 / (1.0 + np.exp(-logits))
                for (i_idx, j_idx), p in zip(meta, probs.tolist()):
                    scores[i_idx, j_idx] = p
                    scores[j_idx, i_idx] = p

        # cluster with agglomerative on distance = 1 - prob
        dist = 1.0 - scores
        # enforce zero diagonal
        for k in range(n): dist[k, k] = 0.0
        clusterer = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            distance_threshold=args.distance_threshold,
            n_clusters=None
        )
        labels = clusterer.fit_predict(dist)

        out_fh.write(json.dumps({
            "topic_id": tid,
            "clusters": labels.tolist()
        }) + "\n")

    out_fh.close()
    print(f"Wrote predicted clusters to {args.out_path}")

if __name__ == "__main__":
    main()
