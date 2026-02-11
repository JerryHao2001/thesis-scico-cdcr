# scripts/dump_pair_scores.py
import json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path

from models.signature_crossencoder import SignatureCorefCrossEncoder
from models.signature_pair_dataset import add_special_tokens, load_signatures_jsonl, build_topic_mentions

class TopicPairs(Dataset):
    def __init__(self, signatures):
        self.sigs = signatures
        self.pairs = []
        n = len(signatures)
        for i in range(n):
            for j in range(i+1, n):
                self.pairs.append((i, j))
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return {"i": i, "j": j, "a": self.sigs[i], "b": self.sigs[j]}

def collate(batch, tok, max_len):
    A = [b["a"] for b in batch]
    B = [b["b"] for b in batch]
    enc = tok(A, text_pair=B, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    meta = [(b["i"], b["j"]) for b in batch]
    return enc, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["train","validation","test"])
    ap.add_argument("--signatures_path", required=True)
    ap.add_argument("--bert_model", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--adapter_name", default="", help="Optional HF adapter id (e.g., allenai/specter2)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_path", default="pair_scores_dev.jsonl")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_dataset("allenai/scico")[args.split]
    sig_map = load_signatures_jsonl(args.signatures_path)

    tok = AutoTokenizer.from_pretrained(args.bert_model, use_fast=True)
    add_special_tokens(tok, ("<m>", "</m>"))

    adapter = args.adapter_name.strip() or None
    model = SignatureCorefCrossEncoder(bert_model=args.bert_model, adapter_name=adapter)
    model.bert.resize_token_embeddings(len(tok))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    # out = open(args.out_path, "w", encoding="utf-8")
    out_file = Path(args.out_path)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as out:

        for r in ds:
            tid = int(r["id"])
            mentions = build_topic_mentions(r, sig_map)
            signatures = [m["signature"] for m in mentions]
            n = len(signatures)
            if n <= 1:
                out.write(json.dumps({"topic_id": tid, "n": n, "edges": []}) + "\n")
                continue

            topic_ds = TopicPairs(signatures)
            dl = DataLoader(topic_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate(b, tok, args.max_length))

            edges = []
            with torch.no_grad():
                for enc, meta in dl:
                    input_ids = enc["input_ids"].to(device)
                    attn = enc["attention_mask"].to(device)
                    tti = enc.get("token_type_ids")
                    if tti is not None: tti = tti.to(device)
                    out_logits = model(input_ids, attn, token_type_ids=tti)["logits"].cpu().numpy()
                    for (i, j), z in zip(meta, out_logits.tolist()):
                        edges.append({"i": i, "j": j, "logit": float(z)})
            out.write(json.dumps({"topic_id": tid, "n": n, "edges": edges}) + "\n")

    print(f"Wrote pair scores to {args.out_path}")

if __name__ == "__main__":
    main()
