# train_signature_coref.py
import os
import math
import argparse
import numpy as np
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from models.signature_crossencoder import SignatureCorefCrossEncoder
from models.signature_pair_dataset import SignaturePairDataset, PairCollator, add_special_tokens

def set_seed(seed: int = 13):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def evaluate(model, dl, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            tti = batch.get("token_type_ids")
            if tti is not None:
                tti = tti.to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids, attn, token_type_ids=tti, labels=None)
            logits = out["logits"]
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int32)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"loss": float( -np.mean(labels*np.log(np.clip(probs,1e-8,1)) + (1-labels)*np.log(np.clip(1-probs,1e-8,1))) ),
            "precision": float(p), "recall": float(r), "f1": float(f1), "acc": float(acc)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="validation")
    ap.add_argument("--signatures_path_train", required=True)
    ap.add_argument("--signatures_path_val", required=True)
    ap.add_argument("--bert_model", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--adapter_name", default="", help="Optional HF adapter id, e.g. allenai/specter2")
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--neg_pos_ratio", type=float, default=1.0)
    ap.add_argument("--topics_limit_train", type=int, default=-1)
    ap.add_argument("--topics_limit_val", type=int, default=-1)
    ap.add_argument("--output_dir", default="checkpoints_signature_ce")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SignaturePairDataset(
        split=args.train_split,
        signatures_path=args.signatures_path_train,
        bert_model=args.bert_model,
        max_length=args.max_length,
        neg_pos_ratio=args.neg_pos_ratio,
        topics_limit=None if args.topics_limit_train < 0 else args.topics_limit_train
    )
    val_ds = SignaturePairDataset(
        split=args.val_split,
        signatures_path=args.signatures_path_val,
        bert_model=args.bert_model,
        max_length=args.max_length,
        neg_pos_ratio=1.0,  # balance doesn't matter for val
        topics_limit=None if args.topics_limit_val < 0 else args.topics_limit_val
    )

    collator = PairCollator(train_ds.tokenizer, max_length=args.max_length)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    adapter = args.adapter_name.strip() or None
    model = SignatureCorefCrossEncoder(bert_model=args.bert_model, adapter_name=adapter)
    # IMPORTANT: resize embeddings for added <m>, </m>
    add_special_tokens(train_ds.tokenizer, ("<m>", "</m>"))
    model.bert.resize_token_embeddings(len(train_ds.tokenizer))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dl) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1 = -1.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_dl, 1):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            tti = batch.get("token_type_ids")
            if tti is not None:
                tti = tti.to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids, attn, token_type_ids=tti, labels=labels)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train_dl)}  loss={running/step:.4f}")

        # validation
        metrics = evaluate(model, val_dl, device)
        print(f"[epoch {epoch}] val: {metrics}")
        if 1 or metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            ckpt_path = os.path.join(args.output_dir, f"best_epoch{epoch}_f1{best_f1:.4f}.pt")
            torch.save({"state_dict": model.state_dict(),
                        "tokenizer": train_ds.tokenizer.get_vocab()}, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
