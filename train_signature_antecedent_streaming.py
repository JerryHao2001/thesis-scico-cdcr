#!/usr/bin/env python3
"""
train_signature_antecedent_streaming.py

Antecedent / mention-ranking training for SciCo signature coreference, with
streaming loss computation to avoid GPU OOM.

Why this file exists:
- A naive implementation that computes and stores logits for all (i,j) pairs in a
  topic while `with_grad=True` will retain thousands of autograd graphs and OOM,
  even when pair_batch_size=1.
- This trainer computes loss *per mention* and backprops immediately, so memory
  is bounded by O(K) candidate pairs (K=cand_max_candidates), not O(n^2).

Core idea:
For each mention i, score candidates j<i plus a dummy antecedent ε ("new entity").
Train a locally-normalized softmax. If multiple gold antecedents exist, use a
marginal likelihood over the set.

Decoding (for evaluation):
Greedy antecedent linking + union-find (connected components of the chosen links).

This script reuses:
- models.signature_crossencoder.SignatureCorefCrossEncoder  (pair scorer)
- models.signature_pair_dataset.load_signatures_jsonl, build_topic_mentions, add_special_tokens
- signature_topic_dataset.SignatureTopicDataset (topic-level dataset)
"""

import os
import json
import math
import argparse
import importlib.util
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# optional progress bar
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x=None, **kwargs):
        return x

from models.signature_crossencoder import SignatureCorefCrossEncoder
from models.signature_pair_dataset import add_special_tokens
from models.signature_topic_dataset import SignatureTopicDataset, TopicCollator


# -------------------- misc utils --------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dynamic_import(path: str, module_name: str = "eval_sigcoref"):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# -------------------- union-find --------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

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

    def labels(self) -> List[int]:
        # map roots to compact ids
        root_to_id: Dict[int, int] = {}
        out = []
        for i in range(len(self.parent)):
            r = self.find(i)
            if r not in root_to_id:
                root_to_id[r] = len(root_to_id)
            out.append(root_to_id[r])
        return out


# -------------------- candidate building --------------------

def build_candidates_train(
    cluster_ids: List[int],
    i: int,
    strategy: str,
    window: int,
    max_candidates: int,
    rng: np.random.Generator,
) -> List[int]:
    """
    Candidate indices for mention i during training.
    Always includes all gold antecedents j<i in the same gold cluster.

    strategy:
      - all:     all j<i
      - window:  only last W mentions (plus any gold antecedents outside window)
      - hybrid:  last W mentions + random earlier mentions to fill up to K (plus gold)
      - random:  random subset of size K from j<i (plus gold)
    """
    assert i >= 0
    if i == 0:
        return []

    gold = [j for j in range(i) if cluster_ids[j] == cluster_ids[i]]
    gold_set = set(gold)

    if strategy == "all":
        base = list(range(i))
    elif strategy == "window":
        start = max(0, i - window)
        base = list(range(start, i))
    elif strategy == "hybrid":
        start = max(0, i - window)
        base = list(range(start, i))
        # add random earlier mentions (not in base) to fill later
    elif strategy == "random":
        base = []
    else:
        raise ValueError(f"Unknown cand_strategy={strategy}")

    base_set = set(base)

    # if max_candidates==0 => no cap
    if max_candidates == 0:
        # union(base, all gold, plus for 'random' just all j<i)
        if strategy == "random":
            return list(range(i))
        return sorted(base_set.union(gold_set))

    # Ensure gold always included. If gold > K, allow over-cap rather than drop gold.
    K = max_candidates
    cand: List[int] = []

    if strategy == "random":
        # sample K from all j<i, but ensure gold included
        pool = list(range(i))
        rng.shuffle(pool)
        # start with gold
        cand = list(gold)
        # fill from pool skipping gold
        for j in pool:
            if j in gold_set:
                continue
            cand.append(j)
            if len(cand) >= max(K, len(gold)):
                break
        return sorted(set(cand))

    # start from base
    cand = list(base)
    # union with gold
    for j in gold:
        if j not in base_set:
            cand.append(j)

    cand = list(dict.fromkeys(cand))  # keep order, unique

    if len(cand) <= max(K, len(gold)):
        # may need to fill for hybrid
        if strategy == "hybrid" and len(cand) < K:
            earlier = [j for j in range(0, max(0, i - window)) if j not in set(cand)]
            rng.shuffle(earlier)
            need = K - len(cand)
            cand.extend(earlier[:need])
        return sorted(set(cand))

    # too many candidates: keep all gold and subsample non-gold
    non_gold = [j for j in cand if j not in gold_set]
    rng.shuffle(non_gold)
    keep_non_gold = non_gold[: max(0, K - len(gold))]
    out = sorted(set(gold + keep_non_gold))
    return out


def build_candidates_infer(
    i: int,
    strategy: str,
    window: int,
    max_candidates: int,
    rng: np.random.Generator,
) -> List[int]:
    """
    Candidate indices for mention i during inference/validation decoding.
    No gold access, so we only use positional heuristics.
    """
    if i == 0:
        return []

    if strategy == "all":
        cand = list(range(i))
        return cand if max_candidates == 0 else cand[-max_candidates:]
    if strategy == "window":
        start = max(0, i - window)
        cand = list(range(start, i))
        return cand if max_candidates == 0 else cand[-max_candidates:]
    if strategy == "hybrid":
        start = max(0, i - window)
        cand = list(range(start, i))
        if max_candidates == 0:
            return cand + list(range(0, start))  # effectively all, but window first
        if len(cand) >= max_candidates:
            return cand[-max_candidates:]
        earlier = list(range(0, start))
        rng.shuffle(earlier)
        need = max_candidates - len(cand)
        cand = cand + earlier[:need]
        return sorted(set(cand))
    if strategy == "random":
        if max_candidates == 0:
            return list(range(i))
        pool = list(range(i))
        rng.shuffle(pool)
        return sorted(pool[:max_candidates])
    raise ValueError(f"Unknown cand_strategy={strategy}")


# -------------------- scoring helpers --------------------

def score_pairs(
    model: SignatureCorefCrossEncoder,
    tokenizer: AutoTokenizer,
    sig_i: str,
    sig_js: List[str],
    device: torch.device,
    max_length: int,
    pair_batch_size: int,
    amp: bool,
) -> torch.Tensor:
    """
    Score pairs (i, j) for fixed i and a list of candidate signatures sig_js.

    Returns logits tensor [len(sig_js)] on device.
    Uses micro-batching over candidates to bound memory.
    """
    if len(sig_js) == 0:
        return torch.empty((0,), dtype=torch.float32, device=device)

    logits_chunks = []
    for s in range(0, len(sig_js), pair_batch_size):
        chunk = sig_js[s:s+pair_batch_size]
        A = [sig_i] * len(chunk)
        enc = tokenizer(
            A,
            text_pair=chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        tti = enc.get("token_type_ids")
        if tti is not None:
            tti = tti.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=amp):
            out = model(input_ids=input_ids, attention_mask=attn, token_type_ids=tti, labels=None)
            z = out["logits"].view(-1)  # [bs]
        logits_chunks.append(z)
    return torch.cat(logits_chunks, dim=0)


# -------------------- loss + decode --------------------

def marginal_nll_for_mention(
    cand_logits: torch.Tensor,        # [m]
    cand_ids: List[int],              # len m
    gold_ants: List[int],             # subset of cand_ids
    eps_logit: torch.Tensor,          # scalar tensor on device
) -> torch.Tensor:
    """
    Compute -log p(gold antecedent) where p is softmax over [eps] + candidates.
    If multiple gold antecedents, use marginal sum over those.
    If no gold antecedent, the correct label is eps.
    """
    # logits over actions: 0 = eps, 1..m = candidates
    all_logits = torch.cat([eps_logit.view(1), cand_logits], dim=0)  # [1+m]
    log_probs = torch.log_softmax(all_logits, dim=0)

    if len(gold_ants) == 0:
        return -log_probs[0]

    # map gold antecedent ids -> positions in candidates
    pos = []
    idx_map = {cid: k for k, cid in enumerate(cand_ids)}  # cand index 0..m-1
    for g in gold_ants:
        if g in idx_map:
            pos.append(1 + idx_map[g])  # shift by 1 due to eps
    if len(pos) == 0:
        # should not happen if we ensured gold included, but be safe:
        return -log_probs[0]
    # log sum exp over gold positions in log-prob space:
    return -(torch.logsumexp(log_probs[torch.tensor(pos, device=log_probs.device)], dim=0))


def decode_topic_greedy(
    model: SignatureCorefCrossEncoder,
    tokenizer: AutoTokenizer,
    signatures: List[str],
    cand_strategy: str,
    cand_window: int,
    cand_max_candidates: int,
    eps_logit: torch.Tensor,
    device: torch.device,
    max_length: int,
    pair_batch_size: int,
    amp: bool,
    seed: int,
) -> List[int]:
    """
    Greedy antecedent decoding:
      for each mention i, pick argmax among eps and candidates.
      if eps: new cluster; else union(i, best_j).
    """
    uf = UnionFind(len(signatures))
    rng = np.random.default_rng(seed)

    for i in range(len(signatures)):
        cand_ids = build_candidates_infer(i, cand_strategy, cand_window, cand_max_candidates, rng)
        cand_sigs = [signatures[j] for j in cand_ids]
        with torch.no_grad():
            cand_logits = score_pairs(model, tokenizer, signatures[i], cand_sigs, device, max_length, pair_batch_size, amp)
            all_logits = torch.cat([eps_logit.view(1), cand_logits], dim=0)
            best = int(torch.argmax(all_logits).item())
        if best == 0:
            continue
        j = cand_ids[best - 1]
        uf.union(i, j)
    return uf.labels()


# -------------------- evaluation formatting --------------------

def build_system_from_pred_labels(split: str, labels_by_tid: Dict[int, List[int]]) -> List[Dict[str, Any]]:
    """
    Construct 'system' list in the structure expected by evaluate_signature_coref.get_coref_scores.
    Mirrors sweep_thresholds.make_system_from_labels. fileciteturn5file13
    """
    from datasets import load_dataset
    ds = load_dataset("allenai/scico")[split]
    by_id = {int(r["id"]): r for r in ds}
    system = []
    for tid, row in by_id.items():
        tid = int(tid)
        if tid not in labels_by_tid:
            continue
        labels = labels_by_tid[tid]
        mentions = row["mentions"]
        if len(labels) != len(mentions):
            raise ValueError(f"Topic {tid}: labels({len(labels)}) != mentions({len(mentions)})")
        sys_mentions = []
        for i, (pid, s, e, _gold) in enumerate(mentions):
            sys_mentions.append([int(pid), int(s), int(e), int(labels[i])])
        system.append({
            "id": tid,
            "tokens": row["tokens"],
            "doc_ids": row.get("doc_ids", []),
            "relations": row.get("relations", []),
            "mentions": sys_mentions,
        })
    return system


def build_gold_topics(split: str) -> List[Dict[str, Any]]:
    """
    Gold topics in evaluator format. Matches sweep_thresholds.build_gold_topics. fileciteturn5file10
    """
    from datasets import load_dataset
    ds = load_dataset("allenai/scico")[split]
    gold = []
    for r in ds:
        rec = {
            "id": int(r["id"]),
            "tokens": r["tokens"],
            "doc_ids": r.get("doc_ids", []),
            "relations": r.get("relations", []),
            "mentions": [],
        }
        for pid, s, e, cid in r["mentions"]:
            rec["mentions"].append([int(pid), int(s), int(e), int(cid)])
        gold.append(rec)
    return gold


# -------------------- main train loop --------------------

def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--train_split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--val_split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--signatures_path_train", required=True)
    ap.add_argument("--signatures_path_val", required=True)
    ap.add_argument("--topics_limit_train", type=int, default=-1,
                    help="If >=0, limit number of topics for quick debugging.")
    ap.add_argument("--topics_limit_val", type=int, default=-1)

    # model/tokenization
    ap.add_argument("--bert_model", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--max_length", type=int, default=256)

    # training
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=25)

    # topic and pair batching
    ap.add_argument("--topic_batch_size", type=int, default=1)
    ap.add_argument("--pair_batch_size", type=int, default=8,
                    help="Micro-batch size for scoring candidates for one mention.")

    # candidate pruning
    ap.add_argument("--cand_strategy", default="hybrid", choices=["all", "window", "hybrid", "random"])
    ap.add_argument("--cand_window", type=int, default=12)
    ap.add_argument("--cand_max_candidates", type=int, default=12,
                    help="Max candidates per mention (excluding ε). 0 means no cap (may be O(n^2)).")

    # epsilon / new-entity scoring
    ap.add_argument("--eps_init", type=float, default=0.0)
    ap.add_argument("--train_eps", action="store_true")

    # memory / speed
    ap.add_argument("--amp", action="store_true",
                    help="Use torch.cuda.amp autocast + GradScaler (recommended on CUDA).")
    ap.add_argument("--grad_checkpointing", action="store_true",
                    help="Enable transformer gradient checkpointing (saves memory, slower).")
    ap.add_argument("--freeze_bert", action="store_true",
                    help="Freeze BERT encoder weights (large memory win, but less adaptable).")

    # evaluation / output
    ap.add_argument("--eval_module_path", default="",
                    help="Optional path to evaluate_signature_coref.py; if set, compute CoNLL on val each epoch.")
    ap.add_argument("--output_dir", default="ckpts/checkpoints_signature_antecedent")
    ap.add_argument("--save_every_epoch", action="store_true")
    ap.add_argument("--eval_every_epoch", action="store_true",
                    help="If set, run decode+eval each epoch (may be slow).")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.output_dir)

    # dataset
    train_ds = SignatureTopicDataset(
        split=args.train_split,
        signatures_path=args.signatures_path_train,
        bert_model=args.bert_model,
        topics_limit=None if args.topics_limit_train < 0 else args.topics_limit_train,
        seed=args.seed,
    )
    val_ds = SignatureTopicDataset(
        split=args.val_split,
        signatures_path=args.signatures_path_val,
        bert_model=args.bert_model,
        topics_limit=None if args.topics_limit_val < 0 else args.topics_limit_val,
        seed=args.seed,
    )

    print(f"[data] train topics={len(train_ds)}  val topics={len(val_ds)}")
    n_train_topics = len(train_ds)
    if len(train_ds) == 0:
        raise RuntimeError(
            "Training dataset has 0 topics. Check train_split and topics_limit_train."
        )

    collator = TopicCollator()
    train_dl = DataLoader(train_ds, batch_size=args.topic_batch_size, shuffle=True, collate_fn=collator)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collator)

    # tokenizer (use dataset's tokenizer which already has <m>, </m> added)
    tok = train_ds.tokenizer

    # model
    model = SignatureCorefCrossEncoder(bert_model=args.bert_model)
    # ensure token embeddings cover <m>, </m>
    add_special_tokens(tok, ("<m>", "</m>"))
    model.bert.resize_token_embeddings(len(tok))

    if args.grad_checkpointing:
        # available on most HF BERT models
        model.bert.gradient_checkpointing_enable()
        model.bert.config.use_cache = False

    if args.freeze_bert:
        for p in model.bert.parameters():
            p.requires_grad = False

    model.to(device)

    # ε logit (dummy antecedent)
    if args.train_eps:
        eps_logit = torch.nn.Parameter(torch.tensor(float(args.eps_init), device=device))
        params = list(model.parameters()) + [eps_logit]
    else:
        eps_logit = torch.tensor(float(args.eps_init), device=device)
        params = list(model.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_dl) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    eval_mod = None
    get_coref_scores = None
    gold_val = None
    if args.eval_module_path:
        eval_mod = dynamic_import(args.eval_module_path)
        get_coref_scores = eval_mod.get_coref_scores
        gold_val = build_gold_topics(args.val_split)

    best_conll = -1.0
    best_path = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        if isinstance(eps_logit, torch.nn.Parameter):
            eps_logit.requires_grad_(True)

        running_loss = 0.0
        n_topics = 0

        # Progress bar counts topics (not batches). This makes it informative even when
        # topic_batch_size > 1.
        pbar = tqdm(
            total=n_train_topics,
            desc=f"train epoch {epoch}/{args.epochs}",
            dynamic_ncols=True,
            leave=True,
        )

        for batch_topics in train_dl:
            # collator returns list of topic dicts
            for topic in batch_topics:
                n_topics += 1
                optimizer.zero_grad(set_to_none=True)

                signatures: List[str] = topic["signatures"]
                cluster_ids: List[int] = topic["cluster_ids"]
                n = len(signatures)

                rng = np.random.default_rng(args.seed + epoch * 10_000 + n_topics)

                # per-mention streaming loss
                loss_accum = 0.0
                denom = 0

                for i in range(n):
                    cand_ids = build_candidates_train(
                        cluster_ids=cluster_ids,
                        i=i,
                        strategy=args.cand_strategy,
                        window=args.cand_window,
                        max_candidates=args.cand_max_candidates,
                        rng=rng,
                    )
                    if len(cand_ids) == 0:
                        # only eps action exists; if no gold antecedent then loss=-log p(eps)=0
                        # If there *is* gold antecedent (can't happen at i=0), safe fallback.
                        continue

                    gold_ants = [j for j in cand_ids if cluster_ids[j] == cluster_ids[i]]

                    cand_sigs = [signatures[j] for j in cand_ids]

                    # forward for this mention
                    cand_logits = score_pairs(
                        model=model,
                        tokenizer=tok,
                        sig_i=signatures[i],
                        sig_js=cand_sigs,
                        device=device,
                        max_length=args.max_length,
                        pair_batch_size=args.pair_batch_size,
                        amp=use_amp,
                    )
                    # eps logit value as tensor
                    eps_t = eps_logit if torch.is_tensor(eps_logit) else torch.tensor(float(eps_logit), device=device)

                    loss_i = marginal_nll_for_mention(
                        cand_logits=cand_logits,
                        cand_ids=cand_ids,
                        gold_ants=gold_ants,
                        eps_logit=eps_t,
                    )

                    # normalize by mentions to keep scale stable across topic sizes
                    loss_i = loss_i / max(1, n)

                    scaler.scale(loss_i).backward()
                    loss_accum += float(loss_i.detach().item())
                    denom += 1

                # step
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_loss += loss_accum

                # tqdm update per topic
                try:
                    pbar.update(1)
                    lr = float(optimizer.param_groups[0]["lr"])
                    pbar.set_postfix({
                        "avg_loss": f"{(running_loss / max(1, n_topics)):.4f}",
                        "topic_loss": f"{loss_accum:.4f}",
                        "lr": f"{lr:.2e}",
                    })
                except Exception:
                    pass

        try:
            pbar.close()
        except Exception:
            pass

        avg_loss = running_loss / max(1, n_topics)
        print(f"[epoch {epoch}] train_loss={avg_loss:.6f} topics={n_topics}")

        # optional evaluation
        conll = None
        if args.eval_module_path and args.eval_every_epoch:
            model.eval()
            labels_by_tid: Dict[int, List[int]] = {}
            with torch.no_grad():
                for batch_topics in val_dl:
                    topic = batch_topics[0]
                    tid = int(topic["topic_id"])
                    sigs = topic["signatures"]
                    # eps tensor
                    eps_t = eps_logit if torch.is_tensor(eps_logit) else torch.tensor(float(eps_logit), device=device)
                    pred_labels = decode_topic_greedy(
                        model=model,
                        tokenizer=tok,
                        signatures=sigs,
                        cand_strategy=args.cand_strategy,
                        cand_window=args.cand_window,
                        cand_max_candidates=args.cand_max_candidates,
                        eps_logit=eps_t,
                        device=device,
                        max_length=args.max_length,
                        pair_batch_size=args.pair_batch_size,
                        amp=use_amp,
                        seed=args.seed,
                    )
                    labels_by_tid[tid] = pred_labels

            system = build_system_from_pred_labels(args.val_split, labels_by_tid)
            scores = get_coref_scores(gold_val, system)
            conll = (scores["conll"] / 3.0) * 100.0
            print(f"[epoch {epoch}] val CoNLL={conll:.2f}")
            # print a few key metrics
            for metric in ["muc", "bcub", "ceafe"]:
                if metric in scores:
                    r, p, f = scores[metric]
                    print(f"  {metric}: R={r*100:.2f} P={p*100:.2f} F1={f*100:.2f}")

        # save
        if args.save_every_epoch or (conll is not None and conll > best_conll):
            ckpt_path = os.path.join(args.output_dir, f"epoch{epoch}.pt")
            torch.save({
                "state_dict": model.state_dict(),
                "eps_logit": float(eps_logit.detach().item()) if torch.is_tensor(eps_logit) else float(eps_logit),
                "args": vars(args),
            }, ckpt_path)
            print(f"[save] {ckpt_path} {conll:.2f}" if conll is not None else f"[save] {ckpt_path}")

        if conll is not None and conll > best_conll:
            best_conll = conll
            best_path = ckpt_path

    if best_path is not None:
        print(f"[done] best CoNLL={best_conll:.2f} checkpoint={best_path}")
    else:
        print("[done] training complete (no CoNLL eval performed).")

if __name__ == "__main__":
    main()
