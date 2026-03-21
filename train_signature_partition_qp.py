#!/usr/bin/env python3
from __future__ import annotations

"""
train_signature_partition_qp.py

Partition-aligned training for SciCo signature coreference.

Pipeline per topic:
  1) encode each mention signature once with a bi-encoder
  2) build a sparse undirected proposal graph from current node embeddings
  3) score proposal edges with a symmetric edge scorer
  4) solve a QP-relaxed sparse multicut with an unrolled PDHG decoder
  5) train on the decoder output (cut probabilities), not on local antecedent choices

This script keeps the current project conventions:
  - topic-level dataset from models.signature_topic_dataset.SignatureTopicDataset
  - one optimizer step per topic
  - evaluation formatting compatible with evaluate_signature_coref.get_coref_scores
"""

import os
import json
import math
import argparse
import importlib.util
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x=None, **kwargs):
        return x

from models.signature_topic_dataset import (
    SignatureTopicDataset,
    TopicCollator,
    SIGNATURE_SPECIAL_TOKENS,
    build_signature_tokenizer,
    ensure_signature_special_tokens,
)
from models.signature_partition import (
    SignaturePartitionModel,
    PDHGMulticutQP,
    build_sparse_proposal_graph,
    gold_cut_labels,
    round_partition_labels,
    count_triangle_violations,
)


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


def safe_resize_token_embeddings(model, new_size: int) -> None:
    try:
        model.resize_token_embeddings(new_size, mean_resizing=False)
    except TypeError:
        model.resize_token_embeddings(new_size)


def build_gold_topics(split: str) -> List[Dict[str, Any]]:
    ds = load_dataset("allenai/scico")[split]
    gold = []
    for row in ds:
        gold.append({
            "id": int(row["id"]),
            "tokens": row["tokens"],
            "doc_ids": row.get("doc_ids", []),
            "relations": row.get("relations", []),
            "mentions": [[int(pid), int(s), int(e), int(cid)] for (pid, s, e, cid) in row["mentions"]],
        })
    return gold


def build_system_from_pred_labels(split: str, labels_by_tid: Dict[int, List[int]]) -> List[Dict[str, Any]]:
    ds = load_dataset("allenai/scico")[split]
    by_id = {int(r["id"]): r for r in ds}
    system = []
    for tid, row in by_id.items():
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
            "id": int(tid),
            "tokens": row["tokens"],
            "doc_ids": row.get("doc_ids", []),
            "relations": row.get("relations", []),
            "mentions": sys_mentions,
        })
    return system


def encode_signatures_train(
    model: SignaturePartitionModel,
    tokenizer: AutoTokenizer,
    signatures: Sequence[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
    amp: bool,
) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    for s in range(0, len(signatures), batch_size):
        chunk = list(signatures[s:s + batch_size])
        enc = tokenizer(
            chunk,
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
            z = model.encoder(input_ids=input_ids, attention_mask=attn, token_type_ids=tti)
        outs.append(z)
    return torch.cat(outs, dim=0) if outs else torch.zeros((0, model.edge_scorer.net[0].in_features // 4), device=device)


def balanced_edge_weights(y: torch.Tensor) -> torch.Tensor:
    if y.numel() == 0:
        return y.new_zeros((0,))
    pos = float((y > 0.5).sum().item())
    neg = float((y <= 0.5).sum().item())
    total = max(1.0, pos + neg)
    w_pos = total / max(1.0, 2.0 * pos)
    w_neg = total / max(1.0, 2.0 * neg)
    return torch.where(y > 0.5, y.new_full(y.shape, w_pos), y.new_full(y.shape, w_neg))


def weighted_bce_probs(probs: torch.Tensor, y: torch.Tensor, balance: bool) -> torch.Tensor:
    if probs.numel() == 0:
        return probs.new_tensor(0.0)
    # BCE-on-probabilities is not autocast-safe. Always evaluate the loss in float32
    # while preserving gradients to the upstream graph.
    probs_f = probs.float().clamp(1e-6, 1.0 - 1e-6)
    y_f = y.float()
    weight = balanced_edge_weights(y_f) if balance else None
    dev_type = probs.device.type if probs.is_cuda else 'cpu'
    with torch.amp.autocast(device_type=dev_type, enabled=False):
        return F.binary_cross_entropy(probs_f, y_f, weight=weight)


def maybe_load_partition_checkpoint(model: SignaturePartitionModel, ckpt_path: Optional[str]) -> None:
    if not ckpt_path:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    print(f"[init] loaded partition checkpoint: {ckpt_path}")


def maybe_init_encoder_from_crossencoder(model: SignaturePartitionModel, ckpt_path: Optional[str]) -> None:
    if not ckpt_path:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    bert_state = {k[len("bert."):]: v for k, v in state_dict.items() if k.startswith("bert.")}
    if not bert_state:
        raise ValueError(f"No bert.* weights found in cross-encoder checkpoint: {ckpt_path}")
    missing, unexpected = model.encoder.bert.load_state_dict(bert_state, strict=False)
    print(f"[init] loaded encoder backbone from cross-encoder ckpt: {ckpt_path}")
    print(f"[init] encoder backbone missing={len(missing)} unexpected={len(unexpected)}")


def save_checkpoint(path: str, model: SignaturePartitionModel, args: argparse.Namespace, extra: Optional[Dict[str, Any]] = None) -> None:
    blob = {
        "state_dict": model.state_dict(),
        "args": vars(args),
        "signature_special_tokens": list(SIGNATURE_SPECIAL_TOKENS),
    }
    if extra:
        blob.update(extra)
    torch.save(blob, path)
    print(f"[write] {path}")


def topic_forward(
    *,
    model: SignaturePartitionModel,
    decoder: PDHGMulticutQP,
    tokenizer: AutoTokenizer,
    signatures: Sequence[str],
    mentions: Sequence[Sequence[int]],
    cluster_ids: Sequence[int],
    device: torch.device,
    max_length: int,
    encode_batch_size: int,
    amp: bool,
    semantic_k: int,
    window: int,
    add_lexical_edges: bool,
    lexical_max_group: int,
    add_triangle_closure: bool,
    closure_max_degree: int,
    threshold: float,
    lambda_aux: float,
    balance_edge_loss: bool,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    node_embs = encode_signatures_train(
        model=model,
        tokenizer=tokenizer,
        signatures=signatures,
        device=device,
        max_length=max_length,
        batch_size=encode_batch_size,
        amp=amp,
    )

    graph = build_sparse_proposal_graph(
        node_embs=node_embs,
        signatures=signatures,
        mentions=mentions,
        semantic_k=semantic_k,
        window=window,
        add_lexical_edges=add_lexical_edges,
        lexical_max_group=lexical_max_group,
        add_triangle_closure=add_triangle_closure,
        closure_max_degree=closure_max_degree,
    )

    all_w = model.score_edges(node_embs, graph.all_edge_idx, graph.all_scalar_features)
    x = decoder(all_w, graph.row_edge_index, graph.row_coeff)

    y_all = gold_cut_labels(cluster_ids, graph.all_edges, device=device)
    loss_part = weighted_bce_probs(x, y_all, balance=balance_edge_loss)

    loss_aux = x.new_tensor(0.0)
    if lambda_aux > 0.0 and all_w.numel() > 0:
        p_cut_raw = torch.sigmoid(-all_w)
        loss_aux = weighted_bce_probs(p_cut_raw, y_all, balance=balance_edge_loss)

    loss = loss_part + (float(lambda_aux) * loss_aux)

    pred_labels = round_partition_labels(graph.n_nodes, graph.all_edges, x, threshold=threshold)
    dbg = {
        "n_nodes": int(graph.n_nodes),
        "n_prop_edges": int(len(graph.proposal_edges)),
        "n_all_edges": int(len(graph.all_edges)),
        "n_tri_rows": int(graph.row_edge_index.size(0)),
        "loss_partition": float(loss_part.detach().item()),
        "loss_aux": float(loss_aux.detach().item()),
        "loss_total_raw": float(loss.detach().item()),
        "triangle_violations": int(count_triangle_violations(x.detach(), graph.row_edge_index, graph.row_coeff)),
        "n_clusters": int(len(set(pred_labels))),
        "pred_labels": pred_labels,
    }
    return loss, dbg


@torch.no_grad()
def evaluate_partition(
    *,
    model: SignaturePartitionModel,
    decoder: PDHGMulticutQP,
    tokenizer: AutoTokenizer,
    dl: DataLoader,
    split: str,
    eval_module_path: Optional[str],
    device: torch.device,
    max_length: int,
    encode_batch_size: int,
    amp: bool,
    semantic_k: int,
    window: int,
    add_lexical_edges: bool,
    lexical_max_group: int,
    add_triangle_closure: bool,
    closure_max_degree: int,
    threshold: float,
) -> Dict[str, Any]:
    model.eval()
    labels_by_tid: Dict[int, List[int]] = {}
    debug_rows: List[Dict[str, Any]] = []

    for batch in dl:
        topic = batch[0]
        tid = int(topic["topic_id"])
        signatures = topic["signatures"]
        mentions = topic["mentions"]
        cluster_ids = topic["cluster_ids"]
        loss, dbg = topic_forward(
            model=model,
            decoder=decoder,
            tokenizer=tokenizer,
            signatures=signatures,
            mentions=mentions,
            cluster_ids=cluster_ids,
            device=device,
            max_length=max_length,
            encode_batch_size=encode_batch_size,
            amp=amp,
            semantic_k=semantic_k,
            window=window,
            add_lexical_edges=add_lexical_edges,
            lexical_max_group=lexical_max_group,
            add_triangle_closure=add_triangle_closure,
            closure_max_degree=closure_max_degree,
            threshold=threshold,
            lambda_aux=0.0,
            balance_edge_loss=False,
        )
        labels_by_tid[tid] = dbg.pop("pred_labels")
        dbg["topic_id"] = tid
        dbg["loss"] = float(loss.item())
        debug_rows.append(dbg)

    out: Dict[str, Any] = {"labels_by_tid": labels_by_tid, "debug_rows": debug_rows}
    if eval_module_path:
        eval_mod = dynamic_import(eval_module_path)
        get_coref_scores = eval_mod.get_coref_scores
        gold = build_gold_topics(split)
        system = build_system_from_pred_labels(split, labels_by_tid)
        scores = get_coref_scores(gold, system)
        conll = (scores["conll"] / 3.0) * 100.0
        out["system"] = system
        out["scores"] = scores
        out["conll"] = float(conll)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_signatures_path", required=True)
    ap.add_argument("--val_signatures_path", required=True)
    ap.add_argument("--output_dir", default="partition_qp_outputs")
    ap.add_argument("--train_split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--val_split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--eval_module_path", default=None)
    ap.add_argument("--topics_limit", type=int, default=None)

    ap.add_argument("--bert_model", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--tokenizer_name", default=None)
    ap.add_argument("--adapter_name", default=None)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--proj_hidden", type=int, default=512)
    ap.add_argument("--proj_dim", type=int, default=256)
    ap.add_argument("--proj_layers", type=int, default=2)
    ap.add_argument("--edge_hidden", type=int, default=256)
    ap.add_argument("--edge_layers", type=int, default=2)
    ap.add_argument("--edge_init_bias", type=float, default=-1.0)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--encode_batch_size", type=int, default=16)
    ap.add_argument("--grad_checkpointing", action="store_true")

    ap.add_argument("--semantic_k", type=int, default=12)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--no_lexical_edges", action="store_true")
    ap.add_argument("--lexical_max_group", type=int, default=32)
    ap.add_argument("--no_triangle_closure", action="store_true")
    ap.add_argument("--closure_max_degree", type=int, default=24)

    ap.add_argument("--qp_mu", type=float, default=0.1)
    ap.add_argument("--pdhg_iters", type=int, default=30)
    ap.add_argument("--pdhg_theta", type=float, default=1.0)
    ap.add_argument("--pdhg_step_scale", type=float, default=0.9)
    ap.add_argument("--threshold", type=float, default=0.5)

    ap.add_argument("--lambda_aux", type=float, default=0.0)
    ap.add_argument("--balance_edge_loss", action="store_true")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--topic_batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--init_partition_ckpt", default=None)
    ap.add_argument("--init_encoder_from_crossencoder_ckpt", default=None)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.topic_batch_size != 1:
        raise ValueError("topic_batch_size must be 1 for this topic-level trainer")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.output_dir)

    train_ds = SignatureTopicDataset(
        split=args.train_split,
        signatures_path=args.train_signatures_path,
        topics_limit=args.topics_limit,
        seed=args.seed,
    )
    val_ds = SignatureTopicDataset(
        split=args.val_split,
        signatures_path=args.val_signatures_path,
        topics_limit=args.topics_limit,
        seed=args.seed,
    )
    n_train_topics = len(train_ds)
    n_val_topics = len(val_ds)
    print(f"[data] train topics={n_train_topics}  val topics={n_val_topics}")

    total_train_mentions = 0
    for i in range(n_train_topics):
        total_train_mentions += len(train_ds[i]["signatures"])
    avg_mentions_per_topic = max(1.0, total_train_mentions / float(max(1, n_train_topics)))
    print(f"[data] train mentions total={total_train_mentions:,} avg/topic={avg_mentions_per_topic:.2f}")

    collator = TopicCollator()
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collator)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collator)

    tok_name = args.tokenizer_name or args.bert_model
    tok = build_signature_tokenizer(tok_name, use_fast=True)
    ensure_signature_special_tokens(tok)

    model = SignaturePartitionModel(
        bert_model=args.bert_model,
        adapter_name=args.adapter_name,
        dropout=args.dropout,
        proj_hidden=args.proj_hidden,
        proj_dim=args.proj_dim,
        proj_layers=args.proj_layers,
        edge_hidden=args.edge_hidden,
        edge_layers=args.edge_layers,
        scalar_dim=3,
        edge_init_bias=args.edge_init_bias,
    )
    safe_resize_token_embeddings(model.encoder.bert, len(tok))
    if args.grad_checkpointing:
        model.encoder.bert.gradient_checkpointing_enable()
        model.encoder.bert.config.use_cache = False
    for p in model.parameters():
        p.requires_grad = True

    maybe_load_partition_checkpoint(model, args.init_partition_ckpt)
    maybe_init_encoder_from_crossencoder(model, args.init_encoder_from_crossencoder_ckpt)

    model.to(device)
    decoder = PDHGMulticutQP(
        mu=args.qp_mu,
        num_iters=args.pdhg_iters,
        theta=args.pdhg_theta,
        step_scale=args.pdhg_step_scale,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = n_train_topics * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_conll = -1.0
    best_path = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_loss_raw = 0.0
        running_loss_part = 0.0
        running_loss_aux = 0.0
        running_topics = 0
        pbar = tqdm(total=n_train_topics, desc=f"train epoch {epoch}/{args.epochs}", dynamic_ncols=True, leave=True)

        for batch_topics in train_dl:
            topic = batch_topics[0]
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss, dbg = topic_forward(
                    model=model,
                    decoder=decoder,
                    tokenizer=tok,
                    signatures=topic["signatures"],
                    mentions=topic["mentions"],
                    cluster_ids=topic["cluster_ids"],
                    device=device,
                    max_length=args.max_length,
                    encode_batch_size=args.encode_batch_size,
                    amp=use_amp,
                    semantic_k=args.semantic_k,
                    window=args.window,
                    add_lexical_edges=not args.no_lexical_edges,
                    lexical_max_group=args.lexical_max_group,
                    add_triangle_closure=not args.no_triangle_closure,
                    closure_max_degree=args.closure_max_degree,
                    threshold=args.threshold,
                    lambda_aux=args.lambda_aux,
                    balance_edge_loss=args.balance_edge_loss,
                )
                loss = loss / float(avg_mentions_per_topic)

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item())
            running_loss_raw += float(dbg["loss_total_raw"])
            running_loss_part += float(dbg["loss_partition"])
            running_loss_aux += float(dbg["loss_aux"])
            running_topics += 1
            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{running_loss / max(1, running_topics):.4f}",
                "raw": f"{running_loss_raw / max(1, running_topics):.4f}",
                "part": f"{running_loss_part / max(1, running_topics):.4f}",
                "aux": f"{running_loss_aux / max(1, running_topics):.4f}",
                "E": dbg["n_prop_edges"],
                "T": dbg["n_tri_rows"],
                "K": dbg["n_clusters"],
            })

        pbar.close()

        epoch_ckpt = os.path.join(args.output_dir, f"partition_epoch{epoch}.pt")
        save_checkpoint(epoch_ckpt, model, args, extra={"epoch": epoch})

        if args.eval_module_path:
            ev = evaluate_partition(
                model=model,
                decoder=decoder,
                tokenizer=tok,
                dl=val_dl,
                split=args.val_split,
                eval_module_path=args.eval_module_path,
                device=device,
                max_length=args.max_length,
                encode_batch_size=args.encode_batch_size,
                amp=use_amp,
                semantic_k=args.semantic_k,
                window=args.window,
                add_lexical_edges=not args.no_lexical_edges,
                lexical_max_group=args.lexical_max_group,
                add_triangle_closure=not args.no_triangle_closure,
                closure_max_degree=args.closure_max_degree,
                threshold=args.threshold,
            )
            conll = float(ev["conll"])
            print(f"[val] epoch={epoch} CoNLL={conll:.2f}")
            scores = ev.get("scores", {})
            for metric in ["mentions", "muc", "bcub", "ceafe", "lea"]:
                if metric in scores:
                    r, p2, f1 = scores[metric]
                    print(f"  {metric}: R={r*100:.2f} P={p2*100:.2f} F1={f1*100:.2f}")
            if conll > best_conll:
                best_conll = conll
                best_path = os.path.join(args.output_dir, "best_partition.pt")
                save_checkpoint(best_path, model, args, extra={"epoch": epoch, "best_conll": best_conll})
                debug_path = os.path.join(args.output_dir, f"val_debug_epoch{epoch}.jsonl")
                with open(debug_path, "w", encoding="utf-8") as f:
                    for row in ev["debug_rows"]:
                        f.write(json.dumps(row) + "\n")
                print(f"[best] epoch={epoch} CoNLL={best_conll:.2f}")

    if best_path is not None:
        print(f"[done] best checkpoint: {best_path}  CoNLL={best_conll:.2f}")


if __name__ == "__main__":
    main()
