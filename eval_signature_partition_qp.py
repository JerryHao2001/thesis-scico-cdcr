#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from models.signature_topic_dataset import SignatureTopicDataset, TopicCollator, build_signature_tokenizer, ensure_signature_special_tokens
from models.signature_partition import SignaturePartitionModel, PDHGMulticutQP
from train_signature_partition_qp import (
    build_system_from_pred_labels,
    ensure_dir,
    evaluate_partition,
    safe_resize_token_embeddings,
    set_seed,
)


def load_ckpt(path: str) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Checkpoint {path} missing state_dict")
    args = ckpt.get("args", {})
    if not isinstance(args, dict):
        args = {}
    ckpt["args"] = args
    return ckpt


def resolve(cli_val, ckpt_args: Dict[str, Any], key: str, default=None):
    return cli_val if cli_val is not None else ckpt_args.get(key, default)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--signatures_path", required=True)
    ap.add_argument("--output_dir", default="eval_partition_qp")
    ap.add_argument("--save_labels", action="store_true")
    ap.add_argument("--save_system", action="store_true")
    ap.add_argument("--save_debug", action="store_true")
    ap.add_argument("--eval_module_path", default=None)

    ap.add_argument("--bert_model", default=None)
    ap.add_argument("--tokenizer_name", default=None)
    ap.add_argument("--adapter_name", default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--node_proj", default=None, choices=["none", "linear", "mlp"])
    ap.add_argument("--proj_hidden", type=int, default=None)
    ap.add_argument("--proj_dim", type=int, default=None)
    ap.add_argument("--proj_layers", type=int, default=None)
    ap.add_argument("--edge_hidden", type=int, default=None)
    ap.add_argument("--edge_layers", type=int, default=None)
    ap.add_argument("--edge_init_bias", type=float, default=None)
    ap.add_argument("--ceaf_slots_cap", type=int, default=None)
    ap.add_argument("--ceaf_head_hidden", type=int, default=None)
    ap.add_argument("--threshold_head_hidden", type=int, default=None)
    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--encode_batch_size", type=int, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--semantic_k", type=int, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--no_lexical_edges", action="store_true")
    ap.add_argument("--lexical_max_group", type=int, default=None)
    ap.add_argument("--no_triangle_closure", action="store_true")
    ap.add_argument("--closure_max_degree", type=int, default=None)

    ap.add_argument("--qp_mu", type=float, default=None)
    ap.add_argument("--pdhg_iters", type=int, default=None)
    ap.add_argument("--pdhg_theta", type=float, default=None)
    ap.add_argument("--pdhg_step_scale", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--threshold_temp", type=float, default=None)
    ap.add_argument("--ceaf_assign_temp", type=float, default=None)
    ap.add_argument("--ceaf_sinkhorn_tau", type=float, default=None)
    ap.add_argument("--ceaf_sinkhorn_iters", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.output_dir)

    ckpt = load_ckpt(args.ckpt)
    ckpt_args = ckpt["args"]

    bert_model = resolve(args.bert_model, ckpt_args, "bert_model", "allenai/scibert_scivocab_uncased")
    tokenizer_name = resolve(args.tokenizer_name, ckpt_args, "tokenizer_name", None) or bert_model
    adapter_name = resolve(args.adapter_name, ckpt_args, "adapter_name", None)
    dropout = float(resolve(args.dropout, ckpt_args, "dropout", 0.1))
    node_proj = resolve(args.node_proj, ckpt_args, "node_proj", "none")
    proj_hidden = int(resolve(args.proj_hidden, ckpt_args, "proj_hidden", 512))
    proj_dim = int(resolve(args.proj_dim, ckpt_args, "proj_dim", 256))
    proj_layers = int(resolve(args.proj_layers, ckpt_args, "proj_layers", 2))
    edge_hidden = int(resolve(args.edge_hidden, ckpt_args, "edge_hidden", 512))
    edge_layers = int(resolve(args.edge_layers, ckpt_args, "edge_layers", 3))
    edge_init_bias = float(resolve(args.edge_init_bias, ckpt_args, "edge_init_bias", -0.1))
    ceaf_slots_cap = int(resolve(args.ceaf_slots_cap, ckpt_args, "ceaf_slots_cap", 32))
    ceaf_head_hidden = int(resolve(args.ceaf_head_hidden, ckpt_args, "ceaf_head_hidden", 128))
    threshold_head_hidden = int(resolve(args.threshold_head_hidden, ckpt_args, "threshold_head_hidden", 64))
    max_length = int(resolve(args.max_length, ckpt_args, "max_length", 384))
    encode_batch_size = int(resolve(args.encode_batch_size, ckpt_args, "encode_batch_size", 16))

    semantic_k = int(resolve(args.semantic_k, ckpt_args, "semantic_k", 12))
    window = int(resolve(args.window, ckpt_args, "window", 12))
    lexical_max_group = int(resolve(args.lexical_max_group, ckpt_args, "lexical_max_group", 32))
    closure_max_degree = int(resolve(args.closure_max_degree, ckpt_args, "closure_max_degree", 24))

    qp_mu = float(resolve(args.qp_mu, ckpt_args, "qp_mu", 0.1))
    pdhg_iters = int(resolve(args.pdhg_iters, ckpt_args, "pdhg_iters", 30))
    pdhg_theta = float(resolve(args.pdhg_theta, ckpt_args, "pdhg_theta", 1.0))
    pdhg_step_scale = float(resolve(args.pdhg_step_scale, ckpt_args, "pdhg_step_scale", 0.9))
    threshold = args.threshold if args.threshold is not None else resolve(args.threshold, ckpt_args, "threshold", None)
    threshold_temp = float(resolve(args.threshold_temp, ckpt_args, "threshold_temp", 0.05))
    ceaf_assign_temp = float(resolve(args.ceaf_assign_temp, ckpt_args, "ceaf_assign_temp", 0.7))
    ceaf_sinkhorn_tau = float(resolve(args.ceaf_sinkhorn_tau, ckpt_args, "ceaf_sinkhorn_tau", 0.1))
    ceaf_sinkhorn_iters = int(resolve(args.ceaf_sinkhorn_iters, ckpt_args, "ceaf_sinkhorn_iters", 20))

    ds = SignatureTopicDataset(split=args.split, signatures_path=args.signatures_path, topics_limit=None, seed=args.seed)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=TopicCollator())

    tok = build_signature_tokenizer(tokenizer_name, use_fast=True)
    ensure_signature_special_tokens(tok)

    model = SignaturePartitionModel(
        bert_model=bert_model,
        adapter_name=adapter_name,
        dropout=dropout,
        proj_hidden=proj_hidden,
        proj_dim=proj_dim,
        proj_layers=proj_layers,
        node_proj=node_proj,
        edge_hidden=edge_hidden,
        edge_layers=edge_layers,
        edge_init_bias=edge_init_bias,
        ceaf_slots_cap=ceaf_slots_cap,
        ceaf_head_hidden=ceaf_head_hidden,
        threshold_head_hidden=threshold_head_hidden,
    )
    safe_resize_token_embeddings(model.encoder.bert, len(tok))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()

    decoder = PDHGMulticutQP(mu=qp_mu, num_iters=pdhg_iters, theta=pdhg_theta, step_scale=pdhg_step_scale)
    use_amp = bool(args.amp and device.type == "cuda")

    ev = evaluate_partition(
        model=model,
        decoder=decoder,
        tokenizer=tok,
        dl=dl,
        split=args.split,
        eval_module_path=args.eval_module_path,
        device=device,
        max_length=max_length,
        encode_batch_size=encode_batch_size,
        amp=use_amp,
        semantic_k=semantic_k,
        window=window,
        add_lexical_edges=not args.no_lexical_edges,
        lexical_max_group=lexical_max_group,
        add_triangle_closure=not args.no_triangle_closure,
        closure_max_degree=closure_max_degree,
        threshold=threshold,
        threshold_temp=threshold_temp,
        ceaf_assign_temp=ceaf_assign_temp,
        ceaf_sinkhorn_tau=ceaf_sinkhorn_tau,
        ceaf_sinkhorn_iters=ceaf_sinkhorn_iters,
    )

    if args.save_labels:
        p = os.path.join(args.output_dir, f"labels_by_tid_{args.split}_partition.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(ev["labels_by_tid"], f)
        print(f"[write] {p}")

    if args.save_system and args.eval_module_path:
        p = os.path.join(args.output_dir, f"system_{args.split}_partition.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for row in ev["system"]:
                f.write(json.dumps(row) + "\n")
        print(f"[write] {p}")
    elif args.save_system:
        system = build_system_from_pred_labels(args.split, ev["labels_by_tid"])
        p = os.path.join(args.output_dir, f"system_{args.split}_partition.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for row in system:
                f.write(json.dumps(row) + "\n")
        print(f"[write] {p}")

    if args.save_debug:
        p = os.path.join(args.output_dir, f"debug_{args.split}_partition.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for row in ev["debug_rows"]:
                f.write(json.dumps(row) + "\n")
        print(f"[write] {p}")

    if args.eval_module_path:
        conll = float(ev["conll"])
        print(f"[metrics] split={args.split} CoNLL={conll:.2f}")
        for metric in ["mentions", "muc", "bcub", "ceafe", "lea"]:
            if metric in ev["scores"]:
                r, p2, f1 = ev["scores"][metric]
                print(f"  {metric}: R={r*100:.2f} P={p2*100:.2f} F1={f1*100:.2f}")


if __name__ == "__main__":
    main()
