#!/usr/bin/env python3
"""
eval_signature_antecedent_null_decisions.py

Evaluation script for antecedent signature coref with NULL-epsilon (Option 2A),
with per-mention decision dump for Layer-2 diagnostics.

Epsilon action is represented by scoring (sig_i, NULL_SIG) with the same
cross-encoder used for antecedent pairs. For each mention i:

  logits = [ s(i, NULL_SIG) ] + [ s(i, j) for j in candidate antecedents ]

Decision:
- choose argmax over logits (optionally apply link_margin: only link if best_cand > null + margin)
- if choose NULL -> start new cluster
- else link to chosen antecedent and union-find

Outputs:
- labels_by_tid_<split>.json (optional)
- system_<split>.jsonl (optional)
- decisions_<split>.jsonl (optional) : per mention, includes error_type, confidence, and signature meta
"""

import os
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.signature_crossencoder import SignatureCorefCrossEncoder
from models.signature_topic_dataset import SignatureTopicDataset, TopicCollator

# reuse functions from trainer for consistent candidate sampling and scoring
from train_signature_antecedent_streaming import (
    set_seed,
    ensure_dir,
    dynamic_import,
    build_gold_topics,
    build_system_from_pred_labels,
    build_candidates_infer,
    score_pairs,
    UnionFind,
)

DEFAULT_NULL_SIG = "<NULL_ANT> no antecedent </NULL_ANT>"


def load_sig_meta(signatures_path: str) -> Dict[Tuple[int, int, int, int], Dict[str, Any]]:
    """Metadata keyed by (topic_id, para_idx, start, end) inclusive spans."""
    meta = {}
    with open(signatures_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            tid = int(obj["topic_id"])
            pid = int(obj["para_idx"])
            s, e = obj["gold_span"]
            key = (tid, pid, int(s), int(e))
            meta[key] = {
                "match_kind": obj.get("match_kind", ""),
                "match_kind_raw": obj.get("match_kind_raw", ""),
                "used_extended_context": bool(obj.get("used_extended_context", False)),
                "fallback_reason": obj.get("fallback_reason", ""),
                "sig_type": obj.get("sig_type", ""),
                "sig_n_aliases": int(obj.get("sig_n_aliases", 0) or 0),
                "sig_n_out": int(obj.get("sig_n_out", 0) or 0),
                "sig_n_in": int(obj.get("sig_n_in", 0) or 0),
                "sig_len_chars": int(obj.get("sig_len_chars", 0) or 0),
            }
    return meta


def load_ckpt(path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Checkpoint {path} missing state_dict.")
    ckpt_args = ckpt.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}
    ckpt["args"] = ckpt_args
    # null_sig and eps_mode may be stored
    return ckpt


def resolve(cli_val, ckpt_args: Dict[str, Any], key: str, default=None):
    return cli_val if cli_val is not None else ckpt_args.get(key, default)


@torch.no_grad()
def decode_topic_null_with_decisions(
    *,
    model: SignatureCorefCrossEncoder,
    tokenizer,
    topic_id: int,
    signatures: List[str],
    gold_cluster_ids: List[int],
    gold_mentions: List[List[int]],
    sig_meta: Optional[Dict[Tuple[int, int, int, int], Dict[str, Any]]],
    null_sig: str,
    cand_strategy: str,
    cand_window: int,
    cand_max_candidates: int,
    link_margin: float,
    device: torch.device,
    max_length: int,
    pair_batch_size: int,
    amp: bool,
    seed: int,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Returns predicted labels + per-mention decision rows."""
    n = len(signatures)
    uf = UnionFind(n)
    rng = np.random.default_rng(seed + int(topic_id))

    chosen_ante = [-1] * n  # -1 means NULL (new cluster)
    rows: List[Dict[str, Any]] = []

    for i in range(n):
        cand_ids = build_candidates_infer(i, cand_strategy, cand_window, cand_max_candidates, rng)
        cand_sigs = [signatures[j] for j in cand_ids]

        # gold antecedents
        gold_ants = [j for j in range(i) if gold_cluster_ids[j] == gold_cluster_ids[i]]
        has_gold_ant = bool(gold_ants)
        gold_present = None
        nearest_gold_dist = None
        if has_gold_ant:
            cand_set = set(cand_ids)
            gold_present = int(any(j in cand_set for j in gold_ants))
            nearest_gold_dist = int(i - max(gold_ants))

        # score NULL + candidates
        all_sigs = [null_sig] + cand_sigs
        logits = score_pairs(
            model=model,
            tokenizer=tokenizer,
            sig_i=signatures[i],
            sig_js=all_sigs,
            device=device,
            max_length=max_length,
            pair_batch_size=pair_batch_size,
            amp=amp,
        )  # shape [1+K]
        # logits[0] is NULL
        null_logit = float(logits[0].item()) if logits.numel() > 0 else float("-inf")

        best_pos = int(torch.argmax(logits).item()) if logits.numel() > 0 else 0
        best_logit = float(logits[best_pos].item()) if logits.numel() > 0 else float("-inf")

        # second best
        if logits.numel() >= 2:
            top2 = torch.topk(logits, k=2).values
            second_logit = float(top2[1].item())
        else:
            second_logit = float("-inf")
        margin_best_vs_2nd = best_logit - second_logit

        # apply margin vs NULL: only link if best candidate beats NULL by link_margin
        final_pos = best_pos
        if best_pos != 0:
            if best_logit < null_logit + float(link_margin):
                final_pos = 0

        # probs and gold mass
        probs = torch.softmax(logits, dim=0) if logits.numel() > 0 else torch.tensor([1.0])
        p_null = float(probs[0].item()) if probs.numel() > 0 else 1.0

        p_gold = None
        best_gold_rank = None
        if has_gold_ant:
            if len(cand_ids) == 0:
                p_gold = 0.0
            else:
                idx_map = {cid: k for k, cid in enumerate(cand_ids)}
                gold_pos = [idx_map[j] for j in gold_ants if j in idx_map]
                if gold_pos:
                    # positions in logits are 1+pos because 0 is NULL
                    gold_action_pos = [1 + gp for gp in gold_pos]
                    p_gold = float(probs[gold_action_pos].sum().item())
                    # best gold rank among candidates only (exclude NULL)
                    cand_logits = logits[1:].detach().cpu().numpy().tolist()
                    order = np.argsort([-x for x in cand_logits])
                    best_gold_cand = max(gold_pos, key=lambda k: cand_logits[k])
                    best_gold_rank = int(np.where(order == best_gold_cand)[0][0]) + 1
                else:
                    p_gold = 0.0
                    best_gold_rank = None

        # take action
        if final_pos == 0:
            chosen_ante[i] = -1
        else:
            j = cand_ids[final_pos - 1]
            chosen_ante[i] = int(j)
            uf.union(i, j)

        # meta join
        pid, s, e, _ = gold_mentions[i]
        meta = {}
        if sig_meta is not None:
            meta = sig_meta.get((int(topic_id), int(pid), int(s), int(e)), {}) or {}

        # error type
        if not has_gold_ant:
            err = "new_correct" if chosen_ante[i] == -1 else "new_wrong_link"
        else:
            if gold_present == 0:
                err = "candidate_miss"
            elif chosen_ante[i] == -1:
                err = "epsilon_error"  # chose NULL though gold present
            else:
                err = "correct_link" if gold_cluster_ids[chosen_ante[i]] == gold_cluster_ids[i] else "wrong_link"

        rows.append({
            "topic_id": int(topic_id),
            "mention_idx": int(i),
            "pid": int(pid),
            "start": int(s),
            "end": int(e),
            "gold_cluster": int(gold_cluster_ids[i]),
            "has_gold_antecedent": bool(has_gold_ant),
            "gold_present": gold_present,
            "nearest_gold_dist": nearest_gold_dist,
            "cand_size": int(len(cand_ids)),
            "chosen_antecedent": int(chosen_ante[i]),
            "null_logit": float(null_logit),
            "best_logit": float(best_logit),
            "second_logit": float(second_logit),
            "margin_best_vs_2nd": float(margin_best_vs_2nd),
            "p_null": float(p_null),
            "p_gold": p_gold,
            "best_gold_rank": best_gold_rank,
            "error_type": err,
            **meta,
        })

    pred_labels = uf.labels()
    # fill predicted clusters for i and chosen
    for r in rows:
        i = int(r["mention_idx"])
        r["pred_cluster"] = int(pred_labels[i])
        j = int(r["chosen_antecedent"])
        r["chosen_pred_cluster"] = int(pred_labels[j]) if j >= 0 else None
        r["chosen_gold_cluster"] = int(gold_cluster_ids[j]) if j >= 0 else None

    return pred_labels, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--signatures_path", required=True)

    ap.add_argument("--output_dir", default="eval_outputs_null")
    ap.add_argument("--save_labels", action="store_true")
    ap.add_argument("--save_system", action="store_true")
    ap.add_argument("--save_decisions", action="store_true")

    ap.add_argument("--eval_module_path", default=None)

    # overrides
    ap.add_argument("--bert_model", default=None)
    ap.add_argument("--tokenizer_name", default=None)
    ap.add_argument("--adapter_name", default=None)
    ap.add_argument("--mlp_hidden", type=int, default=None)
    ap.add_argument("--mlp_layers", type=int, default=None)

    ap.add_argument("--cand_strategy", default=None, choices=["all", "window", "hybrid", "random"])
    ap.add_argument("--cand_window", type=int, default=None)
    ap.add_argument("--cand_max_candidates", type=int, default=None)

    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--pair_batch_size", type=int, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--null_sig", default=None)
    ap.add_argument("--decode_link_margin", type=float, default=None)

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.output_dir)

    ckpt = load_ckpt(args.ckpt, device=device)
    ckpt_args = ckpt["args"]

    bert_model = resolve(args.bert_model, ckpt_args, "bert_model")
    tokenizer_name = resolve(args.tokenizer_name, ckpt_args, "tokenizer_name", None) or bert_model
    adapter_name = resolve(args.adapter_name, ckpt_args, "adapter_name", None)
    mlp_hidden = int(resolve(args.mlp_hidden, ckpt_args, "mlp_hidden", 256))
    mlp_layers = int(resolve(args.mlp_layers, ckpt_args, "mlp_layers", 2))

    cand_strategy = resolve(args.cand_strategy, ckpt_args, "cand_strategy", "hybrid")
    cand_window = int(resolve(args.cand_window, ckpt_args, "cand_window", 24))
    cand_max_candidates = int(resolve(args.cand_max_candidates, ckpt_args, "cand_max_candidates", 24))

    max_length = int(resolve(args.max_length, ckpt_args, "max_length", 384))
    pair_batch_size = int(resolve(args.pair_batch_size, ckpt_args, "pair_batch_size", 8))
    link_margin = float(resolve(args.decode_link_margin, ckpt_args, "decode_link_margin", 0.0))

    null_sig = args.null_sig or ckpt_args.get("null_sig") or DEFAULT_NULL_SIG

    # dataset
    ds = SignatureTopicDataset(split=args.split, signatures_path=args.signatures_path, topics_limit=None, seed=args.seed)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=TopicCollator())

    # tokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    sig_tokens = ckpt.get("signature_special_tokens", None)
    if sig_tokens is None:
        sig_tokens = ["<m>", "</m>", "<M>", "<CANON>", "<TYPE>", "<CTX>", "<ALIASES>", "<OUT>", "<IN>", "<CO>", "<CAN_CTX>", "<NULL_ANT>", "</NULL_ANT>"]
    tok.add_special_tokens({"additional_special_tokens": list(sig_tokens)})

    # model
    model = SignatureCorefCrossEncoder(
        bert_model=bert_model,
        adapter_name=adapter_name,
        mlp_hidden=mlp_hidden,
        mlp_layers=mlp_layers,
    )
    model.bert.resize_token_embeddings(len(tok))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()

    use_amp = bool(args.amp) and device.type == "cuda"

    # meta
    sig_meta = load_sig_meta(args.signatures_path) if args.save_decisions else None

    labels_by_tid: Dict[int, List[int]] = {}
    decisions: List[Dict[str, Any]] = []

    for batch in dl:
        topic = batch[0]
        tid = int(topic["topic_id"])
        sigs = topic["signatures"]
        gold_cluster_ids = topic["cluster_ids"]
        gold_mentions = topic["mentions"]

        pred, rows = decode_topic_null_with_decisions(
            model=model,
            tokenizer=tok,
            topic_id=tid,
            signatures=sigs,
            gold_cluster_ids=gold_cluster_ids,
            gold_mentions=gold_mentions,
            sig_meta=sig_meta,
            null_sig=null_sig,
            cand_strategy=cand_strategy,
            cand_window=cand_window,
            cand_max_candidates=cand_max_candidates,
            link_margin=link_margin,
            device=device,
            max_length=max_length,
            pair_batch_size=pair_batch_size,
            amp=use_amp,
            seed=args.seed,
        )
        labels_by_tid[tid] = pred
        if args.save_decisions:
            decisions.extend(rows)

    if args.save_labels:
        p = os.path.join(args.output_dir, f"labels_by_tid_{args.split}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(labels_by_tid, f)
        print(f"[write] {p}")

    system = build_system_from_pred_labels(args.split, labels_by_tid)
    if args.save_system:
        p = os.path.join(args.output_dir, f"system_{args.split}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for row in system:
                f.write(json.dumps(row) + "\\n")
        print(f"[write] {p}")

    if args.save_decisions:
        p = os.path.join(args.output_dir, f"decisions_{args.split}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for row in decisions:
                f.write(json.dumps(row) + "\\n")
        print(f"[write] {p}")

    if args.eval_module_path:
        eval_mod = dynamic_import(args.eval_module_path, module_name="eval_sigcoref")
        get_coref_scores = getattr(eval_mod, "get_coref_scores")
        gold = build_gold_topics(args.split)
        scores = get_coref_scores(gold, system)
        conll = (scores["conll"] / 3.0) * 100.0
        print(f"[metrics] split={args.split} CoNLL={conll:.2f}")
        for metric in ["mentions", "muc", "bcub", "ceafe", "lea"]:
            if metric in scores:
                r, p2, f1 = scores[metric]
                print(f"  {metric}: R={r*100:.2f} P={p2*100:.2f} F1={f1*100:.2f}")


if __name__ == "__main__":
    main()
