#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x=None, **kwargs):
        return x

from models.signature_topic_dataset import (
    SIGNATURE_SPECIAL_TOKENS,
    SignatureTopicDataset,
    TopicCollator,
    build_signature_tokenizer,
    ensure_signature_special_tokens,
)
from models.signature_partition import (
    PDHGMulticutQP,
    SignaturePartitionModel,
    build_sparse_proposal_graph,
    gold_cut_labels,
    round_partition_labels,
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
    tokenizer,
    signatures: Sequence[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
    amp: bool,
) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    for s in range(0, len(signatures), batch_size):
        chunk = list(signatures[s:s + batch_size])
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        tti = enc.get("token_type_ids")
        if tti is not None:
            tti = tti.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            z = model.encoder(input_ids=input_ids, attention_mask=attn, token_type_ids=tti)
        outs.append(z)
    return torch.cat(outs, dim=0) if outs else torch.zeros((0, model.node_dim), device=device)


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
    probs_f = probs.float().clamp(1e-6, 1.0 - 1e-6)
    y_f = y.float()
    weight = balanced_edge_weights(y_f) if balance else None
    with torch.amp.autocast(device_type=probs.device.type, enabled=False):
        return F.binary_cross_entropy(probs_f, y_f, weight=weight)


def weighted_bce_logits(logits: torch.Tensor, y: torch.Tensor, balance: bool) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    logits_f = logits.float()
    y_f = y.float()
    weight = balanced_edge_weights(y_f) if balance else None
    with torch.amp.autocast(device_type=logits.device.type, enabled=False):
        return F.binary_cross_entropy_with_logits(logits_f, y_f, weight=weight)


def soft_b3_local_loss(keep_soft: torch.Tensor, y_same: torch.Tensor, edge_idx: torch.Tensor, n_nodes: int) -> torch.Tensor:
    if keep_soft.numel() == 0 or n_nodes == 0:
        return keep_soft.new_tensor(0.0)
    u = edge_idx[:, 0]
    v = edge_idx[:, 1]

    pred_mass = torch.ones((n_nodes,), dtype=torch.float32, device=keep_soft.device)
    gold_mass = torch.ones((n_nodes,), dtype=torch.float32, device=keep_soft.device)
    corr_mass = torch.ones((n_nodes,), dtype=torch.float32, device=keep_soft.device)

    pred_mass.index_add_(0, u, keep_soft)
    pred_mass.index_add_(0, v, keep_soft)
    corr = keep_soft * y_same
    corr_mass.index_add_(0, u, corr)
    corr_mass.index_add_(0, v, corr)
    gold_mass.index_add_(0, u, y_same)
    gold_mass.index_add_(0, v, y_same)

    prec = corr_mass / pred_mass.clamp_min(1e-6)
    rec = corr_mass / gold_mass.clamp_min(1e-6)
    f1 = 2.0 * prec * rec / (prec + rec).clamp_min(1e-6)
    return 1.0 - f1.mean()


def soft_lea_loss(keep_soft: torch.Tensor, cluster_ids: Sequence[int], edges: Sequence[Tuple[int, int]]) -> torch.Tensor:
    if keep_soft.numel() == 0:
        return keep_soft.new_tensor(0.0)
    clus = [int(c) for c in cluster_ids]
    sizes: Dict[int, int] = {}
    for c in clus:
        sizes[c] = sizes.get(c, 0) + 1
    same = torch.tensor([1.0 if clus[u] == clus[v] else 0.0 for (u, v) in edges], dtype=torch.float32, device=keep_soft.device)
    size_w = torch.tensor([float(sizes[clus[u]]) if clus[u] == clus[v] else 1.0 for (u, v) in edges], dtype=torch.float32, device=keep_soft.device)
    num = (keep_soft * same * size_w).sum()
    pred_den = (keep_soft * torch.where(same > 0.5, size_w, torch.ones_like(size_w))).sum().clamp_min(1e-6)
    gold_den = (same * size_w).sum().clamp_min(1e-6)
    prec = num / pred_den
    rec = num / gold_den
    f1 = 2.0 * prec * rec / (prec + rec).clamp_min(1e-6)
    return 1.0 - f1


def cut_rate_calibration_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_tensor(0.0)
    return (x.float().mean() - y.float().mean()).pow(2)


def smooth_node_states(node_embs: torch.Tensor, edge_idx: torch.Tensor, keep_soft: torch.Tensor) -> torch.Tensor:
    if edge_idx.numel() == 0 or keep_soft.numel() == 0:
        return node_embs
    n, d = node_embs.shape
    u = edge_idx[:, 0]
    v = edge_idx[:, 1]
    w = keep_soft.float().unsqueeze(-1)
    agg = torch.zeros((n, d), dtype=node_embs.dtype, device=node_embs.device)
    deg = torch.ones((n, 1), dtype=node_embs.dtype, device=node_embs.device)
    agg.index_add_(0, u, w * node_embs[v])
    agg.index_add_(0, v, w * node_embs[u])
    deg.index_add_(0, u, w)
    deg.index_add_(0, v, w)
    return (node_embs + agg) / deg.clamp_min(1.0)


def sinkhorn_transport(scores: torch.Tensor, tau: float = 0.1, iters: int = 20) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    tau = max(float(tau), 1e-4)
    K, M = scores.shape
    logits = scores.float() / tau
    logits = logits - logits.max()
    Kmat = torch.exp(logits).clamp_min(1e-12)
    a = torch.full((K,), 1.0 / max(1, K), dtype=torch.float32, device=scores.device)
    b = torch.full((M,), 1.0 / max(1, M), dtype=torch.float32, device=scores.device)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(int(iters)):
        u = a / (Kmat @ v).clamp_min(1e-8)
        v = b / (Kmat.transpose(0, 1) @ u).clamp_min(1e-8)
    return (u.unsqueeze(1) * Kmat) * v.unsqueeze(0)


def soft_ceaf_loss(
    model: SignaturePartitionModel,
    node_embs: torch.Tensor,
    edge_idx: torch.Tensor,
    keep_soft: torch.Tensor,
    cluster_ids: Sequence[int],
    assign_temp: float,
    sinkhorn_tau: float,
    sinkhorn_iters: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if node_embs.numel() == 0:
        z = node_embs.new_tensor(0.0)
        return z, {"soft_ceaf": 0.0, "q_entropy": 0.0, "p_entropy": 0.0, "n_slots": 0}

    smoothed = smooth_node_states(node_embs.float(), edge_idx, keep_soft.float())
    uniq = sorted(set(int(c) for c in cluster_ids))
    n_gold = max(1, len(uniq))
    n_slots = min(int(model.ceaf_slots_cap), n_gold)
    logits = model.cluster_assignment_logits(smoothed, n_slots)
    assign_temp = max(float(assign_temp), 1e-4)
    Q = torch.softmax(logits / assign_temp, dim=-1)

    gold_map = {c: i for i, c in enumerate(uniq)}
    g_idx = torch.tensor([gold_map[int(c)] for c in cluster_ids], dtype=torch.long, device=node_embs.device)
    G = F.one_hot(g_idx, num_classes=n_gold).float()

    overlap = Q.transpose(0, 1) @ G  # [K, M]
    pred_mass = Q.sum(dim=0, keepdim=True).transpose(0, 1)  # [K,1]
    gold_mass = G.sum(dim=0, keepdim=True)    # [1,M]
    phi = (2.0 * overlap) / (pred_mass + gold_mass).clamp_min(1e-6)
    P = sinkhorn_transport(phi, tau=sinkhorn_tau, iters=sinkhorn_iters)
    soft_ceaf = (P * phi).sum()
    loss = 1.0 - soft_ceaf

    q_ent = float((-(Q * (Q.clamp_min(1e-8).log())).sum(dim=-1).mean()).item())
    p_ent = float((-(P * (P.clamp_min(1e-8).log())).sum()).item()) if P.numel() > 0 else 0.0
    dbg = {
        "soft_ceaf": float(soft_ceaf.item()),
        "q_entropy": q_ent,
        "p_entropy": p_ent,
        "n_slots": int(n_slots),
    }
    return loss, dbg


def build_topic_stats_tensor(x: torch.Tensor, graph, pdhg_diag: Dict[str, float], device: torch.device) -> torch.Tensor:
    if x.numel() == 0:
        stats = torch.zeros((1, 9), dtype=torch.float32, device=device)
        stats[0, 0] = float(graph.n_nodes) / 64.0
        return stats
    xf = x.detach().float()
    n_edges = max(1, len(graph.all_edges))
    stats = torch.tensor([
        float(graph.n_nodes) / 64.0,
        float(n_edges) / 512.0,
        float(xf.mean().item()),
        float(xf.std(unbiased=False).item()) if xf.numel() > 1 else 0.0,
        float(xf.min().item()),
        float(xf.max().item()),
        float(pdhg_diag.get("objective", 0.0)) / float(max(1, n_edges)),
        float(pdhg_diag.get("mean_abs_dx", 0.0)),
        float(pdhg_diag.get("viol_frac", 0.0)),
    ], dtype=torch.float32, device=device).unsqueeze(0)
    return stats


def soft_keep_from_threshold(x: torch.Tensor, threshold: torch.Tensor, temp: float) -> torch.Tensor:
    temp = max(float(temp), 1e-4)
    return torch.sigmoid((threshold.float() - x.float()) / temp)


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
    blob = {"state_dict": model.state_dict(), "args": vars(args), "signature_special_tokens": list(SIGNATURE_SPECIAL_TOKENS)}
    if extra:
        blob.update(extra)
    torch.save(blob, path)
    print(f"[write] {path}")


def make_optimizer_and_scheduler(
    model: SignaturePartitionModel,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
    total_steps: int,
    warmup_ratio: float,
):
    no_decay_terms = ("bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm.weight", "layer_norm.bias", ".ln.")
    enc_decay: List[torch.nn.Parameter] = []
    enc_nodecay: List[torch.nn.Parameter] = []
    head_decay: List[torch.nn.Parameter] = []
    head_nodecay: List[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = any(term in name for term in no_decay_terms)
        is_encoder = name.startswith("encoder.")
        bucket = None
        if is_encoder and is_no_decay:
            bucket = enc_nodecay
        elif is_encoder:
            bucket = enc_decay
        elif is_no_decay:
            bucket = head_nodecay
        else:
            bucket = head_decay
        bucket.append(param)

    param_groups = []
    if enc_decay:
        param_groups.append({"params": enc_decay, "lr": encoder_lr, "weight_decay": weight_decay})
    if enc_nodecay:
        param_groups.append({"params": enc_nodecay, "lr": encoder_lr, "weight_decay": 0.0})
    if head_decay:
        param_groups.append({"params": head_decay, "lr": head_lr, "weight_decay": weight_decay})
    if head_nodecay:
        param_groups.append({"params": head_nodecay, "lr": head_lr, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(param_groups)
    warmup_steps = int(float(warmup_ratio) * float(total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return optimizer, scheduler


def topic_forward(
    *,
    model: SignaturePartitionModel,
    decoder: PDHGMulticutQP,
    tokenizer,
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
    fixed_threshold: Optional[float],
    threshold_temp: float,
    lambda_aux: float,
    lambda_b3: float,
    lambda_lea: float,
    lambda_ceaf: float,
    lambda_cal: float,
    balance_edge_loss: bool,
    ceaf_assign_temp: float,
    ceaf_sinkhorn_tau: float,
    ceaf_sinkhorn_iters: int,
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

    with torch.amp.autocast(device_type=device.type, enabled=amp):
        all_w = model.score_edges(node_embs, graph.all_edge_idx)

    x, pdhg_diag = decoder(all_w, graph.row_edge_index, graph.row_coeff, return_diagnostics=True)
    y_all = gold_cut_labels(cluster_ids, graph.all_edges, device=device)
    y_same = 1.0 - y_all.float()

    stats = build_topic_stats_tensor(x, graph, pdhg_diag, device)
    if fixed_threshold is None:
        threshold_used = model.predict_threshold(stats).view(())
        threshold_source = "adaptive"
    else:
        threshold_used = torch.tensor(float(fixed_threshold), dtype=torch.float32, device=device)
        threshold_source = "fixed"

    keep_soft = soft_keep_from_threshold(x, threshold_used, threshold_temp)

    loss_part = weighted_bce_probs(x, y_all, balance=balance_edge_loss)
    loss_edge = weighted_bce_logits(-all_w, y_all, balance=balance_edge_loss) if all_w.numel() > 0 else x.new_tensor(0.0)
    loss_b3 = soft_b3_local_loss(keep_soft, y_same, graph.all_edge_idx, graph.n_nodes)
    loss_lea = soft_lea_loss(keep_soft, cluster_ids, graph.all_edges)
    loss_ceaf, ceaf_dbg = soft_ceaf_loss(
        model=model,
        node_embs=node_embs,
        edge_idx=graph.all_edge_idx,
        keep_soft=keep_soft,
        cluster_ids=cluster_ids,
        assign_temp=ceaf_assign_temp,
        sinkhorn_tau=ceaf_sinkhorn_tau,
        sinkhorn_iters=ceaf_sinkhorn_iters,
    )
    loss_cal = cut_rate_calibration_loss(x, y_all)

    loss = (
        loss_part
        + float(lambda_b3) * loss_b3
        + float(lambda_lea) * loss_lea
        + float(lambda_ceaf) * loss_ceaf
        + float(lambda_aux) * loss_edge
        + float(lambda_cal) * loss_cal
    )

    hard_threshold = float(threshold_used.detach().item())
    pred_labels = round_partition_labels(graph.n_nodes, graph.all_edges, x, threshold=hard_threshold)
    dbg = {
        "n_nodes": int(graph.n_nodes),
        "n_prop_edges": int(len(graph.proposal_edges)),
        "n_all_edges": int(len(graph.all_edges)),
        "n_tri_rows": int(graph.row_edge_index.size(0)),
        "loss_partition": float(loss_part.detach().item()),
        "loss_b3": float(loss_b3.detach().item()),
        "loss_lea": float(loss_lea.detach().item()),
        "loss_ceaf": float(loss_ceaf.detach().item()),
        "loss_edge": float(loss_edge.detach().item()),
        "loss_cal": float(loss_cal.detach().item()),
        "loss_total_raw": float(loss.detach().item()),
        "n_clusters": int(len(set(pred_labels))),
        "threshold_used": hard_threshold,
        "threshold_source": threshold_source,
        "pred_cut_rate": float(x.detach().float().mean().item()) if x.numel() > 0 else 0.0,
        "hard_cut_rate": float((x.detach().float() > hard_threshold).float().mean().item()) if x.numel() > 0 else 0.0,
        "gold_cut_rate": float(y_all.detach().float().mean().item()) if y_all.numel() > 0 else 0.0,
        "objective": float(pdhg_diag.get("objective", 0.0)),
        "mean_abs_dx": float(pdhg_diag.get("mean_abs_dx", 0.0)),
        "max_abs_dx": float(pdhg_diag.get("max_abs_dx", 0.0)),
        "viol_frac": float(pdhg_diag.get("viol_frac", 0.0)),
        "viol_mean": float(pdhg_diag.get("viol_mean", 0.0)),
        "viol_max": float(pdhg_diag.get("viol_max", 0.0)),
        "tau_mean": float(pdhg_diag.get("tau_mean", 0.0)),
        "sigma_mean": float(pdhg_diag.get("sigma_mean", 0.0)),
        "ceaf_soft": float(ceaf_dbg.get("soft_ceaf", 0.0)),
        "ceaf_q_entropy": float(ceaf_dbg.get("q_entropy", 0.0)),
        "ceaf_p_entropy": float(ceaf_dbg.get("p_entropy", 0.0)),
        "ceaf_slots": int(ceaf_dbg.get("n_slots", 0)),
        "pred_labels": pred_labels,
    }
    return loss, dbg


@torch.no_grad()
def evaluate_partition(
    *,
    model: SignaturePartitionModel,
    decoder: PDHGMulticutQP,
    tokenizer,
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
    threshold: Optional[float],
    threshold_temp: float,
    ceaf_assign_temp: float,
    ceaf_sinkhorn_tau: float,
    ceaf_sinkhorn_iters: int,
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
            fixed_threshold=threshold,
            threshold_temp=threshold_temp,
            lambda_aux=0.0,
            lambda_b3=0.0,
            lambda_lea=0.0,
            lambda_ceaf=0.0,
            lambda_cal=0.0,
            balance_edge_loss=False,
            ceaf_assign_temp=ceaf_assign_temp,
            ceaf_sinkhorn_tau=ceaf_sinkhorn_tau,
            ceaf_sinkhorn_iters=ceaf_sinkhorn_iters,
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
    ap.add_argument("--node_proj", default="none", choices=["none", "linear", "mlp"])
    ap.add_argument("--proj_hidden", type=int, default=512)
    ap.add_argument("--proj_dim", type=int, default=256)
    ap.add_argument("--proj_layers", type=int, default=2)
    ap.add_argument("--edge_hidden", type=int, default=512)
    ap.add_argument("--edge_layers", type=int, default=3)
    ap.add_argument("--edge_init_bias", type=float, default=-0.1)
    ap.add_argument("--ceaf_slots_cap", type=int, default=32)
    ap.add_argument("--ceaf_head_hidden", type=int, default=128)
    ap.add_argument("--threshold_head_hidden", type=int, default=64)
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
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--threshold_temp", type=float, default=0.05)
    ap.add_argument("--ceaf_assign_temp", type=float, default=0.7)
    ap.add_argument("--ceaf_sinkhorn_tau", type=float, default=0.1)
    ap.add_argument("--ceaf_sinkhorn_iters", type=int, default=20)

    ap.add_argument("--lambda_b3", type=float, default=1.0)
    ap.add_argument("--lambda_lea", type=float, default=1.0)
    ap.add_argument("--lambda_ceaf", type=float, default=0.2)
    ap.add_argument("--lambda_aux", type=float, default=0.0)
    ap.add_argument("--lambda_cal", type=float, default=0.0)
    ap.add_argument("--balance_edge_loss", action="store_true")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--topic_batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=43)

    ap.add_argument("--init_partition_ckpt", default=None)
    ap.add_argument("--init_encoder_from_crossencoder_ckpt", default=None)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.topic_batch_size != 1:
        raise ValueError("topic_batch_size must be 1 for this topic-level trainer")
    args.grad_accum_steps = max(1, int(args.grad_accum_steps))

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
        node_proj=args.node_proj,
        edge_hidden=args.edge_hidden,
        edge_layers=args.edge_layers,
        edge_init_bias=args.edge_init_bias,
        ceaf_slots_cap=args.ceaf_slots_cap,
        ceaf_head_hidden=args.ceaf_head_hidden,
        threshold_head_hidden=args.threshold_head_hidden,
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

    steps_per_epoch = math.ceil(n_train_topics / float(args.grad_accum_steps))
    total_steps = max(1, steps_per_epoch * args.epochs)
    optimizer, scheduler = make_optimizer_and_scheduler(
        model=model,
        encoder_lr=args.lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
    )

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_conll = -1.0
    best_path = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running: Dict[str, float] = {
            "scaled": 0.0,
            "raw": 0.0,
            "part": 0.0,
            "b3": 0.0,
            "lea": 0.0,
            "ceaf": 0.0,
            "edge": 0.0,
            "cal": 0.0,
            "thr": 0.0,
        }
        running_topics = 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(total=n_train_topics, desc=f"train epoch {epoch}/{args.epochs}", dynamic_ncols=True, leave=True)

        for step_idx, batch_topics in enumerate(train_dl, start=1):
            topic = batch_topics[0]
            loss_raw, dbg = topic_forward(
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
                fixed_threshold=args.threshold,
                threshold_temp=args.threshold_temp,
                lambda_aux=args.lambda_aux,
                lambda_b3=args.lambda_b3,
                lambda_lea=args.lambda_lea,
                lambda_ceaf=args.lambda_ceaf,
                lambda_cal=args.lambda_cal,
                balance_edge_loss=args.balance_edge_loss,
                ceaf_assign_temp=args.ceaf_assign_temp,
                ceaf_sinkhorn_tau=args.ceaf_sinkhorn_tau,
                ceaf_sinkhorn_iters=args.ceaf_sinkhorn_iters,
            )
            loss_backward = loss_raw / float(args.grad_accum_steps)
            scaler.scale(loss_backward).backward()

            do_step = (step_idx % args.grad_accum_steps == 0) or (step_idx == n_train_topics)
            if do_step:
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running["scaled"] += float(loss_backward.detach().item())
            running["raw"] += float(dbg["loss_total_raw"])
            running["part"] += float(dbg["loss_partition"])
            running["b3"] += float(dbg["loss_b3"])
            running["lea"] += float(dbg["loss_lea"])
            running["ceaf"] += float(dbg["loss_ceaf"])
            running["edge"] += float(dbg["loss_edge"])
            running["cal"] += float(dbg["loss_cal"])
            running["thr"] += float(dbg["threshold_used"])
            running_topics += 1
            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{running['scaled'] / max(1, running_topics):.4f}",
                "raw": f"{running['raw'] / max(1, running_topics):.4f}",
                "part": f"{running['part'] / max(1, running_topics):.4f}",
                "b3": f"{running['b3'] / max(1, running_topics):.4f}",
                "lea": f"{running['lea'] / max(1, running_topics):.4f}",
                "ceaf": f"{running['ceaf'] / max(1, running_topics):.4f}",
                "thr": f"{running['thr'] / max(1, running_topics):.3f}",
                "E": dbg["n_all_edges"],
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
                threshold_temp=args.threshold_temp,
                ceaf_assign_temp=args.ceaf_assign_temp,
                ceaf_sinkhorn_tau=args.ceaf_sinkhorn_tau,
                ceaf_sinkhorn_iters=args.ceaf_sinkhorn_iters,
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
