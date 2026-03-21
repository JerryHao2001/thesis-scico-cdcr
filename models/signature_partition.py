from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

try:
    from adapters import AutoAdapterModel  # type: ignore
except ImportError:  # pragma: no cover
    AutoAdapterModel = None


Edge = Tuple[int, int]


_TAG_RE = re.compile(r"<[^>]+>")
_CANON_RE = re.compile(r"<CANON>\s*(.*?)(?=\s*<[^>]+>|$)")
_MENTION_RE = re.compile(r"<m>\s*(.*?)\s*</m>")
_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")


@dataclass
class SparsePartitionGraph:
    n_nodes: int
    proposal_edges: List[Edge]
    all_edges: List[Edge]
    proposal_edge_idx: torch.LongTensor  # [Eprop, 2]
    all_edge_idx: torch.LongTensor       # [Eall, 2]
    proposal_positions: torch.LongTensor # [Eprop] positions into all_edges
    proposal_scalar_features: torch.Tensor  # [Eprop, F]
    all_scalar_features: torch.Tensor       # [Eall, F]
    row_edge_index: torch.LongTensor     # [R, 3]
    row_coeff: torch.Tensor              # [R, 3]
    lexical_keys: List[str]


class SignatureNodeEncoder(nn.Module):
    def __init__(
        self,
        bert_model: str = "allenai/scibert_scivocab_uncased",
        dropout: float = 0.1,
        adapter_name: Optional[str] = None,
        proj_hidden: int = 512,
        proj_dim: int = 256,
        proj_layers: int = 2,
    ):
        super().__init__()
        self.bert_model = bert_model
        self.adapter_name = adapter_name

        if adapter_name is None:
            self.bert = AutoModel.from_pretrained(bert_model)
        else:
            if AutoAdapterModel is None:
                raise RuntimeError("Please `pip install -U adapters` to use HF adapters.")
            self.bert = AutoAdapterModel.from_pretrained(bert_model)
            loaded_name = self.bert.load_adapter(adapter_name, source="hf", load_as="sigpart", set_active=True)
            self.bert.set_active_adapters(loaded_name)

        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        if proj_layers <= 1:
            self.proj = nn.Linear(hidden, proj_dim)
        else:
            self.proj = nn.Sequential(
                nn.Linear(hidden, proj_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(proj_hidden, proj_dim),
            )

    def adapter_parameters(self):
        if hasattr(self.bert, "adapter_parameters"):
            return list(self.bert.adapter_parameters())  # type: ignore[attr-defined]
        return []

    def mean_pool(self, last_hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return last_hidden.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = self.mean_pool(out.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        return self.proj(pooled)


class SymmetricEdgeScorer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        scalar_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        init_bias: float = -1.0,
    ):
        super().__init__()
        in_dim = (4 * node_dim) + 1 + scalar_dim
        layers: List[nn.Module] = []
        prev = in_dim
        if num_layers <= 1:
            layers.append(nn.Linear(prev, 1))
        else:
            for _ in range(num_layers - 1):
                layers.extend([
                    nn.Linear(prev, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                prev = hidden_dim
            layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        if isinstance(self.net[-1], nn.Linear):
            nn.init.constant_(self.net[-1].bias, float(init_bias))

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor, scalar_feats: torch.Tensor) -> torch.Tensor:
        cos = F.cosine_similarity(h_i, h_j, dim=-1).unsqueeze(-1)
        pair = torch.cat([h_i, h_j, torch.abs(h_i - h_j), h_i * h_j, cos, scalar_feats], dim=-1)
        return self.net(pair).squeeze(-1)




class SignaturePartitionModel(nn.Module):
    def __init__(
        self,
        bert_model: str = "allenai/scibert_scivocab_uncased",
        adapter_name: Optional[str] = None,
        dropout: float = 0.1,
        proj_hidden: int = 512,
        proj_dim: int = 256,
        proj_layers: int = 2,
        edge_hidden: int = 256,
        edge_layers: int = 2,
        scalar_dim: int = 3,
        edge_init_bias: float = -1.0,
    ):
        super().__init__()
        self.encoder = SignatureNodeEncoder(
            bert_model=bert_model,
            adapter_name=adapter_name,
            dropout=dropout,
            proj_hidden=proj_hidden,
            proj_dim=proj_dim,
            proj_layers=proj_layers,
        )
        self.edge_scorer = SymmetricEdgeScorer(
            node_dim=proj_dim,
            scalar_dim=scalar_dim,
            hidden_dim=edge_hidden,
            num_layers=edge_layers,
            dropout=dropout,
            init_bias=edge_init_bias,
        )

    def score_edges(self, node_embs: torch.Tensor, edge_idx: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
        if edge_idx.numel() == 0:
            return node_embs.new_zeros((0,))
        h_i = node_embs[edge_idx[:, 0]]
        h_j = node_embs[edge_idx[:, 1]]
        return self.edge_scorer(h_i, h_j, scalar_features)

class PDHGMulticutQP(nn.Module):
    def __init__(
        self,
        mu: float = 0.1,
        num_iters: int = 30,
        theta: float = 1.0,
        step_scale: float = 0.9,
    ):
        super().__init__()
        self.mu = float(mu)
        self.num_iters = int(num_iters)
        self.theta = float(theta)
        self.step_scale = float(step_scale)

    def _estimate_steps(self, num_edges: int, row_edge_index: torch.Tensor, row_coeff: torch.Tensor) -> Tuple[float, float]:
        if row_edge_index.numel() == 0:
            return 1.0, 1.0
        col_abs = torch.zeros((num_edges,), dtype=row_coeff.dtype, device=row_coeff.device)
        flat_idx = row_edge_index.reshape(-1)
        flat_val = row_coeff.abs().reshape(-1)
        col_abs.index_add_(0, flat_idx, flat_val)
        max_col = float(col_abs.max().item()) if col_abs.numel() > 0 else 1.0
        norm_bound = max(1.0, math.sqrt(3.0 * max_col))
        step = self.step_scale / norm_bound
        return step, step

    def forward(
        self,
        weights: torch.Tensor,
        row_edge_index: torch.Tensor,
        row_coeff: torch.Tensor,
    ) -> torch.Tensor:
        if weights.numel() == 0:
            return weights.new_zeros((0,))

        # Keep the decoder numerically stable under AMP by solving in float32
        # while preserving gradient flow back to the edge weights.
        w = weights.float()
        coeff = row_coeff.to(device=weights.device, dtype=torch.float32)

        mu = max(self.mu, 1e-6)
        x = torch.clamp(-w / mu, 0.0, 1.0)
        x_bar = x.clone()

        if row_edge_index.numel() == 0:
            for _ in range(self.num_iters):
                x = torch.clamp((x - w) / (1.0 + mu), 0.0, 1.0)
            return x

        tau, sigma = self._estimate_steps(w.numel(), row_edge_index, coeff)
        p = torch.zeros((row_edge_index.size(0),), device=weights.device, dtype=torch.float32)

        for _ in range(self.num_iters):
            ax = (x_bar[row_edge_index] * coeff).sum(dim=-1)
            p = torch.relu(p + sigma * ax)

            atp = torch.zeros_like(w)
            contrib = (coeff * p.unsqueeze(-1)).reshape(-1)
            atp.index_add_(0, row_edge_index.reshape(-1), contrib)

            x_next = (x - tau * (atp + w)) / (1.0 + tau * mu)
            x_next = torch.clamp(x_next, 0.0, 1.0)
            x_bar = x_next + self.theta * (x_next - x)
            x = x_next
        return x


def extract_signature_key(sig: str) -> str:
    m = _CANON_RE.search(sig)
    text = m.group(1) if m is not None else ""
    if not text:
        m2 = _MENTION_RE.search(sig)
        text = m2.group(1) if m2 is not None else _TAG_RE.sub(" ", sig)
    text = text.lower().strip()
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _edge(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


def build_sparse_proposal_graph(
    node_embs: torch.Tensor,
    signatures: Sequence[str],
    mentions: Sequence[Sequence[int]],
    semantic_k: int = 12,
    window: int = 12,
    add_lexical_edges: bool = True,
    lexical_max_group: int = 32,
    add_triangle_closure: bool = True,
    closure_max_degree: int = 24,
) -> SparsePartitionGraph:
    device = node_embs.device
    n = int(node_embs.size(0))
    proposal: set[Edge] = set()
    lexical_keys = [extract_signature_key(s) for s in signatures]

    if n > 1 and semantic_k > 0:
        z = F.normalize(node_embs.detach(), dim=-1)
        sim = torch.matmul(z, z.T)
        sim.fill_diagonal_(torch.finfo(sim.dtype).min)
        k = min(int(semantic_k), max(0, n - 1))
        if k > 0:
            nbrs = torch.topk(sim, k=k, dim=1).indices.detach().cpu().tolist()
            for i, row in enumerate(nbrs):
                for j in row:
                    if i != int(j):
                        proposal.add(_edge(i, int(j)))

    if window > 0:
        w = int(window)
        for i in range(n):
            lo = max(0, i - w)
            hi = min(n, i + w + 1)
            for j in range(lo, hi):
                if i != j:
                    proposal.add(_edge(i, j))

    if add_lexical_edges:
        buckets: Dict[str, List[int]] = {}
        for i, key in enumerate(lexical_keys):
            if key:
                buckets.setdefault(key, []).append(i)
        for ids in buckets.values():
            if len(ids) <= 1:
                continue
            if len(ids) <= lexical_max_group:
                for a in range(len(ids)):
                    for b in range(a + 1, len(ids)):
                        proposal.add(_edge(ids[a], ids[b]))
            else:
                root = ids[0]
                for j in ids[1:]:
                    proposal.add(_edge(root, j))

    proposal_edges = sorted(proposal)
    all_edges_set = set(proposal_edges)

    if add_triangle_closure and proposal_edges:
        adj: List[set[int]] = [set() for _ in range(n)]
        for u, v in proposal_edges:
            adj[u].add(v)
            adj[v].add(u)
        for hub in range(n):
            nbrs = sorted(adj[hub], key=lambda x: (abs(x - hub), x))
            if closure_max_degree > 0:
                nbrs = nbrs[:closure_max_degree]
            for a in range(len(nbrs)):
                u = nbrs[a]
                for b in range(a + 1, len(nbrs)):
                    v = nbrs[b]
                    all_edges_set.add(_edge(u, v))

    all_edges = sorted(all_edges_set)
    edge_to_pos = {e: idx for idx, e in enumerate(all_edges)}
    proposal_positions = torch.tensor([edge_to_pos[e] for e in proposal_edges], dtype=torch.long, device=device)

    def _scalar_features_for_edges(edges: Sequence[Edge]) -> torch.Tensor:
        if not edges:
            return torch.zeros((0, 3), dtype=torch.float32, device=device)
        same_pid = torch.tensor(
            [1.0 if int(mentions[u][0]) == int(mentions[v][0]) else 0.0 for (u, v) in edges],
            dtype=torch.float32,
            device=device,
        )
        order_dist = torch.tensor(
            [min(abs(u - v), 64) / 64.0 for (u, v) in edges],
            dtype=torch.float32,
            device=device,
        )
        lexical_exact = torch.tensor(
            [1.0 if lexical_keys[u] and lexical_keys[u] == lexical_keys[v] else 0.0 for (u, v) in edges],
            dtype=torch.float32,
            device=device,
        )
        return torch.stack([same_pid, order_dist, lexical_exact], dim=-1)

    if proposal_edges:
        proposal_edge_idx = torch.tensor(proposal_edges, dtype=torch.long, device=device)
    else:
        proposal_edge_idx = torch.zeros((0, 2), dtype=torch.long, device=device)
    proposal_scalar_features = _scalar_features_for_edges(proposal_edges)

    if all_edges:
        all_edge_idx = torch.tensor(all_edges, dtype=torch.long, device=device)
    else:
        all_edge_idx = torch.zeros((0, 2), dtype=torch.long, device=device)
    all_scalar_features = _scalar_features_for_edges(all_edges)

    row_edge_index, row_coeff = build_triangle_constraint_rows(n, all_edges, device=device)

    return SparsePartitionGraph(
        n_nodes=n,
        proposal_edges=proposal_edges,
        all_edges=all_edges,
        proposal_edge_idx=proposal_edge_idx,
        all_edge_idx=all_edge_idx,
        proposal_positions=proposal_positions,
        proposal_scalar_features=proposal_scalar_features,
        all_scalar_features=all_scalar_features,
        row_edge_index=row_edge_index,
        row_coeff=row_coeff,
        lexical_keys=lexical_keys,
    )


def build_triangle_constraint_rows(n_nodes: int, all_edges: Sequence[Edge], device: torch.device) -> Tuple[torch.LongTensor, torch.Tensor]:
    if not all_edges:
        return (
            torch.zeros((0, 3), dtype=torch.long, device=device),
            torch.zeros((0, 3), dtype=torch.float32, device=device),
        )
    edge_to_pos = {e: idx for idx, e in enumerate(all_edges)}
    adj: List[set[int]] = [set() for _ in range(n_nodes)]
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)

    rows_idx: List[List[int]] = []
    rows_coef: List[List[float]] = []

    for u in range(n_nodes):
        for v in sorted(x for x in adj[u] if x > u):
            common = adj[u].intersection(adj[v])
            for w in sorted(x for x in common if x > v):
                uv = edge_to_pos[(u, v)]
                uw = edge_to_pos[_edge(u, w)]
                vw = edge_to_pos[(v, w)]
                rows_idx.append([uv, uw, vw]); rows_coef.append([1.0, -1.0, -1.0])
                rows_idx.append([uw, uv, vw]); rows_coef.append([1.0, -1.0, -1.0])
                rows_idx.append([vw, uv, uw]); rows_coef.append([1.0, -1.0, -1.0])

    if not rows_idx:
        return (
            torch.zeros((0, 3), dtype=torch.long, device=device),
            torch.zeros((0, 3), dtype=torch.float32, device=device),
        )
    return (
        torch.tensor(rows_idx, dtype=torch.long, device=device),
        torch.tensor(rows_coef, dtype=torch.float32, device=device),
    )



def gold_cut_labels(cluster_ids: Sequence[int], edges: Sequence[Edge], device: torch.device) -> torch.Tensor:
    if not edges:
        return torch.zeros((0,), dtype=torch.float32, device=device)
    vals = [1.0 if int(cluster_ids[u]) != int(cluster_ids[v]) else 0.0 for (u, v) in edges]
    return torch.tensor(vals, dtype=torch.float32, device=device)


def count_triangle_violations(x: torch.Tensor, row_edge_index: torch.Tensor, row_coeff: torch.Tensor, tol: float = 1e-5) -> int:
    if row_edge_index.numel() == 0 or x.numel() == 0:
        return 0
    vals = (x[row_edge_index] * row_coeff).sum(dim=-1)
    return int((vals > tol).sum().item())


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
        root_to_id: Dict[int, int] = {}
        out: List[int] = []
        for i in range(len(self.parent)):
            r = self.find(i)
            if r not in root_to_id:
                root_to_id[r] = len(root_to_id)
            out.append(root_to_id[r])
        return out


@torch.no_grad()
def encode_signatures(
    encoder: SignatureNodeEncoder,
    tokenizer,
    signatures: Sequence[str],
    device: torch.device,
    max_length: int,
    batch_size: int = 16,
    amp: bool = False,
) -> torch.Tensor:
    was_training = encoder.training
    encoder.eval()
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
            z = encoder(input_ids=input_ids, attention_mask=attn, token_type_ids=tti)
        outs.append(z)
    if was_training:
        encoder.train()
    return torch.cat(outs, dim=0) if outs else torch.zeros((0, encoder.proj[-1].out_features if isinstance(encoder.proj, nn.Sequential) else encoder.proj.out_features), device=device)


def round_partition_labels(n_nodes: int, all_edges: Sequence[Edge], x: torch.Tensor, threshold: float = 0.5) -> List[int]:
    uf = UnionFind(n_nodes)
    if x.numel() == 0:
        return uf.labels()
    vals = x.detach().cpu().tolist()
    for (u, v), xv in zip(all_edges, vals):
        if float(xv) <= float(threshold):
            uf.union(int(u), int(v))
    return uf.labels()
