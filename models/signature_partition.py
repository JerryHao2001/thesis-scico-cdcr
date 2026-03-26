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
    all_edge_idx: torch.LongTensor     # [E, 2]
    row_edge_index: torch.LongTensor   # [R, 3]
    row_coeff: torch.Tensor            # [R, 3]
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
        node_proj: str = "none",
    ):
        super().__init__()
        self.bert_model = bert_model
        self.adapter_name = adapter_name
        self.node_proj = node_proj

        if adapter_name is None:
            self.bert = AutoModel.from_pretrained(bert_model)
        else:
            if AutoAdapterModel is None:
                raise RuntimeError("Please `pip install -U adapters` to use HF adapters.")
            self.bert = AutoAdapterModel.from_pretrained(bert_model)
            loaded_name = self.bert.load_adapter(adapter_name, source="hf", load_as="sigpart", set_active=True)
            self.bert.set_active_adapters(loaded_name)

        hidden = int(self.bert.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden if node_proj == "none" else proj_dim

        if node_proj == "none":
            self.proj = nn.Identity()
        elif node_proj == "linear":
            self.proj = nn.Linear(hidden, proj_dim)
        elif node_proj == "mlp":
            if proj_layers <= 1:
                self.proj = nn.Linear(hidden, proj_dim)
            else:
                self.proj = nn.Sequential(
                    nn.Linear(hidden, proj_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(proj_hidden, proj_dim),
                )
        else:
            raise ValueError(f"Unknown node_proj={node_proj!r}")

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


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class ResidualEdgeScorer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        init_bias: float = -0.1,
    ):
        super().__init__()
        self.node_ln = nn.LayerNorm(node_dim)
        in_dim = 4 * node_dim
        self.input = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim, dropout) for _ in range(max(0, num_layers - 1))])
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.out.bias, float(init_bias))

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        h_i = self.node_ln(h_i)
        h_j = self.node_ln(h_j)
        pair = torch.cat([h_i, h_j, torch.abs(h_i - h_j), h_i * h_j], dim=-1)
        h = F.gelu(self.input(pair))
        h = self.dropout(h)
        for block in self.blocks:
            h = block(h)
        h = self.out_ln(h)
        h = self.dropout(h)
        return self.out(h).squeeze(-1)


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SignaturePartitionModel(nn.Module):
    def __init__(
        self,
        bert_model: str = "allenai/scibert_scivocab_uncased",
        adapter_name: Optional[str] = None,
        dropout: float = 0.1,
        proj_hidden: int = 512,
        proj_dim: int = 256,
        proj_layers: int = 2,
        node_proj: str = "none",
        edge_hidden: int = 512,
        edge_layers: int = 3,
        edge_init_bias: float = -0.1,
        ceaf_slots_cap: int = 32,
        ceaf_head_hidden: int = 128,
        threshold_head_hidden: int = 64,
        threshold_stats_dim: int = 9,
    ):
        super().__init__()
        self.encoder = SignatureNodeEncoder(
            bert_model=bert_model,
            adapter_name=adapter_name,
            dropout=dropout,
            proj_hidden=proj_hidden,
            proj_dim=proj_dim,
            proj_layers=proj_layers,
            node_proj=node_proj,
        )
        self.node_dim = int(self.encoder.output_dim)
        self.edge_scorer = ResidualEdgeScorer(
            node_dim=self.node_dim,
            hidden_dim=edge_hidden,
            num_layers=edge_layers,
            dropout=dropout,
            init_bias=edge_init_bias,
        )
        self.ceaf_slots_cap = int(ceaf_slots_cap)
        self.cluster_assign_head = TinyMLP(self.node_dim, ceaf_head_hidden, self.ceaf_slots_cap, dropout=dropout)
        self.threshold_head = TinyMLP(threshold_stats_dim, threshold_head_hidden, 1, dropout=dropout)

    def score_edges(self, node_embs: torch.Tensor, edge_idx: torch.Tensor) -> torch.Tensor:
        if edge_idx.numel() == 0:
            return node_embs.new_zeros((0,))
        h_i = node_embs[edge_idx[:, 0]]
        h_j = node_embs[edge_idx[:, 1]]
        return self.edge_scorer(h_i, h_j)

    def cluster_assignment_logits(self, node_states: torch.Tensor, n_slots: int) -> torch.Tensor:
        n_slots = max(1, min(int(n_slots), int(self.ceaf_slots_cap)))
        return self.cluster_assign_head(node_states)[:, :n_slots]

    def predict_threshold(self, stats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.threshold_head(stats)).squeeze(-1)


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

    def _precondition(self, num_edges: int, row_edge_index: torch.Tensor, row_coeff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_rows = int(row_edge_index.size(0))
        if n_rows == 0:
            return (
                row_coeff.new_ones((num_edges,), dtype=torch.float32),
                row_coeff.new_ones((0,), dtype=torch.float32),
            )
        coeff_abs = row_coeff.abs().float()
        row_sum = coeff_abs.sum(dim=-1).clamp_min(1.0)
        col_sum = torch.zeros((num_edges,), dtype=torch.float32, device=row_coeff.device)
        col_sum.index_add_(0, row_edge_index.reshape(-1), coeff_abs.reshape(-1))
        col_sum = col_sum.clamp_min(1.0)
        tau = self.step_scale / col_sum
        sigma = self.step_scale / row_sum
        return tau, sigma

    def forward(
        self,
        weights: torch.Tensor,
        row_edge_index: torch.Tensor,
        row_coeff: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        if weights.numel() == 0:
            x0 = weights.new_zeros((0,), dtype=torch.float32)
            diag = {
                "objective": 0.0,
                "mean_abs_dx": 0.0,
                "max_abs_dx": 0.0,
                "viol_frac": 0.0,
                "viol_mean": 0.0,
                "viol_max": 0.0,
                "tau_mean": 1.0,
                "sigma_mean": 1.0,
            }
            return (x0, diag) if return_diagnostics else x0

        w = weights.float()
        rei = row_edge_index.long()
        rcf = row_coeff.float()
        mu = max(self.mu, 1e-6)
        tau, sigma = self._precondition(w.numel(), rei, rcf)

        x = torch.clamp(-w / mu, 0.0, 1.0)
        x_bar = x.clone()
        p = torch.zeros((rei.size(0),), dtype=torch.float32, device=w.device)
        mean_abs_dx = 0.0
        max_abs_dx = 0.0

        if rei.numel() == 0:
            for _ in range(self.num_iters):
                x_next = torch.clamp((x - tau * w) / (1.0 + tau * mu), 0.0, 1.0)
                dx = (x_next - x).abs()
                mean_abs_dx = float(dx.mean().item())
                max_abs_dx = float(dx.max().item())
                x = x_next
            diag = {
                "objective": float((w * x + 0.5 * mu * x.pow(2)).sum().item()),
                "mean_abs_dx": mean_abs_dx,
                "max_abs_dx": max_abs_dx,
                "viol_frac": 0.0,
                "viol_mean": 0.0,
                "viol_max": 0.0,
                "tau_mean": float(tau.mean().item()),
                "sigma_mean": 1.0,
            }
            return (x, diag) if return_diagnostics else x

        for _ in range(self.num_iters):
            ax = (x_bar[rei] * rcf).sum(dim=-1)
            p = torch.relu(p + sigma * ax)

            atp = torch.zeros_like(w)
            contrib = (rcf * p.unsqueeze(-1)).reshape(-1)
            atp.index_add_(0, rei.reshape(-1), contrib)

            x_next = (x - tau * (atp + w)) / (1.0 + tau * mu)
            x_next = torch.clamp(x_next, 0.0, 1.0)
            dx = (x_next - x).abs()
            mean_abs_dx = float(dx.mean().item())
            max_abs_dx = float(dx.max().item())
            x_bar = x_next + self.theta * (x_next - x)
            x = x_next

        vals = (x[rei] * rcf).sum(dim=-1)
        pos = torch.relu(vals)
        viol_frac = float((pos > 1e-5).float().mean().item()) if pos.numel() > 0 else 0.0
        viol_mean = float(pos.mean().item()) if pos.numel() > 0 else 0.0
        viol_max = float(pos.max().item()) if pos.numel() > 0 else 0.0
        diag = {
            "objective": float((w * x + 0.5 * mu * x.pow(2)).sum().item()),
            "mean_abs_dx": mean_abs_dx,
            "max_abs_dx": max_abs_dx,
            "viol_frac": viol_frac,
            "viol_mean": viol_mean,
            "viol_max": viol_max,
            "tau_mean": float(tau.mean().item()),
            "sigma_mean": float(sigma.mean().item()) if sigma.numel() > 0 else 1.0,
        }
        return (x, diag) if return_diagnostics else x


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
    del mentions
    device = node_embs.device
    n = int(node_embs.size(0))
    proposal: set[Edge] = set()
    lexical_keys = [extract_signature_key(s) for s in signatures]

    if n > 1 and semantic_k > 0:
        z = F.normalize(node_embs.detach().float(), dim=-1)
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
    all_edge_idx = torch.tensor(all_edges, dtype=torch.long, device=device) if all_edges else torch.zeros((0, 2), dtype=torch.long, device=device)
    row_edge_index, row_coeff = build_triangle_constraint_rows(n, all_edges, device=device)
    return SparsePartitionGraph(
        n_nodes=n,
        proposal_edges=proposal_edges,
        all_edges=all_edges,
        all_edge_idx=all_edge_idx,
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


def triangle_violation_stats(x: torch.Tensor, row_edge_index: torch.Tensor, row_coeff: torch.Tensor, tol: float = 1e-5) -> Dict[str, float]:
    if x.numel() == 0 or row_edge_index.numel() == 0:
        return {"viol_frac": 0.0, "viol_mean": 0.0, "viol_max": 0.0}
    vals = (x[row_edge_index.long()] * row_coeff.float()).sum(dim=-1)
    pos = torch.relu(vals - tol)
    return {
        "viol_frac": float((pos > 0).float().mean().item()),
        "viol_mean": float(pos.mean().item()),
        "viol_max": float(pos.max().item()),
    }


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
        nxt = 0
        for i in range(len(self.parent)):
            r = self.find(i)
            if r not in root_to_id:
                root_to_id[r] = nxt
                nxt += 1
            out.append(root_to_id[r])
        return out


def round_partition_labels(n_nodes: int, edges: Sequence[Edge], x: torch.Tensor, threshold: float) -> List[int]:
    uf = UnionFind(n_nodes)
    if x.numel() > 0:
        xv = x.detach().float().cpu().tolist()
        for (u, v), cut in zip(edges, xv):
            if float(cut) <= float(threshold):
                uf.union(int(u), int(v))
    return uf.labels()
