import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

from .signature_pair_dataset import add_special_tokens, load_signatures_jsonl, build_topic_mentions


class SignatureTopicDataset(Dataset):
    """
    Topic-level dataset for antecedent / mention-ranking coreference.

    Each item corresponds to a *SciCo topic* and provides:
      - topic_id: int
      - signatures: List[str]          (in the same order as SciCo row["mentions"])
      - cluster_ids: List[int]         (gold cluster id for each mention; same order)
      - mentions: List[List[int]]      (SciCo mention tuples: [pid, s, e, gold_cluster_id])

    Notes:
      - We rely on `load_signatures_jsonl()` and `build_topic_mentions()` to align signatures
        to SciCo mentions deterministically.
      - Tokenizer is stored for convenience and uses the same special tokens as the pair dataset.
    """
    def __init__(
        self,
        split: str,
        signatures_path: str,
        bert_model: str = "allenai/scibert_scivocab_uncased",
        topics_limit: Optional[int] = None,
        seed: int = 13,
    ):
        super().__init__()
        self.split = split
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model, use_fast=True)
        add_special_tokens(self.tokenizer, ("<m>", "</m>"))

        ds = load_dataset("allenai/scico")[split]
        if topics_limit is not None:
            ds = ds.select(range(min(topics_limit, len(ds))))

        self._ds = ds
        self._sig_map = load_signatures_jsonl(signatures_path)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._ds[idx]
        tid = int(row["id"])

        # Mentions are aligned to SciCo order by build_topic_mentions
        mentions = build_topic_mentions(row, self._sig_map)
        signatures = [m["signature"] for m in mentions]
        cluster_ids = [int(m["cluster_id"]) for m in mentions]

        # Keep original mention triples (pid,s,e,cid) from dataset row for downstream evaluation output.
        # SciCo spans are inclusive; do not alter here.
        gold_mentions = [[int(pid), int(s), int(e), int(cid)] for (pid, s, e, cid) in row["mentions"]]

        return {
            "topic_id": tid,
            "signatures": signatures,
            "cluster_ids": cluster_ids,
            "mentions": gold_mentions,
        }


@dataclass
class TopicCollator:
    """
    Collator for topic-level batches.

    The trainer typically uses batch_size=1 for simplicity. If batch_size>1,
    we return a list[topic_dict] so the trainer can process each topic independently.
    """
    def __call__(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return batch
