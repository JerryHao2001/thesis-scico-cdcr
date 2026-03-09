from __future__ import annotations

"""signature_topic_dataset.py

Topic-level dataset for antecedent / mention-ranking coreference.

Important design note (tokenizer consistency)
--------------------------------------------
The dataset does *not* tokenize signatures. It only returns raw signature
strings aligned to SciCo mentions.

However, multiple scripts in this project need a tokenizer configured with the
same "signature" special tokens (field tags + <m></m>). If different scripts
add different tokens, or add them in a different order, the model's embedding
matrix shape will not match checkpoints.

To prevent this, we define the canonical token list and helper functions here.
Training and evaluation should:
  1) build/load the tokenizer from tokenizer_name (or bert_model)
  2) call ensure_signature_special_tokens(tokenizer)
  3) call model.bert.resize_token_embeddings(len(tokenizer))
  4) only then load / train the model.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .signature_pair_dataset import build_topic_mentions, load_signatures_jsonl


# These appear in the flat signature format produced by build_signature.py.
# Keep this list stable and ordered.
SIGNATURE_SPECIAL_TOKENS: List[str] = [
    "<m>",
    "</m>",
    "<M>",
    "<CANON>",
    "<TYPE>",
    "<CTX>",
    "<ALIASES>",
    "<OUT>",
    "<IN>",
    "<CO>",
    "<CAN_CTX>",
    "<NULL_ANT>"
]


def ensure_signature_special_tokens(tokenizer: AutoTokenizer) -> AutoTokenizer:
    """Add signature special tokens to a tokenizer (idempotent).

    Must be called before resize_token_embeddings.
    """
    tokenizer.add_special_tokens({"additional_special_tokens": SIGNATURE_SPECIAL_TOKENS})
    return tokenizer


def build_signature_tokenizer(model_or_name: str, use_fast: bool = True) -> AutoTokenizer:
    """Create a tokenizer and ensure signature tokens are registered."""
    tok = AutoTokenizer.from_pretrained(model_or_name, use_fast=use_fast)
    ensure_signature_special_tokens(tok)
    return tok


class SignatureTopicDataset(Dataset):
    """Topic-level dataset.

    Each item corresponds to one SciCo topic and returns:
      - topic_id: int
      - signatures: List[str]    (aligned to SciCo row['mentions'] order)
      - cluster_ids: List[int]   (gold cluster id per mention)
      - mentions: List[List[int]]  SciCo mention tuples [pid,s,e,cid]
    """

    def __init__(
        self,
        split: str,
        signatures_path: str,
        # Deprecated / kept for backwards compatibility with older codepaths.
        bert_model: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        topics_limit: Optional[int] = None,
        seed: int = 13,
    ):
        super().__init__()
        self.split = split
        self.seed = seed

        # Intentionally unused. Tokenization is handled by training/eval scripts.
        self.bert_model = bert_model
        self.tokenizer_name = tokenizer_name

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

        # Keep original mention tuples (pid,s,e,cid) from dataset row for downstream evaluation output.
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
    """Collator for topic-level batches.

    The trainer typically uses batch_size=1. If batch_size>1,
    we return a list[topic_dict] so the trainer can process each topic independently.
    """

    def __call__(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return batch
