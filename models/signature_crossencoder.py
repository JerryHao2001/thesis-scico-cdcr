# models/signature_crossencoder.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

try:
    # AdapterHub's `adapters` package (aka adapter-transformers)
    from adapters import AutoAdapterModel  # type: ignore
except ImportError:  # pragma: no cover
    AutoAdapterModel = None


class SignatureCorefCrossEncoder(nn.Module):
    def __init__(
        self,
        bert_model: str = "allenai/scibert_scivocab_uncased",
        dropout: float = 0.1,
        adapter_name: str | None = None,
        mlp_hidden: int = 256,
        mlp_layers: int = 2,
    ):
        super().__init__()
        # self.config = AutoConfig.from_pretrained(bert_model)

        self.bert_model = bert_model
        self.adapter_name = adapter_name

        if adapter_name is None:
            self.bert = AutoModel.from_pretrained(bert_model)
        else:
            if AutoAdapterModel is None:
                raise RuntimeError("Please `pip install -U adapters` to use HF adapters (e.g., SPECTER2).")
            self.bert = AutoAdapterModel.from_pretrained(bert_model)
            # Load and activate adapter (e.g. "allenai/specter2")
            loaded_name = self.bert.load_adapter(self.adapter_name, source="hf", load_as="specter2", set_active=True)

            # Robust: explicitly activate what was actually loaded
            self.bert.set_active_adapters(loaded_name)

        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        if mlp_layers <= 1:
            self.head = nn.Linear(hidden, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, 1),
            )

    # ---------- adapter helpers (for freezing + optimizer grouping) ----------

    def adapter_parameters(self):
        """
        Return a list of adapter parameters (empty if no adapters / unsupported).
        For AdapterHub's `adapters` package, AutoAdapterModel exposes `adapter_parameters()`.
        """
        if hasattr(self.bert, "adapter_parameters"):
            return list(self.bert.adapter_parameters())  # type: ignore[attr-defined]
        return []

    def adapter_parameter_ids(self) -> set[int]:
        return {id(p) for p in self.adapter_parameters()}

    # ---------- forward ----------

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # pooled rep: use [CLS]
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        logit = self.head(cls).squeeze(-1)  # shape [B]
        return logit
