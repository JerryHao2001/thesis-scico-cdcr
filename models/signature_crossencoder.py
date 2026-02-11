# models/signature_crossencoder.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
try:
    from adapters import AutoAdapterModel  # from `adapters` package
except ImportError:
    AutoAdapterModel = None


class SignatureCorefCrossEncoder(nn.Module):
    def __init__(self, bert_model="allenai/scibert_scivocab_uncased",
                 dropout=0.1,
                 adapter_name: str | None = None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(bert_model)

        self.adapter_name = adapter_name
        if adapter_name is None:
            self.bert = AutoModel.from_pretrained(bert_model, config=self.config)
        else:
            if AutoAdapterModel is None:
                raise RuntimeError("Please `pip install -U adapters` to use SPECTER2 adapters.")
            self.bert = AutoAdapterModel.from_pretrained(bert_model, config=self.config)
            # Load and activate adapter (e.g. "allenai/specter2")
            self.bert.load_adapter(adapter_name, set_active=True)

        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        cls = out.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        logits = self.classifier(cls).squeeze(-1)

        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits, labels.float())
        return {"logits": logits, "loss": loss}
