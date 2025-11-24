# models/signature_crossencoder.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class SignatureCorefCrossEncoder(nn.Module):
    """
    Binary cross-encoder: [CLS] yaml_i [SEP] yaml_j [SEP] -> score (coref logit)
    """
    def __init__(self, bert_model="allenai/scibert_scivocab_uncased", dropout=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(bert_model)
        self.bert = AutoModel.from_pretrained(bert_model, config=self.config)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        return_dict=True)
        cls = out.last_hidden_state[:, 0]  # [B, H]
        cls = self.dropout(cls)
        logits = self.classifier(cls).squeeze(-1)  # [B]
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        return {"logits": logits, "loss": loss}
