# models/signature_crossencoder.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class SignatureCorefCrossEncoder(nn.Module):
    def __init__(self, bert_model="allenai/scibert_scivocab_uncased",
                 dropout=0.1, mlp_hidden=512, mlp_layers=1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(bert_model)
        self.bert = AutoModel.from_pretrained(bert_model, config=self.config)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        blocks = []
        in_dim = hidden
        for l in range(mlp_layers - 1):
            blocks += [nn.Linear(in_dim, mlp_hidden),
                       nn.GELU(),
                       nn.Dropout(dropout)]
            in_dim = mlp_hidden
        blocks += [nn.Linear(in_dim, 1)]
        self.head = nn.Sequential(*blocks)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        return_dict=True)
        cls = out.last_hidden_state[:, 0]
        x = self.dropout(cls)
        logits = self.head(x).squeeze(-1)
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits, labels.float())
        return {"logits": logits, "loss": loss}
