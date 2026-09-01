"""Role-aware residual item representations for the three main-table methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from seqrec.models.SASRec._model import SASRec
from seqrec.modules import gather_indexes, get_attention_mask


@dataclass(frozen=True)
class GateConfig:
    mode: Literal["off", "identity", "density"]
    tau: float | None = None
    gamma: float | None = None


def compute_gate(counts: torch.Tensor, config: GateConfig) -> torch.Tensor:
    counts = counts.to(torch.float32)
    if config.mode == "off":
        return torch.zeros_like(counts)
    if config.mode == "identity":
        return torch.ones_like(counts)
    if config.mode != "density":
        raise ValueError(config.mode)
    if config.tau is None or config.tau <= 0:
        raise ValueError("density gate requires tau > 0")
    if config.gamma is None or config.gamma <= 0:
        raise ValueError("density gate requires gamma > 0")
    gate = (counts / (counts + config.tau)).pow(config.gamma)
    return torch.where(counts > 0, gate, torch.zeros_like(gate))


class RoleDensitySASRec(SASRec):
    """SASRec with separate history-side and candidate-side residual gates."""

    def __init__(
        self,
        config: dict,
        pretrained_item_embeddings: torch.Tensor,
        positive_support: torch.Tensor,
        history_support: torch.Tensor,
        history_gate_config: GateConfig,
        candidate_gate_config: GateConfig,
    ) -> None:
        super().__init__(config, pretrained_item_embeddings)
        if positive_support.shape != history_support.shape:
            raise ValueError("role supports must have identical shapes")
        if positive_support.shape[0] != config["item_num"] + 1:
            raise ValueError("support and catalog sizes do not match")

        self.residual_embeddings = nn.Embedding(
            config["item_num"] + 1,
            config["hidden_size"],
            padding_idx=0,
        )
        nn.init.zeros_(self.residual_embeddings.weight)
        self.register_buffer("positive_support", positive_support.to(torch.long), persistent=True)
        self.register_buffer("history_support", history_support.to(torch.long), persistent=True)
        self.register_buffer(
            "history_gate",
            compute_gate(history_support, history_gate_config),
            persistent=True,
        )
        self.register_buffer(
            "candidate_gate",
            compute_gate(positive_support, candidate_gate_config),
            persistent=True,
        )

    def base_embedding_provider(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_embeddings(item_ids)

    def residual_embedding_provider(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.residual_embeddings(item_ids)

    def base_candidate_table(self) -> torch.Tensor:
        return self.item_embeddings.weight.data

    def residual_table(self) -> torch.Tensor:
        return self.residual_embeddings.weight

    def history_vectors(self, item_ids: torch.Tensor) -> torch.Tensor:
        base = self.base_embedding_provider(item_ids)
        residual = self.residual_embedding_provider(item_ids)
        weight = self.history_gate[item_ids].masked_fill(item_ids == 0, 0)
        return base + weight.unsqueeze(-1) * residual

    def candidate_vectors(self, catalog_ids: torch.Tensor | None = None) -> torch.Tensor:
        if catalog_ids is None:
            base = self.base_candidate_table()
            residual = self.residual_table()
            gate = self.candidate_gate
        else:
            base = self.base_embedding_provider(catalog_ids)
            residual = self.residual_embedding_provider(catalog_ids)
            gate = self.candidate_gate[catalog_ids]
        return base + gate.unsqueeze(-1) * residual

    def get_embeddings(self, items: torch.Tensor) -> torch.Tensor:
        return self.history_vectors(items)

    def get_all_embeddings(self, device=None) -> torch.Tensor:
        return self.candidate_vectors()

    def get_representation(self, batch: dict) -> torch.Tensor:
        vectors = self.history_vectors(batch["item_seqs"])
        vectors = vectors + self.positional_embeddings(
            torch.arange(self.config["max_seq_length"], device=vectors.device)
        )
        sequence = self.emb_dropout(vectors)
        attention = get_attention_mask((batch["item_seqs"] != 0).float(), bidirectional=False)
        sequence = self.transformer_encoder(sequence, attention_mask=attention)
        return gather_indexes(sequence[-1], batch["seq_lengths"] - 1)

    def forward(self, batch: dict) -> dict:
        query = self.get_representation(batch).view(-1, self.config["hidden_size"])
        logits = query @ self.candidate_vectors().T
        return {"loss": self.loss_func(logits, batch["labels"].view(-1))}

    def predict(self, batch: dict, n_return_sequences: int = 1) -> torch.Tensor:
        query = self.get_representation(batch).view(-1, self.config["hidden_size"])
        scores = query @ self.candidate_vectors().T
        start, end = self.config["select_pool"]
        return scores[:, start:end].topk(n_return_sequences, dim=1).indices + start
