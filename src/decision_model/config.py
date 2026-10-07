"""Training settings shared by the scratch net and the pretrained backbones."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    hidden_size: int = 64
    n_layers: int = 2
    n_heads: int = 4
    proj_size: int = 32
    max_len: int = 256
    epochs: int = 12
    batch_size: int = 32
    lr: float = 2e-3
    seed: int = 0
    name: str = "pointer-tiny"
    model: str = "scratch"
    model_id: str = ""
