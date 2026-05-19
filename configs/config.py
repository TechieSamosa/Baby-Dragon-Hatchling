"""
Configuration dataclasses for BDH architecture experiments.

Hyperparameters match Table 2 of the paper:
    V=256, L=6, d=256, H=4, T=128, B=8, m=128 (base) / 32 (lowdim),
    FFN=1024, dropout=0.1, AdamW(lr=1e-3, wd=0.1), 2900 steps, seed=42.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Architecture hyperparameters shared across all BDH variants."""

    vocab_size: int = 256                       # V  — byte-level vocabulary
    n_layer: int = 6                            # L  — number of repeated layers
    n_embd: int = 256                           # d  — embedding dimension
    n_head: int = 4                             # H  — attention heads
    block_size: int = 128                       # T  — context window length
    mlp_internal_dim_multiplier: int = 128      # m  — latent multiplier (BDH)
    ffn_dim: int = 1024                         # 4d — Transformer FFN dimension
    dropout: float = 0.1                        # p  — dropout probability


@dataclass
class TrainConfig:
    """Training-loop hyperparameters."""

    batch_size: int = 8                         # B
    max_iters: int = 2_900                      # total training steps
    learning_rate: float = 1e-3                 # η  for AdamW
    weight_decay: float = 0.1                   # λ  for AdamW
    log_freq: int = 100                         # log every N steps
    eval_iters: int = 20                        # validation batches per eval
    seed: int = 42                              # reproducibility seed
    device: str = "auto"                        # "auto" | "cuda" | "cpu"
    log_file: str = "results/log.jsonl"         # JSONL output path


# ---------------------------------------------------------------------------
# Convenience: pre-built configs for every variant listed in Table 1
# ---------------------------------------------------------------------------

def get_model_config(model_name: str) -> ModelConfig:
    """Return a ModelConfig appropriate for *model_name*."""
    cfg = ModelConfig()
    if model_name == "bdh_lowdim":
        cfg.mlp_internal_dim_multiplier = 32    # reduced latent multiplier
    return cfg


def get_train_config() -> TrainConfig:
    """Return the default TrainConfig used across all experiments."""
    return TrainConfig()
