"""
Delta-Matryoshka++ for Text Encoders

A training recipe for nested sub-models with delta residual corrections,
prefix depth dropout, and token-conditional routing.
"""

from .model import build_model, Budget, DeltaEncoder
from .dds import DDSManager, DeltaDictionary
from .losses import (
    masked_ce_loss, kd_kl_loss, mse_loss, monotonic_upgrade_loss,
    cscf_loss, budget_penalty, compute_ece
)
from .routing import ResidualPreview, token_topk_mask
from .droppath import DropPath, SurvivalSchedule, PrefixDepthDropout
from .scheduler import PhaseScheduler, PhaseSchedulerConfig, BudgetSampler
from .weight_init import load_rexbert_to_delta
from .eval_metrics import MetricsTracker, run_full_evaluation, compute_delta_pack_size
from .hf_export import (
    DeltaMatryoshkaConfig, DeltaMatryoshkaModel, DeltaMatryoshkaForMaskedLM,
    export_to_huggingface, export_all_budgets, load_sliced_model
)
from .data_utils import (
    pretokenize_dataset, TokenizedMLMDataset, 
    create_dataloader, load_tokenized_dataset
)

__all__ = [
    # Model
    "build_model",
    "Budget", 
    "DeltaEncoder",
    
    # DDS
    "DDSManager",
    "DeltaDictionary",
    
    # Losses
    "masked_ce_loss",
    "kd_kl_loss", 
    "mse_loss",
    "monotonic_upgrade_loss",
    "cscf_loss",
    "budget_penalty",
    "compute_ece",
    
    # Routing
    "ResidualPreview",
    "token_topk_mask",
    
    # Drop path
    "DropPath",
    "SurvivalSchedule", 
    "PrefixDepthDropout",
    
    # Scheduler
    "PhaseScheduler",
    "PhaseSchedulerConfig",
    "BudgetSampler",
    
    # Weight init
    "load_rexbert_to_delta",
    
    # Evaluation
    "MetricsTracker",
    "run_full_evaluation",
    "compute_delta_pack_size",
    
    # HuggingFace Export
    "DeltaMatryoshkaConfig",
    "DeltaMatryoshkaModel",
    "DeltaMatryoshkaForMaskedLM",
    "export_to_huggingface",
    "export_all_budgets",
    "load_sliced_model",
    
    # Data Utilities (tokenization only - packing at training time)
    "pretokenize_dataset",
    "TokenizedMLMDataset",
    "create_dataloader",
    "load_tokenized_dataset",
]
