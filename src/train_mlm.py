"""
Delta-Matryoshka++ MLM Training Script

Features:
- Three-phase training (Packing -> Residualization -> Calibration)
- E-comniverse dataset support with streaming
- DDS regularization
- CSCF loss for feature alignment
- Budget controller penalty
- RexBERT weight initialization
"""

import os
import math
import random
import argparse
import time
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

from .model import build_model, Budget, DeltaEncoder
from .losses import (
    masked_ce_loss, kd_kl_loss, mse_loss, monotonic_upgrade_loss,
    cscf_loss, budget_penalty, packing_regularization, compute_ece
)
from .routing import ResidualPreview, token_topk_mask
from .scheduler import (
    PhaseScheduler, PhaseSchedulerConfig, WarmupCosineScheduler, BudgetSampler
)
from .droppath import SurvivalSchedule, PrefixDepthDropout


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Args:
    # Dataset
    dataset_name: Optional[str] = "thebajajra/Ecom-niverse"
    dataset_config: Optional[str] = None
    text_column: str = "text"
    streaming: bool = True
    shuffle_buffer: int = 10000
    
    # Pretokenized data (faster training)
    pretokenized_path: Optional[str] = None  # Path to pretokenized .pt file
    use_packing: bool = True  # Use sequence packing
    
    # Tokenizer
    tokenizer: str = "thebajajra/RexBERT-base"
    max_length: int = 512  # Standard BERT/RexBERT sequence length
    
    # Training
    batch_size: int = 64
    grad_accum: int = 1
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    train_steps: int = 100000
    eval_steps: int = 2000
    save_steps: int = 5000
    log_steps: int = 50
    output_dir: str = "./ckpts"
    fp16: bool = True
    seed: int = 1234

    # Model dims
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    layer_norm_eps: float = 1e-5
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Delta/packing
    base_ratio: float = 0.5
    base_heads: int = 6
    budget_cond_ln: bool = True
    ln_hyper_hidden: int = 64

    # Stochastic depth
    drop_path: float = 0.2
    depth_floor: int = 3

    # Routing
    enable_token_delta: bool = True
    token_delta_ratio: float = 0.35
    residual_preview_hidden: int = 128

    # Budgets
    width_budgets: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0])
    head_budgets: List[int] = field(default_factory=lambda: [4, 8, 12])
    depth_budgets: List[int] = field(default_factory=lambda: [4, 8, 12])
    sample_per_step: int = 4

    # DDS
    enable_dds: bool = True
    dds_num_atoms: int = 16
    dds_rank: int = 64

    # Loss
    enable_kd: bool = True
    alpha_kd: float = 1.0
    kd_temperature: float = 2.0
    enable_delta_depth: bool = True
    enable_mug: bool = True
    enable_cscf: bool = True
    tap_layers: List[int] = field(default_factory=lambda: [4, 8, 11])

    # Three-phase training
    enable_three_phase: bool = True
    phase_ratios: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])

    # Weight initialization (hierarchical)
    base_model_path: Optional[str] = "thebajajra/RexBERT-mini"  # Core base model
    target_model_path: Optional[str] = "thebajajra/RexBERT-base"  # Target size
    init_delta_from_target: bool = False  # Init delta slices from target weights
    
    # Legacy (for backward compat)
    init_from_rexbert: Optional[str] = None

    # Export
    export_only: bool = False
    load_path: Optional[str] = None
    export_budgets: List[str] = field(default_factory=list)


def parse_args():
    p = argparse.ArgumentParser()
    
    # Dataset
    p.add_argument("--dataset_name", type=str, default="thebajajra/Ecom-niverse")
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--streaming", action="store_true", default=True)
    p.add_argument("--no_streaming", action="store_false", dest="streaming")
    p.add_argument("--shuffle_buffer", type=int, default=10000)
    
    # Pretokenized data (faster training)
    p.add_argument("--pretokenized_path", type=str, default=None,
                   help="Path to pretokenized packed .pt file (use data_utils.py to create)")
    p.add_argument("--use_packing", action="store_true", default=True,
                   help="Use sequence packing for efficiency")
    p.add_argument("--no_packing", action="store_false", dest="use_packing")
    
    # Tokenizer
    p.add_argument("--tokenizer", type=str, default="thebajajra/RexBERT-base")
    p.add_argument("--max_length", type=int, default=512,
                   help="Maximum sequence length (512 for BERT/RexBERT, up to 2048 for extended)")
    
    # Training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--train_steps", type=int, default=100000)
    p.add_argument("--eval_steps", type=int, default=2000)
    p.add_argument("--save_steps", type=int, default=5000)
    p.add_argument("--log_steps", type=int, default=50)
    p.add_argument("--output_dir", type=str, default="./ckpts")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=1234)

    # Model dims
    p.add_argument("--vocab_size", type=int, default=30522)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=12)
    p.add_argument("--num_attention_heads", type=int, default=12)
    p.add_argument("--intermediate_size", type=int, default=3072)
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--layer_norm_eps", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--attention_dropout", type=float, default=0.1)

    # Delta/packing
    p.add_argument("--base_ratio", type=float, default=0.5)
    p.add_argument("--base_heads", type=int, default=6)
    p.add_argument("--budget_cond_ln", action="store_true")
    p.add_argument("--ln_hyper_hidden", type=int, default=64)

    # Stochastic depth
    p.add_argument("--drop_path", type=float, default=0.2)
    p.add_argument("--depth_floor", type=int, default=3)

    # Routing
    p.add_argument("--enable_token_delta", action="store_true")
    p.add_argument("--token_delta_ratio", type=float, default=0.35)
    p.add_argument("--residual_preview_hidden", type=int, default=128)

    # Budgets
    p.add_argument("--width_budgets", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    p.add_argument("--head_budgets", type=int, nargs="+", default=[4, 8, 12])
    p.add_argument("--depth_budgets", type=int, nargs="+", default=[4, 8, 12])
    p.add_argument("--sample_per_step", type=int, default=4)

    # DDS
    p.add_argument("--enable_dds", action="store_true")
    p.add_argument("--dds_num_atoms", type=int, default=16)
    p.add_argument("--dds_rank", type=int, default=64)

    # Loss
    p.add_argument("--enable_kd", action="store_true")
    p.add_argument("--alpha_kd", type=float, default=1.0)
    p.add_argument("--kd_temperature", type=float, default=2.0)
    p.add_argument("--enable_delta_depth", action="store_true")
    p.add_argument("--enable_mug", action="store_true")
    p.add_argument("--enable_cscf", action="store_true")
    p.add_argument("--tap_layers", type=int, nargs="+", default=[4, 8, 11])

    # Three-phase training
    p.add_argument("--enable_three_phase", action="store_true")
    p.add_argument("--phase_ratios", type=float, nargs=3, default=[0.33, 0.33, 0.34])

    # Weight initialization (hierarchical)
    p.add_argument("--base_model_path", type=str, default="thebajajra/RexBERT-mini",
                   help="Base model (smallest, e.g., mini) for core weights")
    p.add_argument("--target_model_path", type=str, default="thebajajra/RexBERT-base",
                   help="Target model (larger, e.g., base/large) for delta init")
    p.add_argument("--init_delta_from_target", action="store_true",
                   help="Initialize delta slices from target model weights")
    p.add_argument("--no_pretrained_init", action="store_true",
                   help="Skip pretrained weight initialization (random init)")
    
    # Legacy (backward compat)
    p.add_argument("--init_from_rexbert", type=str, default=None,
                   help="Legacy: single model path (use --base_model_path instead)")

    # Export
    p.add_argument("--export_only", action="store_true")
    p.add_argument("--load_path", type=str, default=None)
    p.add_argument("--export_budgets", type=str, nargs="+", default=None)

    args = p.parse_args()
    return args


def build_tokenizer(name: str, max_length: int):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    tok.model_max_length = max_length
    return tok


def prepare_ecomniverse(args, tokenizer):
    """Prepare E-comniverse dataset with streaming support."""
    print(f"Loading dataset: {args.dataset_name}")
    
    if args.streaming:
        ds = load_dataset(args.dataset_name, args.dataset_config, streaming=True)
        
        def tokenize_fn(examples):
            texts = examples[args.text_column]
            # Filter empty texts
            texts = [t for t in texts if t and len(t.strip()) > 0]
            if not texts:
                return {"input_ids": [], "attention_mask": []}
            return tokenizer(
                texts, 
                padding=False, 
                truncation=True, 
                max_length=args.max_length
            )
        
        ds = ds.map(tokenize_fn, batched=True, remove_columns=[args.text_column])
        
        # Shuffle with buffer
        if "train" in ds:
            ds["train"] = ds["train"].shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    else:
        ds = load_dataset(args.dataset_name, args.dataset_config)
        
        def tokenize_fn(examples):
            return tokenizer(
                examples[args.text_column], 
                padding=False, 
                truncation=True, 
                max_length=args.max_length
            )
        
        remove_cols = [c for c in ds["train"].column_names if c not in ("input_ids", "attention_mask")]
        ds = ds.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
    return ds, collator


def prepare_dataset(args, tokenizer):
    """Prepare dataset (E-comniverse or generic)."""
    if args.dataset_name is None:
        raise ValueError("Provide --dataset_name")
    
    if "ecom" in args.dataset_name.lower() or "Ecom" in args.dataset_name:
        return prepare_ecomniverse(args, tokenizer)
    
    # Generic dataset loading
    ds = load_dataset(args.dataset_name, args.dataset_config)
    
    def tok_fn(ex):
        return tokenizer(ex[args.text_column], padding=False, truncation=True, max_length=args.max_length)
    
    ds = ds.map(tok_fn, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in ("input_ids", "attention_mask")])
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
    return ds, collator


def budget_from_tuple(tpl) -> Budget:
    """Create Budget object from tuple."""
    return Budget(width=float(tpl[0]), heads=int(tpl[1]), depth=int(tpl[2]))


def main():
    args = parse_args()
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")

    tokenizer = build_tokenizer(args.tokenizer, args.max_length)
    vocab_size = tokenizer.vocab_size if tokenizer.vocab_size is not None else args.vocab_size

    # Build model
    model_cfg = dict(
        vocab_size=vocab_size,
        max_position_embeddings=args.max_position_embeddings,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        layer_norm_eps=args.layer_norm_eps,
        base_ratio=args.base_ratio,
        base_heads=args.base_heads,
        drop_path=args.drop_path,
        depth_floor=args.depth_floor,
        budget_cond_ln=args.budget_cond_ln,
        ln_hyper_hidden=args.ln_hyper_hidden,
        pad_token_id=tokenizer.pad_token_id or 0,
        enable_dds=args.enable_dds,
        dds_num_atoms=args.dds_num_atoms,
        dds_rank=args.dds_rank,
    )
    
    model = build_model(model_cfg)
    
    # Initialize from pretrained weights (hierarchical)
    if not args.export_only and not args.load_path and not getattr(args, 'no_pretrained_init', False):
        from .weight_init import init_hierarchical_delta, load_rexbert_to_delta
        
        # Legacy path
        if args.init_from_rexbert:
            print(f"[Legacy] Initializing from: {args.init_from_rexbert}")
            model = load_rexbert_to_delta(
                model, 
                args.init_from_rexbert,
                base_ratio=args.base_ratio,
                base_heads=args.base_heads
            )
        # Hierarchical path (preferred)
        elif args.base_model_path:
            print(f"Hierarchical initialization:")
            print(f"  Base model (core): {args.base_model_path}")
            if args.target_model_path:
                print(f"  Target model (delta source): {args.target_model_path}")
            model = init_hierarchical_delta(
                delta_model=model,
                base_model_path=args.base_model_path,
                target_model_path=args.target_model_path if args.init_delta_from_target else None,
                init_delta_from_target=args.init_delta_from_target,
                verbose=True
            )
    
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Routing preview head (token-level)
    preview = ResidualPreview(args.hidden_size, args.residual_preview_hidden).to(device) if args.enable_token_delta else None

    # Data - use pretokenized if available, otherwise standard loading
    if args.pretokenized_path:
        print(f"Using pretokenized data from: {args.pretokenized_path}")
        from .data_utils import PackedMLMDataset, create_dataloader
        
        train_loader = create_dataloader(
            data_path=args.pretokenized_path,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            tokenizer_name=args.tokenizer,
            mlm_probability=0.15,
        )
        train_iter = None
        eval_loader = None  # TODO: support pretokenized validation
        collator = None
        
        print(f"  Loaded {len(train_loader.dataset)} packed examples")
        print(f"  Effective sequences (with packing): ~{len(train_loader.dataset) * 3}x")
    elif args.use_packing and not args.streaming:
        # Use on-the-fly packing for non-streaming
        print("Using on-the-fly sequence packing...")
        from .data_utils import StreamingPackedDataset
        
        train_dataset = StreamingPackedDataset(
            dataset_name=args.dataset_name,
            tokenizer_name=args.tokenizer,
            max_length=args.max_length,
            text_column=args.text_column,
            mlm_probability=0.15,
            shuffle_buffer=args.shuffle_buffer,
            dataset_config=args.dataset_config,
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2)
        train_iter = None
        eval_loader = None
        collator = None
    else:
        # Standard data loading (original behavior)
        ds, collator = prepare_dataset(args, tokenizer)
        
        if args.streaming:
            # For streaming datasets, we iterate directly
            train_iter = iter(ds["train"])
            train_loader = None
        else:
            train_loader = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collator)
            train_iter = None
        
        # Validation loader (if available)
        eval_loader = None
        if not args.streaming and "validation" in ds:
            eval_loader = DataLoader(ds["validation"], batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # Phase scheduler
    phase_scheduler = None
    if args.enable_three_phase:
        phase_config = PhaseSchedulerConfig(
            total_steps=args.train_steps,
            phase_ratios=args.phase_ratios,
        )
        phase_scheduler = PhaseScheduler(phase_config)
        print("Three-phase training enabled")
    
    # LR scheduler
    lr_scheduler = WarmupCosineScheduler(
        base_lr=args.lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.train_steps,
        min_lr_ratio=0.1,
        phase_scheduler=phase_scheduler
    )

    # Budget sampler
    budget_sampler = BudgetSampler(
        width_budgets=args.width_budgets,
        head_budgets=args.head_budgets,
        depth_budgets=args.depth_budgets,
        num_samples=args.sample_per_step
    )
    
    # Survival schedule for depth dropout
    survival_schedule = SurvivalSchedule(
        num_layers=args.num_hidden_layers,
        initial_alpha=0.5 if args.enable_three_phase else 0.2
    )

    step = 0
    best_eval = float("inf")
    last_log_time = time.time()
    running_loss = 0.0
    running_metrics = {"mlm": 0.0, "kd": 0.0, "delta": 0.0, "mug": 0.0, "cscf": 0.0, "dds": 0.0}

    print(f"Starting training for {args.train_steps} steps...")
    
    # Streaming batch accumulator
    batch_buffer = []

    while step < args.train_steps:
        # Get batch
        if args.streaming:
            # Accumulate samples for batch
            while len(batch_buffer) < args.batch_size:
                try:
                    sample = next(train_iter)
                    if sample.get("input_ids"):
                        batch_buffer.append(sample)
                except StopIteration:
                    train_iter = iter(ds["train"])
            
            # Create batch from buffer
            batch_samples = batch_buffer[:args.batch_size]
            batch_buffer = batch_buffer[args.batch_size:]
            batch = collator(batch_samples)
        else:
            if train_iter is None:
                train_iter = iter(train_loader)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

        step += 1
        
        # Get phase configuration
        if phase_scheduler is not None:
            phase_cfg = phase_scheduler.get_config(step)
            survival_schedule.set_alpha(phase_cfg['alpha_depth_drop'])
            
            # Log phase transitions
            if phase_scheduler.should_log_phase_transition(step):
                print(f"\n{'='*60}")
                print(f"[Step {step}] {phase_scheduler.get_phase_summary(step)}")
                print(f"{'='*60}\n")
        else:
            phase_cfg = {
                'beta_delta': 0.7,
                'rho_mug': 0.1,
                'zeta_cscf': 0.1,
                'eta_dds_sparsity': 1e-4,
                'gamma_packing': 0.0,
                'oversample_min': True,
                'enable_tce': args.enable_token_delta,
                'target_delta_ratio': args.token_delta_ratio,
            }

        # Update LR
        current_lr = lr_scheduler.step(opt, step)
        
        # Apply survival schedule to model
        survival_schedule.apply_to_model(model)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)  # MLM labels with -100 for non-masked

        # Teacher full-budget pass
        full_budget = budget_from_tuple((max(args.width_budgets), max(args.head_budgets), max(args.depth_budgets)))
        with torch.no_grad():
            out_full = model(
                input_ids, attention_mask, budget=full_budget, 
                return_hidden=True, 
                tap_layers=args.tap_layers if args.enable_cscf else None
            )
            logits_full = out_full["logits"]
            hiddens_full = out_full.get("tap_hiddens", [])

        # Sample budgets
        picks = budget_sampler.sample(
            oversample_min=phase_cfg.get('oversample_min', True),
            include_endpoints=True
        )

        total_loss = 0.0
        batch_metrics = {"mlm": 0.0, "kd": 0.0, "delta": 0.0, "mug": 0.0, "cscf": 0.0, "dds": 0.0}
        
        for (w, h, d) in picks:
            b = budget_from_tuple((w, h, d))
            
            # Token-level delta routing
            token_delta_mask = None
            enable_tce = phase_cfg.get('enable_tce', False) and args.enable_token_delta
            
            if enable_tce:
                with torch.no_grad():
                    base_out = model(input_ids, attention_mask, budget=b)["logits"]
                    conf = base_out.softmax(-1).max(-1).values
                    
                    if args.token_delta_ratio > 0:
                        scores = -conf
                        token_delta_mask = token_topk_mask(scores, ratio=args.token_delta_ratio)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                # Student forward
                out_b = model(
                    input_ids, attention_mask, budget=b, 
                    token_delta_mask=token_delta_mask, 
                    return_delta=True,
                    tap_layers=args.tap_layers if args.enable_cscf else None
                )
                logits_b = out_b["logits"]
                logits_delta_intra = out_b["delta_logits"]
                hiddens_b = out_b.get("tap_hiddens", [])

                # Task loss (masked CE)
                loss_task = masked_ce_loss(logits_b, labels)
                batch_metrics["mlm"] += loss_task.item()

                # KD loss from full budget
                loss_kd = torch.tensor(0.0, device=device)
                if args.enable_kd and b.depth < full_budget.depth:
                    loss_kd = kd_kl_loss(
                        student_logits=logits_b[labels != -100],
                        teacher_logits=logits_full[labels != -100],
                        T=args.kd_temperature
                    )
                    batch_metrics["kd"] += loss_kd.item()

                # Delta residualization (intra-layer)
                res_logits = (logits_full - logits_b).detach()
                loss_delta = torch.tensor(0.0, device=device)
                beta_delta = phase_cfg.get('beta_delta', 0.7)
                
                if beta_delta > 0:
                    loss_delta = mse_loss(logits_delta_intra[labels != -100], res_logits[labels != -100])

                # Delta depth residualization
                if args.enable_delta_depth and d < max(args.depth_budgets):
                    logits_delta_depth = model.forward_depth_delta_only(
                        input_ids, attention_mask, base_budget=b, full_budget=full_budget
                    )
                    loss_delta = loss_delta + mse_loss(logits_delta_depth[labels != -100], res_logits[labels != -100])
                
                batch_metrics["delta"] += loss_delta.item()

                # MUG loss
                loss_mug = torch.tensor(0.0, device=device)
                rho_mug = phase_cfg.get('rho_mug', 0.0)
                if args.enable_mug and rho_mug > 0 and d < max(args.depth_budgets):
                    sup_budget = budget_from_tuple((w, h, max(args.depth_budgets)))
                    logits_sup = model(input_ids, attention_mask, budget=sup_budget)["logits"].detach()
                    loss_mug = monotonic_upgrade_loss(logits_b[labels != -100], logits_sup[labels != -100], margin=0.0)
                    batch_metrics["mug"] += loss_mug.item()

                # CSCF loss
                loss_cscf = torch.tensor(0.0, device=device)
                zeta_cscf = phase_cfg.get('zeta_cscf', 0.0)
                if args.enable_cscf and zeta_cscf > 0 and len(hiddens_b) > 0 and len(hiddens_full) > 0:
                    loss_cscf = cscf_loss(hiddens_b, hiddens_full)
                    batch_metrics["cscf"] += loss_cscf.item()

                # DDS regularization
                loss_dds = torch.tensor(0.0, device=device)
                eta_dds = phase_cfg.get('eta_dds_sparsity', 0.0)
                if args.enable_dds and eta_dds > 0:
                    loss_dds = model.get_dds_sparsity_loss()
                    batch_metrics["dds"] += loss_dds.item()

                # Packing regularization
                loss_packing = torch.tensor(0.0, device=device)
                gamma_packing = phase_cfg.get('gamma_packing', 0.0)
                if gamma_packing > 0:
                    loss_packing = packing_regularization(model, reg_type='spectral')

                # Total loss
                loss = (
                    loss_task + 
                    args.alpha_kd * loss_kd + 
                    beta_delta * loss_delta +
                    rho_mug * loss_mug +
                    zeta_cscf * loss_cscf +
                    eta_dds * loss_dds +
                    gamma_packing * loss_packing
                )

            scaler.scale(loss / max(1, args.grad_accum)).backward()
            total_loss += loss.item()

        # Gradient step
        if step % args.grad_accum == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        # Update running metrics
        running_loss += total_loss
        for k in running_metrics:
            running_metrics[k] += batch_metrics[k]

        # Logging
        if step % args.log_steps == 0:
            elapsed = time.time() - last_log_time
            last_log_time = time.time()
            
            avg_loss = running_loss / args.log_steps
            avg_metrics = {k: v / args.log_steps for k, v in running_metrics.items()}
            
            phase_str = ""
            if phase_scheduler:
                phase_str = f" [Phase {phase_scheduler.get_phase(step)+1}]"
            
            print(
                f"[step {step}]{phase_str} loss={avg_loss:.4f} "
                f"mlm={avg_metrics['mlm']:.3f} kd={avg_metrics['kd']:.3f} "
                f"delta={avg_metrics['delta']:.3f} mug={avg_metrics['mug']:.4f} "
                f"cscf={avg_metrics['cscf']:.4f} lr={current_lr:.2e} ({elapsed:.1f}s)"
            )
            
            running_loss = 0.0
            running_metrics = {k: 0.0 for k in running_metrics}

        # Evaluation
        if step % args.eval_steps == 0 and eval_loader is not None:
            model.eval()
            with torch.no_grad():
                total_eval = 0.0
                count = 0
                for eb in eval_loader:
                    ei = eb["input_ids"].to(device)
                    em = eb["attention_mask"].to(device)
                    el = eb["labels"].to(device)
                    out = model(ei, em, budget=full_budget)
                    total_eval += masked_ce_loss(out["logits"], el).item()
                    count += 1
                    if count >= 100:  # Limit eval batches
                        break
                
                ppl = math.exp(total_eval / max(1, count))
                print(f"[eval @ {step}] MLM ppl={ppl:.3f}")
                
                if ppl < best_eval:
                    best_eval = ppl
                    spath = os.path.join(args.output_dir, "best.pt")
                    torch.save({"model": model.state_dict(), "cfg": model_cfg, "step": step}, spath)
                    print(f"  saved best to {spath}")
            model.train()

        # Checkpoint
        if step % args.save_steps == 0:
            spath = os.path.join(args.output_dir, f"step_{step}.pt")
            torch.save({"model": model.state_dict(), "cfg": model_cfg, "step": step}, spath)
            print(f"  checkpoint -> {spath}")
            
            # Export DDS pack if enabled
            if args.enable_dds:
                dds_path = os.path.join(args.output_dir, f"dds_pack_step_{step}.pt")
                torch.save(model.export_delta_pack(), dds_path)
                print(f"  DDS pack -> {dds_path}")

        if step >= args.train_steps:
            break

    # Final save
    spath = os.path.join(args.output_dir, "final.pt")
    torch.save({"model": model.state_dict(), "cfg": model_cfg, "step": step}, spath)
    print(f"Training complete! Final checkpoint -> {spath}")

    # Export-only path
    if args.export_only:
        if args.load_path is None or args.export_budgets is None:
            raise ValueError("--export_only requires --load_path and --export_budgets like 0.5:4:6 1.0:12:12")
        
        ckpt = torch.load(args.load_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        
        os.makedirs(args.output_dir, exist_ok=True)
        for spec in args.export_budgets:
            w, h, d = spec.split(":")
            fname = os.path.join(args.output_dir, f"delta_m_{w}_{h}_{d}.pt")
            torch.save({
                "model": model.state_dict(), 
                "budget": (float(w), int(h), int(d)), 
                "cfg": ckpt["cfg"]
            }, fname)
            print(f"Exported budget {spec} -> {fname}")
        
        # Export DDS packs if enabled
        if args.enable_dds:
            dds_path = os.path.join(args.output_dir, "dds_atoms.pt")
            torch.save({
                "dds_fc1_U": model.dds_manager.dds_fc1.atom_U.data,
                "dds_fc1_V": model.dds_manager.dds_fc1.atom_V.data,
                "dds_fc2_U": model.dds_manager.dds_fc2.atom_U.data,
                "dds_fc2_V": model.dds_manager.dds_fc2.atom_V.data,
                "dds_qkv_U": model.dds_manager.dds_qkv.atom_U.data,
                "dds_qkv_V": model.dds_manager.dds_qkv.atom_V.data,
                "dds_out_U": model.dds_manager.dds_out.atom_U.data,
                "dds_out_V": model.dds_manager.dds_out.atom_V.data,
            }, dds_path)
            print(f"Exported DDS atom banks -> {dds_path}")
            
            delta_pack_path = os.path.join(args.output_dir, "dds_coeffs.pt")
            torch.save(model.export_delta_pack(), delta_pack_path)
            print(f"Exported DDS coefficients -> {delta_pack_path}")


if __name__ == "__main__":
    main()
