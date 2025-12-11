#!/usr/bin/env python3
"""
Export Delta-Matryoshka models to HuggingFace format.

Usage:
    # Export single budget
    python -m src.export_hf \
        --checkpoint ./ckpts/best.pt \
        --budget 0.5:6:6 \
        --output ./hf_exports/mini \
        --tokenizer thebajajra/RexBERT-base

    # Export multiple budgets
    python -m src.export_hf \
        --checkpoint ./ckpts/best.pt \
        --budgets 0.5:6:6 0.75:9:9 1.0:12:12 \
        --output ./hf_exports \
        --tokenizer thebajajra/RexBERT-base
        
    # Then load with HuggingFace:
    from transformers import AutoModel
    model = AutoModel.from_pretrained("./hf_exports/mini", trust_remote_code=True)
"""

import argparse
import os
import torch
from typing import List, Tuple

from .model import build_model
from .hf_export import export_to_huggingface, export_all_budgets


def parse_budget(budget_str: str) -> Tuple[float, int, int]:
    """Parse budget string like '0.5:6:6' to (width, heads, depth)."""
    parts = budget_str.split(":")
    if len(parts) != 3:
        raise ValueError(f"Budget must be in format 'width:heads:depth', got {budget_str}")
    return (float(parts[0]), int(parts[1]), int(parts[2]))


def main():
    parser = argparse.ArgumentParser(description="Export Delta-Matryoshka to HuggingFace format")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pt file)")
    parser.add_argument("--budget", type=str, default=None,
                        help="Single budget to export (e.g., '0.5:6:6')")
    parser.add_argument("--budgets", type=str, nargs="+", default=None,
                        help="Multiple budgets to export (e.g., '0.5:6:6' '1.0:12:12')")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="thebajajra/RexBERT-base",
                        help="Tokenizer to include with export")
    parser.add_argument("--model_name", type=str, default="delta-matryoshka",
                        help="Model name prefix for exports")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to load model on")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.budget is None and args.budgets is None:
        raise ValueError("Must specify either --budget or --budgets")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Build model from checkpoint config
    cfg = checkpoint.get("cfg", {})
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    model.to(args.device)
    
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Export
    if args.budget:
        # Single budget export
        budget = parse_budget(args.budget)
        export_to_huggingface(
            delta_model=model,
            budget=budget,
            output_dir=args.output,
            tokenizer_path=args.tokenizer,
            model_name=args.model_name
        )
    else:
        # Multiple budgets export
        budgets = [parse_budget(b) for b in args.budgets]
        export_all_budgets(
            delta_model=model,
            budgets=budgets,
            output_base_dir=args.output,
            tokenizer_path=args.tokenizer,
            model_name_prefix=args.model_name
        )
    
    print("\nExport complete!")
    print(f"Models saved to: {args.output}")
    print("\nTo use with HuggingFace:")
    print("  from transformers import AutoModel, AutoTokenizer")
    print(f'  model = AutoModel.from_pretrained("{args.output}", trust_remote_code=True)')
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{args.output}")')


if __name__ == "__main__":
    main()

