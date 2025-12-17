#!/usr/bin/env python3
"""
Data Preprocessing Utilities for Delta-Matryoshka++ Training

This module handles TOKENIZATION ONLY:
- Multi-CPU parallel tokenization (num_proc)
- Overflowing tokens - split long texts into windows with stride
- Language filtering
- Minimum length filtering
- Saves tokenizer alongside dataset

Packing happens at TRAINING TIME in train_mlm.py (like ModernBERT).
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

logger = logging.getLogger(__name__)

try:
    import datasets
    from datasets import Dataset as HFDataset, DatasetDict
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("datasets/transformers not available")


# ============== Dataset Resolution ==============

def resolve_dataset(
    dataset_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
    split: str = "train",
    seed: int = 42,
    max_samples: Optional[int] = None,
    lang: Optional[str] = None,
    text_column: str = "text",
    num_proc: int = 8,
) -> "HFDataset":
    """
    Load and preprocess a dataset from HF hub or local disk.
    
    Args:
        dataset_name: HuggingFace hub dataset name
        dataset_path: Local dataset path (load_from_disk)
        split: Dataset split to use
        seed: Random seed for shuffling
        max_samples: Maximum samples to keep (None = all)
        lang: Language code to filter by (if dataset has 'lang' column)
        text_column: Column containing text
        num_proc: Number of processes for filtering
        
    Returns:
        Preprocessed HuggingFace Dataset
    """
    if not HF_AVAILABLE:
        raise ImportError("datasets library required")
    
    if dataset_path:
        logger.info(f"Loading dataset from disk: {dataset_path}")
        ds = datasets.load_from_disk(dataset_path)
        if isinstance(ds, DatasetDict):
            ds = ds.get(split) if split in ds else list(ds.values())[0]
    elif dataset_name:
        logger.info(f"Loading dataset from hub: {dataset_name} ({split})")
        ds = datasets.load_dataset(dataset_name, split=split)
    else:
        raise ValueError("Provide either dataset_name or dataset_path")
    
    # Language filter
    if lang and "lang" in ds.column_names:
        logger.info(f"Filtering to lang == {lang}")
        ds = ds.filter(
            lambda ex: ex.get("lang", None) == lang,
            num_proc=num_proc,
            desc="Filtering by language"
        )
    
    # Remove empty/whitespace texts
    if text_column in ds.column_names:
        ds = ds.filter(
            lambda ex: isinstance(ex[text_column], str) and ex[text_column].strip() != "",
            num_proc=num_proc,
            desc="Removing empty texts"
        )
    else:
        raise ValueError(f"Dataset must have a '{text_column}' column")
    
    # Shuffle before selecting
    logger.info(f"Shuffling with seed={seed}")
    ds = ds.shuffle(seed=seed)
    
    if max_samples and max_samples > 0:
        logger.info(f"Selecting first {max_samples} samples after shuffle")
        ds = ds.select(range(min(max_samples, len(ds))))
    
    logger.info(f"Base dataset size: {len(ds)}")
    return ds


# ============== Tokenization ==============

def tokenize_dataset(
    tokenizer,
    ds: "HFDataset",
    max_length: int = 512,
    min_length: int = 16,
    stride: int = 64,
    num_proc: int = 8,
    text_column: str = "text",
    return_overflowing_tokens: bool = True,
    batch_size: int = 1024,
) -> "HFDataset":
    """
    Tokenize dataset with multi-CPU parallelization.
    
    NO PADDING at preproc time - the trainer will pack.
    
    Args:
        tokenizer: HuggingFace tokenizer
        ds: Input dataset
        max_length: Maximum sequence length
        min_length: Minimum sequence length (filter shorter)
        stride: Overlap when splitting long texts
        num_proc: Number of processes
        text_column: Column containing text
        return_overflowing_tokens: Split long texts into multiple windows
        batch_size: Batch size for map operation
        
    Returns:
        Tokenized dataset with variable-length sequences
    """
    logger.info(f"Tokenizing with max_length={max_length}, stride={stride}, num_proc={num_proc}")
    logger.info(f"Return overflowing tokens: {return_overflowing_tokens}")
    
    def tokenize_fn(batch):
        out = tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,  # No padding at preproc time - trainer will pack
            return_attention_mask=True,
            return_overflowing_tokens=return_overflowing_tokens,
            stride=stride if return_overflowing_tokens else 0,
        )
        return {
            "input_ids": out["input_ids"],
            "attention_mask": out["attention_mask"]
        }
    
    # Tokenize in parallel
    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc="Tokenizing",
        batch_size=batch_size,
    )
    
    # Filter short sequences
    logger.info(f"Filtering sequences shorter than {min_length} tokens")
    tokenized = tokenized.filter(
        lambda ex: len(ex["input_ids"]) >= min_length,
        num_proc=num_proc,
        desc="Filtering short sequences"
    )
    
    # Set format for efficient storage
    tokenized = tokenized.with_format(type="python")
    
    logger.info(f"Finished tokenizing: {len(tokenized)} sequences")
    return tokenized


# ============== Main Preprocessing Function ==============

def pretokenize_dataset(
    dataset_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
    output_dir: str = "./data/pretokenized",
    tokenizer_name: str = "thebajajra/RexBERT-base",
    max_length: int = 512,
    min_length: int = 16,
    stride: int = 64,
    text_column: str = "text",
    num_proc: Optional[int] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
    lang: Optional[str] = None,
    return_overflowing_tokens: bool = True,
    add_mask_token: bool = False,
    seed: int = 42,
) -> str:
    """
    Pretokenize a dataset (NO PACKING - packing happens at training time).
    
    This is similar to tokenize_ds.py from rexgemma.
    
    Args:
        dataset_name: HuggingFace dataset name
        dataset_path: Local dataset path
        output_dir: Output directory
        tokenizer_name: Tokenizer to use
        max_length: Maximum sequence length
        min_length: Minimum sequence length (filter shorter)
        stride: Overlap for splitting long texts
        text_column: Column containing text
        num_proc: Number of CPU processes (default: all CPUs)
        split: Dataset split
        max_samples: Max samples to process
        lang: Language filter
        return_overflowing_tokens: Split long texts into windows
        add_mask_token: Add <mask> token if missing
        seed: Random seed
        
    Returns:
        Path to output directory
    """
    if not HF_AVAILABLE:
        raise ImportError("datasets and transformers required")
    
    # Set num_proc to all CPUs if not specified
    if num_proc is None:
        num_proc = min(32, os.cpu_count() or 8)
    
    # Reduce thread oversubscription
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Pretokenization Pipeline (tokenization only, no packing)")
    logger.info("=" * 60)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Tokenizer: {tokenizer_name}")
    logger.info(f"Max length: {max_length}")
    logger.info(f"Num processes: {num_proc}")
    logger.info(f"Return overflowing tokens: {return_overflowing_tokens}")
    
    # Load tokenizer
    logger.info(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    
    if add_mask_token and tokenizer.mask_token is None:
        logger.info("Adding <mask> token to tokenizer")
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
    
    # Load dataset
    logger.info("\nLoading dataset...")
    ds = resolve_dataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        split=split,
        seed=seed,
        max_samples=max_samples,
        lang=lang,
        text_column=text_column,
        num_proc=num_proc,
    )
    
    # Tokenize
    logger.info("\nTokenizing...")
    tokenized = tokenize_dataset(
        tokenizer=tokenizer,
        ds=ds,
        max_length=max_length,
        min_length=min_length,
        stride=stride,
        num_proc=num_proc,
        text_column=text_column,
        return_overflowing_tokens=return_overflowing_tokens,
    )
    
    # Save tokenized dataset (Arrow format - variable length sequences)
    logger.info(f"\nSaving tokenized dataset to {output_dir}")
    tokenized.save_to_disk(output_dir)
    
    # Save tokenizer alongside dataset
    tokenizer_path = os.path.join(output_dir, "tokenizer")
    logger.info(f"Saving tokenizer to {tokenizer_path}")
    tokenizer.save_pretrained(tokenizer_path)
    
    # Save config
    config = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "tokenizer_name": tokenizer_name,
        "max_length": max_length,
        "min_length": min_length,
        "stride": stride,
        "return_overflowing_tokens": return_overflowing_tokens,
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "num_sequences": len(tokenized),
        "note": "Packing happens at training time, not preprocessing",
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("Pretokenization complete!")
    logger.info("=" * 60)
    logger.info(f"Tokenized sequences: {len(tokenized)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("Note: Packing will happen at training time")
    
    return output_dir


# ============== PyTorch Dataset for Loading Tokenized Data ==============

class TokenizedMLMDataset(Dataset):
    """
    PyTorch Dataset for loading pretokenized data (variable length).
    
    This loads tokenized sequences saved via save_to_disk.
    Packing into fixed-length blocks happens at training time.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_path: Optional[str] = None,
        tokenizer_name: str = "thebajajra/RexBERT-base",
    ):
        """
        Args:
            data_path: Path to tokenized dataset (Arrow format from save_to_disk)
            tokenizer_path: Optional path to tokenizer
            tokenizer_name: Fallback tokenizer name
        """
        logger.info(f"Loading tokenized dataset from {data_path}")
        self.dataset = datasets.load_from_disk(data_path)
        
        # Handle DatasetDict
        if isinstance(self.dataset, DatasetDict):
            if "train" in self.dataset:
                self.dataset = self.dataset["train"]
            else:
                self.dataset = list(self.dataset.values())[0]
        
        # Load tokenizer for special token IDs
        tok_path = tokenizer_path or os.path.join(data_path, "tokenizer")
        if os.path.exists(tok_path):
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.pad_token_id = self.tokenizer.pad_token_id or 0
        self.sep_token_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id
        
        logger.info(f"Loaded {len(self.dataset)} variable-length sequences")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx) -> Dict[str, List[int]]:
        """Return raw tokenized sequence (no padding, no MLM masking yet)."""
        example = self.dataset[idx]
        return {
            "input_ids": example["input_ids"],
            "attention_mask": example.get("attention_mask", [1] * len(example["input_ids"])),
        }


# ============== DataLoader Factory ==============

def create_dataloader(
    data_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    tokenizer_path: Optional[str] = None,
    tokenizer_name: str = "thebajajra/RexBERT-base",
) -> DataLoader:
    """
    Create DataLoader from pretokenized data.
    
    Note: This returns variable-length sequences. For training,
    use make_packed_dataset() in train_mlm.py to pack into fixed-length blocks.
    """
    dataset = TokenizedMLMDataset(
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        tokenizer_name=tokenizer_name,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


# ============== Helper to load tokenized dataset ==============

def load_tokenized_dataset(data_path: str) -> "HFDataset":
    """
    Load a pretokenized dataset from disk.
    
    Args:
        data_path: Path to tokenized dataset (Arrow format)
        
    Returns:
        HuggingFace Dataset with input_ids and attention_mask columns
    """
    ds = datasets.load_from_disk(data_path)
    if isinstance(ds, DatasetDict):
        if "train" in ds:
            ds = ds["train"]
        else:
            ds = list(ds.values())[0]
    return ds


# ============== CLI ==============

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    parser = argparse.ArgumentParser(
        description="Pretokenize dataset for Delta-Matryoshka++ training (packing at training time)"
    )
    
    # Dataset source (mutually exclusive)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-name", type=str,
                     help="HuggingFace hub dataset name (e.g., thebajajra/Ecom-niverse)")
    src.add_argument("--dataset-path", type=str,
                     help="Local dataset path (from datasets.save_to_disk)")
    
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for tokenized data")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to process")
    
    # Tokenizer
    parser.add_argument("--tokenizer", type=str, default="thebajajra/RexBERT-base",
                        help="Tokenizer name or path")
    parser.add_argument("--add-mask-token", action="store_true",
                        help="Add <mask> token if missing")
    
    # Sampling / filtering
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Maximum samples to process (0 = all)")
    parser.add_argument("--lang", type=str, default=None,
                        help="Language code to filter by (if dataset has 'lang' column)")
    parser.add_argument("--text-column", type=str, default="text",
                        help="Column containing text")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    
    # Processing
    parser.add_argument("--num-proc", type=int, default=min(32, os.cpu_count() or 8),
                        help="Number of CPU processes for parallel tokenization")
    
    # Tokenization
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--min-length", type=int, default=16,
                        help="Minimum sequence length (filter shorter)")
    parser.add_argument("--stride", type=int, default=64,
                        help="Token overlap when splitting long texts")
    parser.add_argument("--return-overflowing-tokens", action="store_true",
                        help="Split long texts into multiple windows instead of truncating")
    
    args = parser.parse_args()
    
    pretokenize_dataset(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        min_length=args.min_length,
        stride=args.stride,
        text_column=args.text_column,
        num_proc=args.num_proc,
        split=args.split,
        max_samples=None if args.max_samples == 0 else args.max_samples,
        lang=args.lang,
        return_overflowing_tokens=args.return_overflowing_tokens,
        add_mask_token=args.add_mask_token,
        seed=args.seed,
    )
