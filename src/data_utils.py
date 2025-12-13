#!/usr/bin/env python3
"""
Data Preprocessing Utilities for Delta-Matryoshka++ Training

Features:
1. Multi-CPU parallel tokenization (num_proc)
2. Sequence packing - concatenate short sequences to minimize padding
3. Overflowing tokens - split long texts into windows with stride
4. Language filtering
5. Minimum length filtering
6. Saves tokenizer alongside dataset

Based on efficient preprocessing patterns for large-scale MLM training.
"""

import os
import json
import random
import logging
from typing import Optional, List, Dict, Iterator, Any
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback tqdm that does nothing
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
        Tokenized dataset
    """
    logger.info(f"Tokenizing with max_length={max_length}, stride={stride}, num_proc={num_proc}")
    logger.info(f"Return overflowing tokens: {return_overflowing_tokens}")
    
    def tokenize_fn(batch):
        out = tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,  # No padding at preproc time
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
    
    # Remove text column if still present
    if text_column in tokenized.column_names:
        tokenized = tokenized.remove_columns([text_column])
    
    logger.info(f"Finished tokenizing: {len(tokenized)} sequences")
    return tokenized


# ============== Sequence Packing ==============

@dataclass
class PackedExample:
    """A packed example containing multiple concatenated sequences."""
    input_ids: List[int]
    attention_mask: List[int]
    sequence_boundaries: List[int]
    document_ids: Optional[List[int]] = None  # Which document each token belongs to


def create_block_diagonal_mask(document_ids: List[int], max_length: int) -> List[List[int]]:
    """
    Create a 2D block-diagonal attention mask from document IDs.
    
    This prevents cross-attention between different documents in packed sequences,
    similar to ModernBERT's approach.
    
    Args:
        document_ids: List of document IDs for each token position
        max_length: Maximum sequence length
        
    Returns:
        2D attention mask where mask[i][j] = 1 if tokens i and j can attend to each other
    """
    seq_len = len(document_ids)
    mask = [[0] * max_length for _ in range(max_length)]
    
    for i in range(seq_len):
        for j in range(seq_len):
            # Tokens can attend to each other only if they're in the same document
            if document_ids[i] == document_ids[j]:
                mask[i][j] = 1
    
    return mask


def create_document_ids_from_boundaries(boundaries: List[int], total_length: int) -> List[int]:
    """
    Convert sequence boundaries to document IDs.
    
    Example:
        boundaries = [10, 25, 40]  # End positions of docs 0, 1, 2
        total_length = 50  # With padding
        Returns: [0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                 ^--- doc 0 (10 tokens) ^--- doc 1 (15 tokens)        ^--- doc 2 (15 tokens)        ^--- padding (-1)
    """
    document_ids = []
    prev_boundary = 0
    
    for doc_id, boundary in enumerate(boundaries):
        # Add doc_id for tokens between prev_boundary and boundary
        document_ids.extend([doc_id] * (boundary - prev_boundary))
        prev_boundary = boundary
    
    # Add -1 for padding tokens (they shouldn't attend to anything)
    if len(document_ids) < total_length:
        document_ids.extend([-1] * (total_length - len(document_ids)))
    
    return document_ids[:total_length]


class SequencePacker:
    """
    Packs multiple short sequences into single max_length sequences.
    Significantly reduces padding waste and improves training efficiency.
    """
    
    def __init__(
        self,
        max_length: int = 512,
        sep_token_id: int = 102,
        cls_token_id: int = 101,
        pad_token_id: int = 0,
        add_special_tokens: bool = True
    ):
        self.max_length = max_length
        self.sep_token_id = sep_token_id
        self.cls_token_id = cls_token_id
        self.pad_token_id = pad_token_id
        self.add_special_tokens = add_special_tokens
        
        self._buffer_ids = []
        self._buffer_boundaries = []
        self._current_length = 0
    
    def add_sequence(self, input_ids: List[int]) -> Optional[PackedExample]:
        """Add a sequence to the buffer. Returns packed example if buffer is full."""
        # Remove existing special tokens if present
        if input_ids and input_ids[0] == self.cls_token_id:
            input_ids = input_ids[1:]
        if input_ids and input_ids[-1] == self.sep_token_id:
            input_ids = input_ids[:-1]
        
        seq_len = len(input_ids)
        if self.add_special_tokens:
            seq_len += 1  # SEP between sequences
        
        if self._current_length + seq_len > self.max_length:
            packed = self._flush_buffer()
            self._buffer_ids = list(input_ids)
            self._buffer_boundaries = [len(input_ids)]
            self._current_length = len(input_ids)
            return packed
        
        if self._buffer_ids and self.add_special_tokens:
            self._buffer_ids.append(self.sep_token_id)
            self._current_length += 1
        
        self._buffer_ids.extend(input_ids)
        self._buffer_boundaries.append(len(self._buffer_ids))
        self._current_length = len(self._buffer_ids)
        
        return None
    
    def _flush_buffer(self) -> Optional[PackedExample]:
        if not self._buffer_ids:
            return None
        
        ids = self._buffer_ids
        boundaries = self._buffer_boundaries.copy()
        
        if self.add_special_tokens:
            ids = [self.cls_token_id] + ids
            # Adjust boundaries for the added CLS token
            boundaries = [b + 1 for b in boundaries]
        
        pad_len = self.max_length - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [self.pad_token_id] * pad_len
        
        # Create document IDs for block-diagonal attention masking
        document_ids = create_document_ids_from_boundaries(boundaries, self.max_length)
        
        packed = PackedExample(
            input_ids=ids,
            attention_mask=attention_mask,
            sequence_boundaries=boundaries,
            document_ids=document_ids,
        )
        
        self._buffer_ids = []
        self._buffer_boundaries = []
        self._current_length = 0
        
        return packed
    
    def flush(self) -> Optional[PackedExample]:
        return self._flush_buffer()


def pack_tokenized_dataset(
    tokenized_dataset: "HFDataset",
    max_length: int = 512,
    tokenizer = None,
    sep_token_id: int = 102,
    cls_token_id: int = 101,
    pad_token_id: int = 0,
    output_path: Optional[str] = None,
    save_interval: int = 100000,
    num_proc: Optional[int] = None,  # Auto-detect CPU count
    batch_size: int = 50000,  # Larger batches = better packing efficiency
) -> Dict[str, Any]:
    """
    Pack tokenized sequences into fixed-length examples with parallel processing.
    
    Uses HuggingFace Datasets' .map() with num_proc for parallelization.
    Large batch sizes are used for better packing efficiency.
    
    Args:
        tokenized_dataset: Dataset with input_ids column
        max_length: Target packed length
        tokenizer: Optional tokenizer to get special token IDs
        sep_token_id: SEP token ID (used if tokenizer not provided)
        cls_token_id: CLS token ID
        pad_token_id: PAD token ID
        output_path: If provided, save directly to disk (memory efficient)
        save_interval: Save progress every N packed examples
        num_proc: Number of processes for parallel packing (default: auto-detect CPUs)
        batch_size: Batch size for map operation (larger = better packing, more memory)
        
    Returns:
        Dict with 'input_ids', 'attention_mask', 'document_ids' as numpy arrays
    """
    import numpy as np
    from tqdm import tqdm
    
    if tokenizer:
        sep_token_id = tokenizer.sep_token_id or sep_token_id
        cls_token_id = tokenizer.cls_token_id or cls_token_id
        pad_token_id = tokenizer.pad_token_id or pad_token_id
    
    # Auto-detect num_proc if not specified
    if num_proc is None:
        num_proc = min(24, os.cpu_count() or 8)
        logger.info(f"Auto-detected {num_proc} CPUs for parallel packing")
    
    if num_proc > 1:
        # Parallel batched packing using .map() with optimized flattening
        logger.info(f"Packing sequences with {num_proc} processes (batch_size={batch_size})...")
        
        def pack_batch(batch, indices):
            """Pack a batch of sequences using greedy bin-packing.
            
            Returns flattened results - each packed sequence as a separate row.
            This avoids the expensive Python-level flattening step.
            """
            input_ids_list = batch["input_ids"]
            
            packer = SequencePacker(
                max_length=max_length,
                sep_token_id=sep_token_id,
                cls_token_id=cls_token_id,
                pad_token_id=pad_token_id,
            )
            
            packed_ids = []
            packed_masks = []
            packed_doc_ids = []
            
            for input_ids in input_ids_list:
                if not input_ids:
                    continue
                
                packed = packer.add_sequence(input_ids)
                if packed:
                    packed_ids.append(packed.input_ids)
                    packed_masks.append(packed.attention_mask)
                    packed_doc_ids.append(packed.document_ids)
            
            # Flush remaining buffer
            final = packer.flush()
            if final:
                packed_ids.append(final.input_ids)
                packed_masks.append(final.attention_mask)
                packed_doc_ids.append(final.document_ids)
            
            # Return as flattened structure - each packed example is a row
            return {
                "input_ids": packed_ids,
                "attention_mask": packed_masks,
                "document_ids": packed_doc_ids,
            }
        
        # Apply packing in parallel batches
        # Use large batch sizes for efficiency
        effective_batch_size = max(batch_size, 50000)  # Process large batches for better packing
        
        packed_dataset = tokenized_dataset.map(
            pack_batch,
            batched=True,
            batch_size=effective_batch_size,
            num_proc=num_proc,
            remove_columns=tokenized_dataset.column_names,
            desc="Packing sequences",
            with_indices=True,  # For debugging if needed
            # Control memory
            writer_batch_size=max(1000, effective_batch_size // 50),
        )
        
        # Fast flattening using numpy - the dataset now has lists of packed sequences per row
        # We need to flatten to have one packed sequence per row
        logger.info("Flattening packed sequences (fast path)...")
        
        # Count total packed examples first
        total_packed = sum(len(ex["input_ids"]) for ex in packed_dataset)
        logger.info(f"Total packed examples: {total_packed}")
        
        # Pre-allocate arrays
        packed_input_ids = np.zeros((total_packed, max_length), dtype=np.int32)
        packed_attention_mask = np.zeros((total_packed, max_length), dtype=np.int8)
        packed_document_ids = np.zeros((total_packed, max_length), dtype=np.int16)
        
        # Fill arrays - this is I/O bound so single-threaded is fine
        current_idx = 0
        for example in tqdm(packed_dataset, desc="Collecting", unit="batch"):
            batch_ids = example["input_ids"]
            batch_masks = example["attention_mask"]
            batch_doc_ids = example["document_ids"]
            
            for ids, mask, doc_ids in zip(batch_ids, batch_masks, batch_doc_ids):
                # Ensure correct length (pad if needed)
                ids_len = len(ids)
                if ids_len < max_length:
                    packed_input_ids[current_idx, :ids_len] = ids
                    packed_attention_mask[current_idx, :ids_len] = mask[:ids_len]
                    packed_document_ids[current_idx, :len(doc_ids)] = doc_ids
                    # Padding already zeros from initialization
                else:
                    packed_input_ids[current_idx] = ids[:max_length]
                    packed_attention_mask[current_idx] = mask[:max_length]
                    packed_document_ids[current_idx] = doc_ids[:max_length]
                current_idx += 1
        
        # Trim to actual size (in case of over-estimation)
        packed_input_ids = packed_input_ids[:current_idx]
        packed_attention_mask = packed_attention_mask[:current_idx]
        packed_document_ids = packed_document_ids[:current_idx]
        
        num_packed = current_idx
        total_seqs = len(tokenized_dataset)
        
    else:
        # Sequential packing (original implementation, memory efficient for small datasets)
        logger.info("Packing sequences sequentially...")
        
        packer = SequencePacker(
            max_length=max_length,
            sep_token_id=sep_token_id,
            cls_token_id=cls_token_id,
            pad_token_id=pad_token_id,
        )
        
        # Pre-allocate numpy arrays (estimate capacity)
        total_seqs = len(tokenized_dataset) if hasattr(tokenized_dataset, '__len__') else None
        if total_seqs:
            estimated_packed = max(1000, total_seqs // 3)
        else:
            estimated_packed = 1000000
        
        chunk_size = min(100000, estimated_packed)
        packed_input_ids = np.zeros((chunk_size, max_length), dtype=np.int32)
        packed_attention_mask = np.zeros((chunk_size, max_length), dtype=np.int8)
        packed_document_ids = np.zeros((chunk_size, max_length), dtype=np.int16)
        current_idx = 0
        
        iterator = tqdm(tokenized_dataset, desc="Packing", total=total_seqs, unit="seq")
        
        for example in iterator:
            input_ids = example["input_ids"]
            if not input_ids:
                continue
            
            packed = packer.add_sequence(input_ids)
            if packed:
                if current_idx >= len(packed_input_ids):
                    new_chunk = np.zeros((chunk_size, max_length), dtype=np.int32)
                    mask_chunk = np.zeros((chunk_size, max_length), dtype=np.int8)
                    doc_chunk = np.zeros((chunk_size, max_length), dtype=np.int16)
                    packed_input_ids = np.vstack([packed_input_ids, new_chunk])
                    packed_attention_mask = np.vstack([packed_attention_mask, mask_chunk])
                    packed_document_ids = np.vstack([packed_document_ids, doc_chunk])
                
                packed_input_ids[current_idx] = packed.input_ids
                packed_attention_mask[current_idx] = packed.attention_mask
                packed_document_ids[current_idx] = packed.document_ids
                current_idx += 1
        
        final = packer.flush()
        if final:
            if current_idx >= len(packed_input_ids):
                new_chunk = np.zeros((chunk_size, max_length), dtype=np.int32)
                mask_chunk = np.zeros((chunk_size, max_length), dtype=np.int8)
                doc_chunk = np.zeros((chunk_size, max_length), dtype=np.int16)
                packed_input_ids = np.vstack([packed_input_ids, new_chunk])
                packed_attention_mask = np.vstack([packed_attention_mask, mask_chunk])
                packed_document_ids = np.vstack([packed_document_ids, doc_chunk])
            
            packed_input_ids[current_idx] = final.input_ids
            packed_attention_mask[current_idx] = final.attention_mask
            packed_document_ids[current_idx] = final.document_ids
            current_idx += 1
        
        packed_input_ids = packed_input_ids[:current_idx]
        packed_attention_mask = packed_attention_mask[:current_idx]
        packed_document_ids = packed_document_ids[:current_idx]
        num_packed = current_idx
    
    # Log results
    if isinstance(total_seqs, int):
        logger.info(f"\nPacking complete: {total_seqs} sequences -> {num_packed} packed examples")
        logger.info(f"Packing ratio: {total_seqs / max(1, num_packed):.2f}x efficiency gain")
    else:
        logger.info(f"\nPacking complete: {num_packed} packed examples")
    
    logger.info(f"  input_ids shape: {packed_input_ids.shape}, size: {packed_input_ids.nbytes / 1e9:.2f} GB")
    logger.info(f"  attention_mask shape: {packed_attention_mask.shape}, size: {packed_attention_mask.nbytes / 1e9:.2f} GB")
    
    return {
        "input_ids": packed_input_ids,
        "attention_mask": packed_attention_mask,
        "document_ids": packed_document_ids,  # For block-diagonal attention masking
        "max_length": max_length,
        "num_examples": num_packed,
    }


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
    pack_sequences: bool = True,
    return_overflowing_tokens: bool = True,
    add_mask_token: bool = False,
    seed: int = 42,
    pack_batch_size: int = 50000,  # Larger batches = better packing efficiency
):
    """
    Pretokenize and optionally pack a dataset with multi-CPU parallelization.
    
    Args:
        dataset_name: HuggingFace dataset name
        dataset_path: Local dataset path
        output_dir: Output directory
        tokenizer_name: Tokenizer to use
        max_length: Maximum sequence length (128/512/1024/2048)
        min_length: Minimum sequence length (filter shorter)
        stride: Overlap for splitting long texts
        text_column: Column containing text
        num_proc: Number of CPU processes (default: all CPUs)
        split: Dataset split
        max_samples: Max samples to process
        lang: Language filter
        pack_sequences: Whether to pack sequences
        return_overflowing_tokens: Split long texts into windows
        add_mask_token: Add <mask> token if missing
        seed: Random seed
        pack_batch_size: Batch size for parallel packing (controls memory)
        
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
    
    logger.info(f"=" * 60)
    logger.info(f"Pretokenization Pipeline")
    logger.info(f"=" * 60)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Tokenizer: {tokenizer_name}")
    logger.info(f"Max length: {max_length}")
    logger.info(f"Num processes: {num_proc}")
    logger.info(f"Pack sequences: {pack_sequences}")
    logger.info(f"Return overflowing tokens: {return_overflowing_tokens}")
    
    # Load tokenizer
    logger.info(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    
    if add_mask_token and tokenizer.mask_token is None:
        logger.info("Adding <mask> token to tokenizer")
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
    
    # Load dataset
    logger.info(f"\nLoading dataset...")
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
    logger.info(f"\nTokenizing...")
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
    
    # Save tokenized dataset
    tokenized_path = os.path.join(output_dir, f"{split}_tokenized")
    logger.info(f"\nSaving tokenized dataset to {tokenized_path}")
    tokenized.save_to_disk(tokenized_path)
    
    # Pack sequences
    if pack_sequences:
        logger.info(f"\nPacking sequences...")
        packed = pack_tokenized_dataset(
            tokenized_dataset=tokenized,
            max_length=max_length,
            tokenizer=tokenizer,
            num_proc=num_proc,
            batch_size=pack_batch_size,
        )
        
        import numpy as np
        
        # Save as .npz (fast, compressed)
        np_path = os.path.join(output_dir, f"{split}_packed.npz")
        logger.info(f"Saving as compressed numpy to {np_path}...")
        np.savez_compressed(
            np_path,
            input_ids=packed["input_ids"],
            attention_mask=packed["attention_mask"],
            document_ids=packed["document_ids"],  # For block-diagonal attention masking
            max_length=packed["max_length"],
            num_examples=packed["num_examples"],
        )
        logger.info(f"Saved to {np_path} ({os.path.getsize(np_path) / 1e9:.2f} GB)")
        
        # Optionally save as .pt for PyTorch (faster with numpy arrays)
        packed_path = os.path.join(output_dir, f"{split}_packed.pt")
        logger.info(f"Also saving as .pt (PyTorch format)...")
        torch.save(packed, packed_path)
        logger.info(f"Saved to {packed_path} ({os.path.getsize(packed_path) / 1e9:.2f} GB)")
    
    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer")
    logger.info(f"\nSaving tokenizer to {tokenizer_path}")
    tokenizer.save_pretrained(tokenizer_path)
    
    # Save config
    config = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "tokenizer_name": tokenizer_name,
        "max_length": max_length,
        "min_length": min_length,
        "stride": stride,
        "packed": pack_sequences,
        "return_overflowing_tokens": return_overflowing_tokens,
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "num_sequences": len(tokenized),
        "num_packed": len(packed["input_ids"]) if pack_sequences else None,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"\n" + "=" * 60)
    logger.info(f"Pretokenization complete!")
    logger.info(f"=" * 60)
    logger.info(f"Tokenized sequences: {len(tokenized)}")
    if pack_sequences:
        logger.info(f"Packed examples: {len(packed['input_ids'])}")
    logger.info(f"Output directory: {output_dir}")
    
    return output_dir


# ============== PyTorch Datasets ==============

class PackedMLMDataset(Dataset):
    """
    PyTorch Dataset for pretokenized and packed MLM data.
    
    Supports block-diagonal attention masking (like ModernBERT) to prevent
    cross-attention between different documents packed into the same sequence.
    """
    
    def __init__(
        self,
        data_path: str,
        mlm_probability: float = 0.15,
        tokenizer_path: Optional[str] = None,
        tokenizer_name: str = "thebajajra/RexBERT-base",
        use_block_diagonal_attention: bool = True,  # Enable document-aware masking
    ):
        import numpy as np
        
        logger.info(f"Loading packed dataset from {data_path}")
        
        self.document_ids = None
        self.use_block_diagonal_attention = use_block_diagonal_attention
        
        # Support both .pt and .npz formats
        if data_path.endswith(".npz"):
            data = np.load(data_path)
            self.input_ids = data["input_ids"]
            self.attention_mask = data["attention_mask"]
            if "document_ids" in data:
                self.document_ids = data["document_ids"]
        elif data_path.endswith(".pt"):
            data = torch.load(data_path)
            self.input_ids = data["input_ids"]
            self.attention_mask = data["attention_mask"]
            if "document_ids" in data:
                self.document_ids = data["document_ids"]
            # Convert to numpy if they're lists (legacy format)
            if isinstance(self.input_ids, list):
                logger.info("Converting lists to numpy arrays...")
                self.input_ids = np.array(self.input_ids, dtype=np.int32)
                self.attention_mask = np.array(self.attention_mask, dtype=np.int8)
        else:
            # Try .npz first, then .pt
            npz_path = data_path.replace(".pt", ".npz") if ".pt" in data_path else data_path + ".npz"
            pt_path = data_path.replace(".npz", ".pt") if ".npz" in data_path else data_path + ".pt"
            
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                self.input_ids = data["input_ids"]
                self.attention_mask = data["attention_mask"]
                if "document_ids" in data:
                    self.document_ids = data["document_ids"]
            elif os.path.exists(pt_path):
                data = torch.load(pt_path)
                self.input_ids = data["input_ids"]
                self.attention_mask = data["attention_mask"]
                if "document_ids" in data:
                    self.document_ids = data["document_ids"]
            else:
                raise FileNotFoundError(f"Could not find {npz_path} or {pt_path}")
        
        self.mlm_probability = mlm_probability
        
        # Load tokenizer
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.mask_token_id = self.tokenizer.mask_token_id or 103
        self.pad_token_id = self.tokenizer.pad_token_id or 0
        self.cls_token_id = self.tokenizer.cls_token_id or 101
        self.sep_token_id = self.tokenizer.sep_token_id or 102
        self.vocab_size = self.tokenizer.vocab_size
        
        if self.document_ids is not None:
            logger.info(f"Loaded {len(self.input_ids)} packed examples with document IDs (block-diagonal attention enabled)")
        else:
            logger.info(f"Loaded {len(self.input_ids)} packed examples (no document IDs - cross-attention allowed)")
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Handle numpy arrays efficiently
        input_ids = torch.from_numpy(self.input_ids[idx].astype('int64')).long()
        attention_mask = torch.from_numpy(self.attention_mask[idx].astype('int64')).long()
        labels = input_ids.clone()
        
        # Don't mask special tokens or padding
        special_tokens_mask = (
            (input_ids == self.pad_token_id) |
            (input_ids == self.cls_token_id) |
            (input_ids == self.sep_token_id)
        )
        
        # Random masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        
        # 80% [MASK], 10% random, 10% unchanged
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        # Add document_ids for block-diagonal attention masking (like ModernBERT)
        if self.use_block_diagonal_attention and self.document_ids is not None:
            result["document_ids"] = torch.from_numpy(self.document_ids[idx].astype('int64')).long()
        
        return result


class TokenizedMLMDataset(Dataset):
    """PyTorch Dataset for tokenized (unpacked) data saved via save_to_disk."""
    
    def __init__(
        self,
        data_path: str,
        mlm_probability: float = 0.15,
        tokenizer_path: Optional[str] = None,
        tokenizer_name: str = "thebajajra/RexBERT-base",
        max_length: int = 512,
    ):
        logger.info(f"Loading tokenized dataset from {data_path}")
        self.dataset = datasets.load_from_disk(data_path)
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        
        # Load tokenizer
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.mask_token_id = self.tokenizer.mask_token_id or 103
        self.pad_token_id = self.tokenizer.pad_token_id or 0
        self.cls_token_id = self.tokenizer.cls_token_id or 101
        self.sep_token_id = self.tokenizer.sep_token_id or 102
        self.vocab_size = self.tokenizer.vocab_size
        
        logger.info(f"Loaded {len(self.dataset)} sequences")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        example = self.dataset[idx]
        input_ids = example["input_ids"]
        attention_mask = example.get("attention_mask", [1] * len(input_ids))
        
        # Pad to max_length
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = input_ids.clone()
        
        # MLM masking
        special_tokens_mask = (
            (input_ids == self.pad_token_id) |
            (input_ids == self.cls_token_id) |
            (input_ids == self.sep_token_id)
        )
        
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class StreamingPackedDataset(IterableDataset):
    """Streaming dataset that packs sequences on-the-fly with stride for long texts."""
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str = "thebajajra/RexBERT-base",
        max_length: int = 512,
        text_column: str = "text",
        mlm_probability: float = 0.15,
        shuffle_buffer: int = 10000,
        dataset_config: Optional[str] = None,
        stride: int = 64,  # Overlap between windows for long texts
        return_overflowing_tokens: bool = True,  # Split long texts into windows
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.text_column = text_column
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.shuffle_buffer = shuffle_buffer
        self.stride = stride
        self.return_overflowing_tokens = return_overflowing_tokens
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mask_token_id = self.tokenizer.mask_token_id or 103
        self.pad_token_id = self.tokenizer.pad_token_id or 0
        self.cls_token_id = self.tokenizer.cls_token_id or 101
        self.sep_token_id = self.tokenizer.sep_token_id or 102
        self.vocab_size = self.tokenizer.vocab_size
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        ds = datasets.load_dataset(
            self.dataset_name,
            self.dataset_config,
            streaming=True,
            split="train"
        )
        ds = ds.shuffle(buffer_size=self.shuffle_buffer)
        
        packer = SequencePacker(
            max_length=self.max_length,
            sep_token_id=self.sep_token_id,
            cls_token_id=self.cls_token_id,
            pad_token_id=self.pad_token_id,
        )
        
        for example in ds:
            text = example.get(self.text_column, "")
            if not text or not text.strip():
                continue
            
            # Tokenize with overflowing tokens for long texts
            tokens = self.tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=self.max_length - 2,  # Leave room for CLS/SEP
                add_special_tokens=False,
                return_overflowing_tokens=self.return_overflowing_tokens,
                stride=self.stride if self.return_overflowing_tokens else 0,
            )
            
            # Handle both single sequence and multiple windows
            if self.return_overflowing_tokens and "overflow_to_sample_mapping" in tokens:
                # Multiple windows from long text
                all_input_ids = tokens["input_ids"]
                if isinstance(all_input_ids[0], list):
                    # Batched output - multiple windows
                    for input_ids in all_input_ids:
                        if not input_ids:
                            continue
                        packed = packer.add_sequence(input_ids)
                        if packed:
                            yield self._apply_mlm(packed)
                else:
                    # Single sequence
                    if all_input_ids:
                        packed = packer.add_sequence(all_input_ids)
                        if packed:
                            yield self._apply_mlm(packed)
            else:
                # Standard tokenization (no overflow)
                input_ids = tokens["input_ids"]
                if not input_ids:
                    continue
                
                packed = packer.add_sequence(input_ids)
                if packed:
                    yield self._apply_mlm(packed)
        
        final = packer.flush()
        if final:
            yield self._apply_mlm(final)
    
    def _apply_mlm(self, packed: PackedExample) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(packed.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(packed.attention_mask, dtype=torch.long)
        labels = input_ids.clone()
        
        special_tokens_mask = (
            (input_ids == self.pad_token_id) |
            (input_ids == self.cls_token_id) |
            (input_ids == self.sep_token_id)
        )
        
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        # Add document_ids for block-diagonal attention masking (like ModernBERT)
        if packed.document_ids is not None:
            result["document_ids"] = torch.tensor(packed.document_ids, dtype=torch.long)
        
        return result


# ============== DataLoader Factory ==============

def create_dataloader(
    data_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    tokenizer_path: Optional[str] = None,
    tokenizer_name: str = "thebajajra/RexBERT-base",
    mlm_probability: float = 0.15,
    max_length: int = 512,
) -> DataLoader:
    """
    Create DataLoader from pretokenized data.
    
    Automatically detects format:
    - .pt or .npz: Packed data (fast)
    - directory: Tokenized Arrow format
    """
    if data_path.endswith(".pt") or data_path.endswith(".npz"):
        # Packed data (.pt or .npz)
        dataset = PackedMLMDataset(
            data_path=data_path,
            mlm_probability=mlm_probability,
            tokenizer_path=tokenizer_path,
            tokenizer_name=tokenizer_name,
        )
    else:
        # Tokenized data (Arrow format directory)
        dataset = TokenizedMLMDataset(
            data_path=data_path,
            mlm_probability=mlm_probability,
            tokenizer_path=tokenizer_path,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


# ============== CLI ==============

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    parser = argparse.ArgumentParser(
        description="Pretokenize and pack dataset for Delta-Matryoshka++ training"
    )
    
    # Dataset source (mutually exclusive)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-name", type=str,
                     help="HuggingFace hub dataset name (e.g., thebajajra/Ecom-niverse)")
    src.add_argument("--dataset-path", type=str,
                     help="Local dataset path (from datasets.save_to_disk)")
    
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for processed data")
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
                        help="Number of CPU processes for parallel tokenization and packing")
    parser.add_argument("--pack-batch-size", type=int, default=50000,
                        help="Batch size for parallel packing (larger = better packing efficiency)")
    
    # Tokenization
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length (128/512/1024/2048)")
    parser.add_argument("--min-length", type=int, default=16,
                        help="Minimum sequence length (filter shorter)")
    parser.add_argument("--stride", type=int, default=64,
                        help="Token overlap when splitting long texts")
    parser.add_argument("--return-overflowing-tokens", action="store_true",
                        help="Split long texts into multiple windows instead of truncating")
    
    # Packing
    parser.add_argument("--no-pack", action="store_true",
                        help="Disable sequence packing")
    
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
        pack_sequences=not args.no_pack,
        return_overflowing_tokens=args.return_overflowing_tokens,
        add_mask_token=args.add_mask_token,
        seed=args.seed,
        pack_batch_size=args.pack_batch_size,
    )
