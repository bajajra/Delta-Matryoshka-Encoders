"""
Data Preprocessing Utilities for Delta-Matryoshka++ Training

Features:
1. Pretokenization - tokenize once, save to disk
2. Sequence Packing - concatenate short sequences to minimize padding
3. Dynamic batching - group similar lengths together
4. Streaming support for large datasets
"""

import os
import json
import random
from typing import Optional, List, Dict, Iterator, Any
from dataclasses import dataclass
from pathlib import Path
import multiprocessing as mp

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

try:
    from datasets import load_dataset, Dataset as HFDataset
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


@dataclass
class PackedExample:
    """A packed example containing multiple concatenated sequences."""
    input_ids: List[int]
    attention_mask: List[int]
    # Boundaries of original sequences (for debugging)
    sequence_boundaries: List[int]


class SequencePacker:
    """
    Packs multiple short sequences into single max_length sequences.
    
    This significantly reduces padding waste and improves training efficiency.
    """
    
    def __init__(
        self,
        max_length: int = 512,
        sep_token_id: int = 102,  # [SEP] for BERT
        cls_token_id: int = 101,  # [CLS] for BERT
        pad_token_id: int = 0,
        add_special_tokens: bool = True
    ):
        """
        Args:
            max_length: Maximum packed sequence length
            sep_token_id: Separator token ID
            cls_token_id: CLS token ID  
            pad_token_id: Padding token ID
            add_special_tokens: Whether to add CLS/SEP between sequences
        """
        self.max_length = max_length
        self.sep_token_id = sep_token_id
        self.cls_token_id = cls_token_id
        self.pad_token_id = pad_token_id
        self.add_special_tokens = add_special_tokens
        
        # Buffer for accumulating sequences
        self._buffer_ids = []
        self._buffer_boundaries = []
        self._current_length = 0
    
    def add_sequence(self, input_ids: List[int]) -> Optional[PackedExample]:
        """
        Add a sequence to the packing buffer.
        
        Returns a PackedExample if buffer is full, None otherwise.
        """
        # Remove existing special tokens if present
        if input_ids and input_ids[0] == self.cls_token_id:
            input_ids = input_ids[1:]
        if input_ids and input_ids[-1] == self.sep_token_id:
            input_ids = input_ids[:-1]
        
        # Calculate length needed
        seq_len = len(input_ids)
        if self.add_special_tokens:
            seq_len += 1  # Add SEP between sequences
        
        # Check if this sequence fits
        if self._current_length + seq_len > self.max_length:
            # Flush buffer and return packed example
            packed = self._flush_buffer()
            
            # Start new buffer with this sequence
            self._buffer_ids = list(input_ids)
            self._buffer_boundaries = [len(input_ids)]
            self._current_length = len(input_ids)
            
            return packed
        
        # Add to buffer
        if self._buffer_ids and self.add_special_tokens:
            self._buffer_ids.append(self.sep_token_id)
            self._current_length += 1
        
        self._buffer_ids.extend(input_ids)
        self._buffer_boundaries.append(len(self._buffer_ids))
        self._current_length = len(self._buffer_ids)
        
        return None
    
    def _flush_buffer(self) -> Optional[PackedExample]:
        """Flush the current buffer and return packed example."""
        if not self._buffer_ids:
            return None
        
        # Add CLS at start if using special tokens
        ids = self._buffer_ids
        if self.add_special_tokens:
            ids = [self.cls_token_id] + ids
        
        # Pad to max_length
        pad_len = self.max_length - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [self.pad_token_id] * pad_len
        
        packed = PackedExample(
            input_ids=ids,
            attention_mask=attention_mask,
            sequence_boundaries=self._buffer_boundaries.copy()
        )
        
        # Clear buffer
        self._buffer_ids = []
        self._buffer_boundaries = []
        self._current_length = 0
        
        return packed
    
    def flush(self) -> Optional[PackedExample]:
        """Flush any remaining sequences in buffer."""
        return self._flush_buffer()


def pretokenize_dataset(
    dataset_name: str,
    output_dir: str,
    tokenizer_name: str = "thebajajra/RexBERT-base",
    max_length: int = 512,
    text_column: str = "text",
    num_proc: int = 4,
    dataset_config: Optional[str] = None,
    pack_sequences: bool = True,
    streaming: bool = False,
    max_samples: Optional[int] = None,
):
    """
    Pretokenize and optionally pack a dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        output_dir: Directory to save pretokenized data
        tokenizer_name: Tokenizer to use
        max_length: Maximum sequence length. Common values:
            - 128: Fast training, short texts
            - 512: Standard BERT/RexBERT (recommended)
            - 1024: Extended context (RexBERT-base supports this)
            - 2048: Maximum extended context
        text_column: Column containing text
        num_proc: Number of processes for tokenization
        dataset_config: Dataset configuration name
        pack_sequences: Whether to pack short sequences together
        streaming: Use streaming mode for large datasets
        max_samples: Maximum samples to process (None = all)
    
    Returns:
        Path to saved dataset
        
    Example:
        # Standard 512 length
        pretokenize_dataset("thebajajra/Ecom-niverse", "./data", max_length=512)
        
        # Extended 1024 for longer product descriptions
        pretokenize_dataset("thebajajra/Ecom-niverse", "./data", max_length=1024)
    """
    if not HF_AVAILABLE:
        raise ImportError("datasets and transformers required")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    print(f"Loading dataset: {dataset_name}")
    if streaming:
        ds = load_dataset(dataset_name, dataset_config, streaming=True)
    else:
        ds = load_dataset(dataset_name, dataset_config)
    
    # Tokenization function
    def tokenize_fn(examples):
        texts = examples[text_column]
        # Filter empty texts
        texts = [t if t else "" for t in texts]
        return tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length - 2,  # Leave room for special tokens
            add_special_tokens=False,
        )
    
    # Process each split
    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        
        print(f"\nProcessing {split} split...")
        split_ds = ds[split]
        
        if max_samples and not streaming:
            split_ds = split_ds.select(range(min(max_samples, len(split_ds))))
        
        # Tokenize
        print("  Tokenizing...")
        if streaming:
            tokenized = split_ds.map(
                tokenize_fn,
                batched=True,
                remove_columns=[text_column]
            )
        else:
            tokenized = split_ds.map(
                tokenize_fn,
                batched=True,
                num_proc=num_proc,
                remove_columns=[c for c in split_ds.column_names if c != "input_ids"]
            )
        
        # Pack sequences
        if pack_sequences:
            print("  Packing sequences...")
            packed_data = pack_tokenized_dataset(
                tokenized,
                max_length=max_length,
                sep_token_id=tokenizer.sep_token_id or 102,
                cls_token_id=tokenizer.cls_token_id or 101,
                pad_token_id=tokenizer.pad_token_id or 0,
                streaming=streaming,
                max_samples=max_samples,
            )
            
            # Save packed data
            output_path = os.path.join(output_dir, f"{split}_packed.pt")
            torch.save(packed_data, output_path)
            print(f"  Saved {len(packed_data['input_ids'])} packed sequences to {output_path}")
        else:
            # Save tokenized (unpacked) data
            output_path = os.path.join(output_dir, split)
            if not streaming:
                tokenized.save_to_disk(output_path)
                print(f"  Saved to {output_path}")
    
    # Save config
    config = {
        "dataset_name": dataset_name,
        "tokenizer_name": tokenizer_name,
        "max_length": max_length,
        "packed": pack_sequences,
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nPretokenization complete! Data saved to {output_dir}")
    return output_dir


def pack_tokenized_dataset(
    tokenized_dataset,
    max_length: int = 512,
    sep_token_id: int = 102,
    cls_token_id: int = 101,
    pad_token_id: int = 0,
    streaming: bool = False,
    max_samples: Optional[int] = None,
) -> Dict[str, List]:
    """
    Pack a tokenized dataset into fixed-length sequences.
    
    Returns dict with 'input_ids' and 'attention_mask' lists.
    """
    packer = SequencePacker(
        max_length=max_length,
        sep_token_id=sep_token_id,
        cls_token_id=cls_token_id,
        pad_token_id=pad_token_id,
    )
    
    packed_input_ids = []
    packed_attention_mask = []
    
    count = 0
    for example in tokenized_dataset:
        if max_samples and count >= max_samples:
            break
        
        input_ids = example["input_ids"]
        if not input_ids:
            continue
        
        packed = packer.add_sequence(input_ids)
        if packed:
            packed_input_ids.append(packed.input_ids)
            packed_attention_mask.append(packed.attention_mask)
        
        count += 1
        if count % 100000 == 0:
            print(f"    Processed {count} sequences, packed {len(packed_input_ids)} examples")
    
    # Flush remaining
    final = packer.flush()
    if final:
        packed_input_ids.append(final.input_ids)
        packed_attention_mask.append(final.attention_mask)
    
    print(f"    Total: {count} sequences -> {len(packed_input_ids)} packed examples")
    print(f"    Packing ratio: {count / max(1, len(packed_input_ids)):.2f}x")
    
    return {
        "input_ids": packed_input_ids,
        "attention_mask": packed_attention_mask,
    }


class PackedMLMDataset(Dataset):
    """
    PyTorch Dataset for pretokenized and packed MLM data.
    """
    
    def __init__(
        self,
        data_path: str,
        mlm_probability: float = 0.15,
        tokenizer_name: str = "thebajajra/RexBERT-base",
    ):
        """
        Args:
            data_path: Path to packed .pt file
            mlm_probability: Probability of masking tokens
            tokenizer_name: Tokenizer for MLM masking
        """
        print(f"Loading packed dataset from {data_path}")
        data = torch.load(data_path)
        
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        
        # Load tokenizer for MLM
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mlm_probability = mlm_probability
        
        # Special token IDs
        self.mask_token_id = self.tokenizer.mask_token_id or 103
        self.pad_token_id = self.tokenizer.pad_token_id or 0
        self.cls_token_id = self.tokenizer.cls_token_id or 101
        self.sep_token_id = self.tokenizer.sep_token_id or 102
        self.vocab_size = self.tokenizer.vocab_size
        
        print(f"  Loaded {len(self.input_ids)} packed examples")
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        attention_mask = torch.tensor(self.attention_mask[idx], dtype=torch.long)
        
        # Apply MLM masking
        labels = input_ids.clone()
        
        # Create mask for MLM (don't mask special tokens or padding)
        special_tokens_mask = (
            (input_ids == self.pad_token_id) |
            (input_ids == self.cls_token_id) |
            (input_ids == self.sep_token_id)
        )
        
        # Random mask
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set labels for non-masked tokens to -100 (ignored in loss)
        labels[~masked_indices] = -100
        
        # 80% replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id
        
        # 10% replace with random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        # 10% keep original (do nothing)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class StreamingPackedDataset(IterableDataset):
    """
    Streaming dataset that packs sequences on-the-fly.
    
    Useful for very large datasets that don't fit in memory.
    """
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str = "thebajajra/RexBERT-base",
        max_length: int = 512,
        text_column: str = "text",
        mlm_probability: float = 0.15,
        shuffle_buffer: int = 10000,
        dataset_config: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.text_column = text_column
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.shuffle_buffer = shuffle_buffer
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mask_token_id = self.tokenizer.mask_token_id or 103
        self.pad_token_id = self.tokenizer.pad_token_id or 0
        self.cls_token_id = self.tokenizer.cls_token_id or 101
        self.sep_token_id = self.tokenizer.sep_token_id or 102
        self.vocab_size = self.tokenizer.vocab_size
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # Load dataset in streaming mode
        ds = load_dataset(
            self.dataset_name,
            self.dataset_config,
            streaming=True,
            split="train"
        )
        ds = ds.shuffle(buffer_size=self.shuffle_buffer)
        
        # Create packer
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
            
            # Tokenize
            tokens = self.tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=self.max_length - 2,
                add_special_tokens=False,
            )
            
            input_ids = tokens["input_ids"]
            if not input_ids:
                continue
            
            # Try to pack
            packed = packer.add_sequence(input_ids)
            if packed:
                yield self._apply_mlm(packed)
        
        # Yield final packed example
        final = packer.flush()
        if final:
            yield self._apply_mlm(final)
    
    def _apply_mlm(self, packed: PackedExample) -> Dict[str, torch.Tensor]:
        """Apply MLM masking to packed example."""
        input_ids = torch.tensor(packed.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(packed.attention_mask, dtype=torch.long)
        labels = input_ids.clone()
        
        # Special tokens mask
        special_tokens_mask = (
            (input_ids == self.pad_token_id) |
            (input_ids == self.cls_token_id) |
            (input_ids == self.sep_token_id)
        )
        
        # MLM masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        labels[~masked_indices] = -100
        
        # 80% [MASK], 10% random, 10% keep
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


def create_dataloader(
    data_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    tokenizer_name: str = "thebajajra/RexBERT-base",
    mlm_probability: float = 0.15,
) -> DataLoader:
    """
    Create DataLoader from pretokenized packed dataset.
    
    Args:
        data_path: Path to packed .pt file
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        tokenizer_name: Tokenizer for MLM masking
        mlm_probability: MLM probability
        
    Returns:
        DataLoader
    """
    dataset = PackedMLMDataset(
        data_path=data_path,
        mlm_probability=mlm_probability,
        tokenizer_name=tokenizer_name,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


# ============== CLI for pretokenization ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pretokenize and pack dataset")
    parser.add_argument("--dataset", type=str, default="thebajajra/Ecom-niverse",
                        help="HuggingFace dataset name")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="thebajajra/RexBERT-base",
                        help="Tokenizer name")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length (128=fast, 512=standard, 1024/2048=extended)")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Text column name")
    parser.add_argument("--num_proc", type=int, default=4,
                        help="Number of processes")
    parser.add_argument("--no_pack", action="store_true",
                        help="Disable sequence packing")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to process")
    
    args = parser.parse_args()
    
    pretokenize_dataset(
        dataset_name=args.dataset,
        output_dir=args.output,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        text_column=args.text_column,
        num_proc=args.num_proc,
        pack_sequences=not args.no_pack,
        streaming=args.streaming,
        max_samples=args.max_samples,
    )

