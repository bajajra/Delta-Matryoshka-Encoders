"""
HuggingFace Export and Integration for Delta-Matryoshka++ Models

This module provides:
1. HuggingFace-compatible configuration class
2. HuggingFace-compatible model class (extends PreTrainedModel)
3. Export functions to save sub-models in HF format
4. Load functions to use exported models with standard HF APIs

Usage:
    # Export a specific budget as HF model
    from src.hf_export import export_to_huggingface
    export_to_huggingface(model, budget=(0.5, 6, 6), output_dir="./hf_mini")
    
    # Load with standard HF API
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained("./hf_mini", trust_remote_code=True)
"""

import os
import json
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass, asdict

try:
    from transformers import PreTrainedModel, PretrainedConfig
    from transformers.modeling_outputs import (
        BaseModelOutput, 
        MaskedLMOutput,
        BaseModelOutputWithPooling
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    PreTrainedModel = nn.Module
    PretrainedConfig = object


# ============== Configuration ==============

class DeltaMatryoshkaConfig(PretrainedConfig if HF_AVAILABLE else object):
    """
    Configuration class for Delta-Matryoshka models.
    
    Compatible with HuggingFace's config system.
    """
    model_type = "delta_matryoshka"
    
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        layer_norm_eps: float = 1e-5,
        pad_token_id: int = 0,
        # Delta-specific
        base_ratio: float = 0.5,
        base_heads: int = 6,
        # Budget (for sliced export)
        budget_width: float = 1.0,
        budget_heads: int = 12,
        budget_depth: int = 12,
        # Flags
        is_sliced: bool = False,
        **kwargs
    ):
        if HF_AVAILABLE:
            super().__init__(pad_token_id=pad_token_id, **kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        
        # Delta-specific
        self.base_ratio = base_ratio
        self.base_heads = base_heads
        
        # Budget for this slice
        self.budget_width = budget_width
        self.budget_heads = budget_heads
        self.budget_depth = budget_depth
        self.is_sliced = is_sliced
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        output = {
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "max_position_embeddings": self.max_position_embeddings,
            "layer_norm_eps": self.layer_norm_eps,
            "pad_token_id": self.pad_token_id,
            "base_ratio": self.base_ratio,
            "base_heads": self.base_heads,
            "budget_width": self.budget_width,
            "budget_heads": self.budget_heads,
            "budget_depth": self.budget_depth,
            "is_sliced": self.is_sliced,
        }
        return output
    
    def save_pretrained(self, save_directory: str):
        """Save config to directory."""
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load config from directory or HF hub."""
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            return cls(**config_dict, **kwargs)
        raise ValueError(f"Config not found at {config_path}")


# ============== Sliced Model (HF Compatible) ==============

class DeltaMatryoshkaModel(PreTrainedModel if HF_AVAILABLE else nn.Module):
    """
    HuggingFace-compatible Delta-Matryoshka model.
    
    This is a sliced/exported version that runs at a fixed budget.
    For full multi-budget training, use DeltaEncoder from model.py.
    """
    config_class = DeltaMatryoshkaConfig
    base_model_prefix = "delta_matryoshka"
    
    def __init__(self, config: DeltaMatryoshkaConfig):
        if HF_AVAILABLE:
            super().__init__(config)
        else:
            super().__init__()
            self.config = config
        
        # Effective dimensions based on budget
        self.hidden_size = config.hidden_size
        self.num_layers = config.budget_depth
        self.num_heads = config.budget_heads
        self.intermediate_size = int(config.budget_width * config.intermediate_size)
        
        # Embeddings
        self.embeddings = SlicedEmbeddings(config)
        
        # Transformer layers (only budget_depth layers)
        self.encoder = nn.ModuleList([
            SlicedTransformerLayer(config, layer_idx=i)
            for i in range(self.num_layers)
        ])
        
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """
        Forward pass compatible with HuggingFace API.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Embeddings
        hidden_states = self.embeddings(input_ids, position_ids, token_type_ids)
        
        # Attention mask -> additive form
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        all_hidden_states = () if output_hidden_states else None
        
        for layer in self.encoder:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer(hidden_states, extended_attention_mask)
        
        hidden_states = self.layernorm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return (hidden_states, all_hidden_states)
        
        if HF_AVAILABLE:
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
            )
        return {"last_hidden_state": hidden_states, "hidden_states": all_hidden_states}
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


class DeltaMatryoshkaForMaskedLM(PreTrainedModel if HF_AVAILABLE else nn.Module):
    """
    Delta-Matryoshka with MLM head for HuggingFace compatibility.
    """
    config_class = DeltaMatryoshkaConfig
    base_model_prefix = "delta_matryoshka"
    
    def __init__(self, config: DeltaMatryoshkaConfig):
        if HF_AVAILABLE:
            super().__init__(config)
        else:
            super().__init__()
            self.config = config
        
        self.delta_matryoshka = DeltaMatryoshkaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.delta_matryoshka.embeddings.word_embeddings.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.delta_matryoshka(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        hidden_states = outputs["last_hidden_state"] if isinstance(outputs, dict) else outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + (outputs["hidden_states"] if isinstance(outputs, dict) else outputs.hidden_states,)
            return ((loss,) + output) if loss is not None else output
        
        if HF_AVAILABLE:
            return MaskedLMOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            )
        return {"loss": loss, "logits": logits}
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings


# ============== Sliced Components ==============

class SlicedEmbeddings(nn.Module):
    """Embeddings for sliced model."""
    
    def __init__(self, config: DeltaMatryoshkaConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class SlicedTransformerLayer(nn.Module):
    """Single transformer layer for sliced model."""
    
    def __init__(self, config: DeltaMatryoshkaConfig, layer_idx: int = 0):
        super().__init__()
        self.attention = SlicedAttention(config)
        self.intermediate = nn.Linear(
            config.hidden_size, 
            int(config.budget_width * config.intermediate_size)
        )
        self.output = nn.Linear(
            int(config.budget_width * config.intermediate_size),
            config.hidden_size
        )
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act = nn.GELU()
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        # Self-attention
        attn_output = self.attention(self.layernorm1(hidden_states), attention_mask)
        hidden_states = hidden_states + self.dropout(attn_output)
        
        # FFN
        ffn_output = self.output(self.act(self.intermediate(self.layernorm2(hidden_states))))
        hidden_states = hidden_states + self.dropout(ffn_output)
        
        return hidden_states


class SlicedAttention(nn.Module):
    """Multi-head attention for sliced model."""
    
    def __init__(self, config: DeltaMatryoshkaConfig):
        super().__init__()
        self.num_heads = config.budget_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out = nn.Linear(self.all_head_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores + attention_mask
        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.all_head_size)
        
        return self.out(context)


# ============== Export Functions ==============

def export_to_huggingface(
    delta_model: nn.Module,
    budget: Tuple[float, int, int],
    output_dir: str,
    tokenizer_path: Optional[str] = None,
    model_name: str = "delta-matryoshka",
):
    """
    Export a Delta-Matryoshka model at a specific budget to HuggingFace format.
    
    Args:
        delta_model: Trained DeltaEncoder model
        budget: (width, heads, depth) tuple for the sub-model to export
        output_dir: Directory to save the HF model
        tokenizer_path: Optional tokenizer to copy
        model_name: Name for the model card
    
    Returns:
        Path to exported model
    """
    width, heads, depth = budget
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create config
    config = DeltaMatryoshkaConfig(
        vocab_size=delta_model.embeddings.word_embeddings.num_embeddings,
        hidden_size=delta_model.hidden_size,
        num_hidden_layers=delta_model.num_layers,
        num_attention_heads=delta_model.num_attention_heads,
        intermediate_size=delta_model.layers[0].mlp.intermediate_size,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=delta_model.embeddings.position_embeddings.num_embeddings,
        layer_norm_eps=1e-5,
        pad_token_id=0,
        base_ratio=delta_model.base_ratio,
        base_heads=delta_model.base_heads,
        budget_width=width,
        budget_heads=heads,
        budget_depth=depth,
        is_sliced=True,
    )
    
    # Create sliced HF model
    hf_model = DeltaMatryoshkaForMaskedLM(config)
    
    # Copy weights from delta_model to hf_model
    _copy_weights_to_hf(delta_model, hf_model, budget)
    
    # Save model
    torch.save(hf_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    config.save_pretrained(output_dir)
    
    # Create model card
    _create_model_card(output_dir, model_name, budget, config)
    
    # Copy tokenizer if provided
    if tokenizer_path:
        _copy_tokenizer(tokenizer_path, output_dir)
    
    print(f"Exported model to {output_dir}")
    print(f"  Budget: width={width}, heads={heads}, depth={depth}")
    print(f"  Parameters: {sum(p.numel() for p in hf_model.parameters()):,}")
    
    return output_dir


def _copy_weights_to_hf(
    delta_model: nn.Module,
    hf_model: nn.Module,
    budget: Tuple[float, int, int]
):
    """Copy weights from DeltaEncoder to HF model at specific budget."""
    width, heads, depth = budget
    
    with torch.no_grad():
        # Embeddings
        hf_model.delta_matryoshka.embeddings.word_embeddings.weight.copy_(
            delta_model.embeddings.word_embeddings.weight
        )
        hf_model.delta_matryoshka.embeddings.position_embeddings.weight.copy_(
            delta_model.embeddings.position_embeddings.weight
        )
        hf_model.delta_matryoshka.embeddings.layernorm.weight.copy_(
            delta_model.embeddings.ln.weight
        )
        hf_model.delta_matryoshka.embeddings.layernorm.bias.copy_(
            delta_model.embeddings.ln.bias
        )
        
        # Transformer layers
        intermediate_size = int(width * delta_model.layers[0].mlp.intermediate_size)
        head_dim = delta_model.hidden_size // delta_model.num_attention_heads
        head_size = heads * head_dim
        
        for i in range(depth):
            delta_layer = delta_model.layers[i]
            hf_layer = hf_model.delta_matryoshka.encoder[i]
            
            # Attention Q, K, V - slice to budget heads
            hf_layer.attention.query.weight.copy_(delta_layer.attn.q.weight[:head_size])
            hf_layer.attention.query.bias.copy_(delta_layer.attn.q.bias[:head_size])
            hf_layer.attention.key.weight.copy_(delta_layer.attn.k.weight[:head_size])
            hf_layer.attention.key.bias.copy_(delta_layer.attn.k.bias[:head_size])
            hf_layer.attention.value.weight.copy_(delta_layer.attn.v.weight[:head_size])
            hf_layer.attention.value.bias.copy_(delta_layer.attn.v.bias[:head_size])
            
            # Attention output - slice input dimension
            hf_layer.attention.out.weight.copy_(delta_layer.attn.o.weight[:, :head_size])
            hf_layer.attention.out.bias.copy_(delta_layer.attn.o.bias)
            
            # MLP - slice intermediate dimension
            hf_layer.intermediate.weight.copy_(delta_layer.mlp.fc1.weight[:intermediate_size])
            hf_layer.intermediate.bias.copy_(delta_layer.mlp.fc1.bias[:intermediate_size])
            hf_layer.output.weight.copy_(delta_layer.mlp.fc2.weight[:, :intermediate_size])
            hf_layer.output.bias.copy_(delta_layer.mlp.fc2.bias)
            
            # LayerNorms
            if hasattr(delta_layer.ln1, 'base_ln'):
                hf_layer.layernorm1.weight.copy_(delta_layer.ln1.base_ln.weight)
                hf_layer.layernorm1.bias.copy_(delta_layer.ln1.base_ln.bias)
                hf_layer.layernorm2.weight.copy_(delta_layer.ln2.base_ln.weight)
                hf_layer.layernorm2.bias.copy_(delta_layer.ln2.base_ln.bias)
        
        # Final LayerNorm
        hf_model.delta_matryoshka.layernorm.weight.copy_(delta_model.ln_f.weight)
        hf_model.delta_matryoshka.layernorm.bias.copy_(delta_model.ln_f.bias)


def _create_model_card(output_dir: str, model_name: str, budget: Tuple, config: DeltaMatryoshkaConfig):
    """Create a model card README."""
    width, heads, depth = budget
    
    card = f"""---
language: en
tags:
- delta-matryoshka
- nested-models
- efficient-inference
license: apache-2.0
---

# {model_name}

This is a **Delta-Matryoshka++** model exported at budget `(width={width}, heads={heads}, depth={depth})`.

## Model Details

- **Hidden Size**: {config.hidden_size}
- **Layers**: {depth} (of {config.num_hidden_layers} max)
- **Attention Heads**: {heads} (of {config.num_attention_heads} max)
- **Intermediate Size**: {int(width * config.intermediate_size)}
- **Vocabulary Size**: {config.vocab_size}

## Usage

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("{model_name}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)
```

## About Delta-Matryoshka++

Delta-Matryoshka++ trains a single model that works at multiple compute budgets.
This export is a "sliced" version optimized for the specific budget above.

The base model can be run at smaller budgets for faster inference, or larger
budgets for better quality, all from the same trained weights.
"""
    
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(card)


def _copy_tokenizer(tokenizer_path: str, output_dir: str):
    """Copy tokenizer files to output directory."""
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"Warning: Could not copy tokenizer: {e}")


def load_sliced_model(
    model_path: str,
    device: str = "cpu"
) -> DeltaMatryoshkaForMaskedLM:
    """
    Load a sliced Delta-Matryoshka model from HuggingFace format.
    
    Args:
        model_path: Path to saved model
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    config = DeltaMatryoshkaConfig.from_pretrained(model_path)
    model = DeltaMatryoshkaForMaskedLM(config)
    
    state_dict = torch.load(
        os.path.join(model_path, "pytorch_model.bin"),
        map_location=device
    )
    model.load_state_dict(state_dict)
    model.to(device)
    
    return model


# ============== Batch Export ==============

def export_all_budgets(
    delta_model: nn.Module,
    budgets: List[Tuple[float, int, int]],
    output_base_dir: str,
    tokenizer_path: Optional[str] = None,
    model_name_prefix: str = "delta-matryoshka"
):
    """
    Export multiple budget configurations to HuggingFace format.
    
    Args:
        delta_model: Trained DeltaEncoder
        budgets: List of (width, heads, depth) tuples
        output_base_dir: Base directory for exports
        tokenizer_path: Tokenizer to copy
        model_name_prefix: Prefix for model names
    """
    os.makedirs(output_base_dir, exist_ok=True)
    
    exported = []
    for width, heads, depth in budgets:
        budget_name = f"{model_name_prefix}-w{width}-h{heads}-d{depth}"
        output_dir = os.path.join(output_base_dir, budget_name)
        
        export_to_huggingface(
            delta_model=delta_model,
            budget=(width, heads, depth),
            output_dir=output_dir,
            tokenizer_path=tokenizer_path,
            model_name=budget_name
        )
        exported.append(output_dir)
    
    print(f"\nExported {len(exported)} models to {output_base_dir}")
    return exported


# ============== Register with HuggingFace ==============

def register_with_auto_classes():
    """Register Delta-Matryoshka with HuggingFace Auto classes."""
    if not HF_AVAILABLE:
        print("Warning: transformers not available, skipping Auto registration")
        return
    
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
    
    AutoConfig.register("delta_matryoshka", DeltaMatryoshkaConfig)
    AutoModel.register(DeltaMatryoshkaConfig, DeltaMatryoshkaModel)
    AutoModelForMaskedLM.register(DeltaMatryoshkaConfig, DeltaMatryoshkaForMaskedLM)
    
    print("Registered Delta-Matryoshka with HuggingFace Auto classes")


# Auto-register on import if HF is available
if HF_AVAILABLE:
    try:
        register_with_auto_classes()
    except Exception:
        pass  # May fail if already registered

