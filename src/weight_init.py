"""
Hierarchical Weight Initialization for Delta-Matryoshka++

Supports building larger models on top of smaller pretrained models:
- RexBERT-mini (68M) -> base slice
- RexBERT-base (150M) -> mini + delta
- RexBERT-large (400M) -> base + delta

Key idea: The smallest model (mini) forms the "base" slice that all sub-models share.
Larger models add delta channels/heads/layers on top.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class RexBERTConfig:
    """Configuration for RexBERT model variants."""
    name: str
    path: str
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


# Known RexBERT configurations
REXBERT_CONFIGS = {
    "mini": RexBERTConfig(
        name="mini",
        path="thebajajra/RexBERT-mini",
        hidden_size=384,
        num_layers=6,
        num_heads=6,
        intermediate_size=1536,
    ),
    "base": RexBERTConfig(
        name="base", 
        path="thebajajra/RexBERT-base",
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
    ),
    "large": RexBERTConfig(
        name="large",
        path="thebajajra/RexBERT-large", 
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        intermediate_size=4096,
    ),
}


def get_rexbert_config(name_or_path: str) -> RexBERTConfig:
    """Get RexBERT config from name or path."""
    # Check if it's a known name
    name = name_or_path.lower()
    for key, cfg in REXBERT_CONFIGS.items():
        if key in name or key in name_or_path.lower():
            return cfg
    
    # Try to load from HuggingFace
    try:
        hf_config = AutoConfig.from_pretrained(name_or_path)
        return RexBERTConfig(
            name=name_or_path.split("/")[-1],
            path=name_or_path,
            hidden_size=hf_config.hidden_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
        )
    except Exception as e:
        raise ValueError(f"Could not load config for {name_or_path}: {e}")


def init_hierarchical_delta(
    delta_model: nn.Module,
    base_model_path: str = "thebajajra/RexBERT-mini",
    target_model_path: Optional[str] = "thebajajra/RexBERT-base",
    delta_init_std: float = 0.02,
    init_delta_from_target: bool = False,
    verbose: bool = True
) -> nn.Module:
    """
    Initialize DeltaEncoder with hierarchical weight loading.
    
    The base model (e.g., mini) provides weights for the base slices.
    The target model (e.g., base/large) optionally provides weights for delta slices.
    
    Args:
        delta_model: DeltaEncoder to initialize
        base_model_path: Path to base model (smallest, e.g., mini)
        target_model_path: Path to target model (larger, e.g., base/large)
        delta_init_std: Std for random delta initialization
        init_delta_from_target: If True, init delta slices from target model weights
        verbose: Print progress
        
    Returns:
        Initialized DeltaEncoder
    """
    base_cfg = get_rexbert_config(base_model_path)
    target_cfg = get_rexbert_config(target_model_path) if target_model_path else None
    
    if verbose:
        print(f"Hierarchical Delta Initialization")
        print(f"  Base model: {base_cfg.name} ({base_cfg.path})")
        print(f"    hidden={base_cfg.hidden_size}, layers={base_cfg.num_layers}, heads={base_cfg.num_heads}")
        if target_cfg:
            print(f"  Target model: {target_cfg.name} ({target_cfg.path})")
            print(f"    hidden={target_cfg.hidden_size}, layers={target_cfg.num_layers}, heads={target_cfg.num_heads}")
    
    # Load base model
    if verbose:
        print(f"\nLoading base model weights from {base_model_path}...")
    base_model = AutoModel.from_pretrained(base_model_path)
    
    # Load target model if using for delta init
    target_model = None
    if init_delta_from_target and target_model_path:
        if verbose:
            print(f"Loading target model weights from {target_model_path}...")
        target_model = AutoModel.from_pretrained(target_model_path)
    
    with torch.no_grad():
        # Initialize embeddings
        _init_embeddings_hierarchical(
            delta_model, base_model, target_model,
            base_cfg, target_cfg, delta_init_std, verbose
        )
        
        # Initialize layers
        _init_layers_hierarchical(
            delta_model, base_model, target_model,
            base_cfg, target_cfg, delta_init_std, verbose
        )
        
        # Tie LM head
        if hasattr(delta_model, 'tie_weights'):
            delta_model.tie_weights()
    
    if verbose:
        print("\nHierarchical initialization complete!")
        _print_model_summary(delta_model, base_cfg, target_cfg)
    
    return delta_model


def _init_embeddings_hierarchical(
    delta_model: nn.Module,
    base_model: nn.Module,
    target_model: Optional[nn.Module],
    base_cfg: RexBERTConfig,
    target_cfg: Optional[RexBERTConfig],
    delta_init_std: float,
    verbose: bool
):
    """Initialize embeddings with dimension expansion support."""
    if verbose:
        print("  Initializing embeddings...")
    
    delta_emb = delta_model.embeddings
    base_emb = base_model.embeddings
    target_emb = target_model.embeddings if target_model else None
    
    delta_hidden = delta_emb.word_embeddings.weight.shape[1]
    base_hidden = base_cfg.hidden_size
    
    # Word embeddings
    base_vocab = base_emb.word_embeddings.weight.shape[0]
    delta_vocab = delta_emb.word_embeddings.weight.shape[0]
    copy_vocab = min(base_vocab, delta_vocab)
    
    if delta_hidden == base_hidden:
        # Same hidden size - direct copy
        delta_emb.word_embeddings.weight[:copy_vocab].copy_(
            base_emb.word_embeddings.weight[:copy_vocab]
        )
    else:
        # Different hidden size - need projection or partial copy
        copy_dim = min(delta_hidden, base_hidden)
        delta_emb.word_embeddings.weight[:copy_vocab, :copy_dim].copy_(
            base_emb.word_embeddings.weight[:copy_vocab, :copy_dim]
        )
        
        # Initialize expanded dimensions from target or random
        if delta_hidden > base_hidden:
            if target_emb and target_cfg.hidden_size >= delta_hidden:
                target_vocab = target_emb.word_embeddings.weight.shape[0]
                copy_vocab_t = min(delta_vocab, target_vocab)
                delta_emb.word_embeddings.weight[:copy_vocab_t, base_hidden:delta_hidden].copy_(
                    target_emb.word_embeddings.weight[:copy_vocab_t, base_hidden:delta_hidden]
                )
            else:
                nn.init.normal_(
                    delta_emb.word_embeddings.weight[:, base_hidden:],
                    std=delta_init_std
                )
    
    # Position embeddings
    base_pos = base_emb.position_embeddings.weight.shape[0]
    delta_pos = delta_emb.position_embeddings.weight.shape[0]
    copy_pos = min(base_pos, delta_pos)
    
    if delta_hidden == base_hidden:
        delta_emb.position_embeddings.weight[:copy_pos].copy_(
            base_emb.position_embeddings.weight[:copy_pos]
        )
    else:
        copy_dim = min(delta_hidden, base_hidden)
        delta_emb.position_embeddings.weight[:copy_pos, :copy_dim].copy_(
            base_emb.position_embeddings.weight[:copy_pos, :copy_dim]
        )
        if delta_hidden > base_hidden:
            if target_emb and target_cfg.hidden_size >= delta_hidden:
                target_pos = target_emb.position_embeddings.weight.shape[0]
                copy_pos_t = min(delta_pos, target_pos)
                delta_emb.position_embeddings.weight[:copy_pos_t, base_hidden:delta_hidden].copy_(
                    target_emb.position_embeddings.weight[:copy_pos_t, base_hidden:delta_hidden]
                )
            else:
                nn.init.normal_(
                    delta_emb.position_embeddings.weight[:, base_hidden:],
                    std=delta_init_std
                )
    
    # LayerNorm
    if hasattr(base_emb, 'LayerNorm') and hasattr(delta_emb, 'ln'):
        copy_dim = min(delta_hidden, base_hidden)
        delta_emb.ln.weight[:copy_dim].copy_(base_emb.LayerNorm.weight[:copy_dim])
        delta_emb.ln.bias[:copy_dim].copy_(base_emb.LayerNorm.bias[:copy_dim])
        
        if delta_hidden > base_hidden:
            nn.init.ones_(delta_emb.ln.weight[base_hidden:])
            nn.init.zeros_(delta_emb.ln.bias[base_hidden:])


def _init_layers_hierarchical(
    delta_model: nn.Module,
    base_model: nn.Module,
    target_model: Optional[nn.Module],
    base_cfg: RexBERTConfig,
    target_cfg: Optional[RexBERTConfig],
    delta_init_std: float,
    verbose: bool
):
    """Initialize transformer layers with hierarchical weight loading."""
    if verbose:
        print("  Initializing transformer layers...")
    
    num_delta_layers = len(delta_model.layers)
    num_base_layers = base_cfg.num_layers
    num_target_layers = target_cfg.num_layers if target_cfg else num_delta_layers
    
    # Compute base dimensions for slicing
    delta_hidden = delta_model.hidden_size
    base_hidden = base_cfg.hidden_size
    
    # Base heads/channels
    base_heads = base_cfg.num_heads
    base_intermediate = base_cfg.intermediate_size
    
    if verbose:
        print(f"    Delta model: {num_delta_layers} layers, {delta_hidden} hidden")
        print(f"    Base slice: {num_base_layers} layers, {base_hidden} hidden, {base_heads} heads")
    
    for layer_idx in range(num_delta_layers):
        delta_layer = delta_model.layers[layer_idx]
        
        # Determine which base/target layer to use
        base_layer = None
        target_layer = None
        
        if layer_idx < num_base_layers:
            base_layer = base_model.encoder.layer[layer_idx]
        
        if target_model and layer_idx < num_target_layers:
            target_layer = target_model.encoder.layer[layer_idx]
        
        _init_single_layer_hierarchical(
            delta_layer=delta_layer,
            base_layer=base_layer,
            target_layer=target_layer,
            base_cfg=base_cfg,
            target_cfg=target_cfg,
            delta_hidden=delta_hidden,
            delta_init_std=delta_init_std,
            layer_idx=layer_idx,
            is_delta_only=(layer_idx >= num_base_layers)
        )


def _init_single_layer_hierarchical(
    delta_layer: nn.Module,
    base_layer: Optional[nn.Module],
    target_layer: Optional[nn.Module],
    base_cfg: RexBERTConfig,
    target_cfg: Optional[RexBERTConfig],
    delta_hidden: int,
    delta_init_std: float,
    layer_idx: int,
    is_delta_only: bool
):
    """Initialize a single transformer layer with hierarchical weights."""
    
    if is_delta_only:
        # This layer is beyond base model - initialize from target or random
        if target_layer:
            _copy_layer_from_target(delta_layer, target_layer, target_cfg, delta_hidden, delta_init_std)
        else:
            _init_layer_random(delta_layer, delta_init_std)
        return
    
    # This layer has base weights - copy base to base slice, init delta
    base_hidden = base_cfg.hidden_size
    base_heads = base_cfg.num_heads
    base_head_dim = base_hidden // base_heads
    base_intermediate = base_cfg.intermediate_size
    
    delta_attn = delta_layer.attn
    delta_mlp = delta_layer.mlp
    
    base_attn = base_layer.attention
    base_attn_self = base_attn.self if hasattr(base_attn, 'self') else base_attn
    base_intermediate_layer = base_layer.intermediate if hasattr(base_layer, 'intermediate') else None
    base_output = base_layer.output if hasattr(base_layer, 'output') else None
    
    # Target layer components (optional)
    target_attn_self = None
    target_intermediate = None
    target_output = None
    if target_layer:
        target_attn = target_layer.attention
        target_attn_self = target_attn.self if hasattr(target_attn, 'self') else target_attn
        target_intermediate = target_layer.intermediate if hasattr(target_layer, 'intermediate') else None
        target_output = target_layer.output if hasattr(target_layer, 'output') else None
    
    # === Attention Q, K, V ===
    for proj_name, base_attr in [('q', 'query'), ('k', 'key'), ('v', 'value')]:
        delta_proj = getattr(delta_attn, proj_name)
        base_proj = getattr(base_attn_self, base_attr, None)
        target_proj = getattr(target_attn_self, base_attr, None) if target_attn_self else None
        
        if base_proj is not None:
            _copy_qkv_hierarchical(
                delta_proj, base_proj, target_proj,
                base_hidden, base_heads, base_head_dim,
                delta_hidden, delta_init_std
            )
    
    # === Attention Output ===
    base_out_dense = base_attn.output.dense if hasattr(base_attn, 'output') else None
    target_out_dense = target_layer.attention.output.dense if target_layer and hasattr(target_layer.attention, 'output') else None
    
    if base_out_dense is not None:
        _copy_output_proj_hierarchical(
            delta_attn.o, base_out_dense, target_out_dense,
            base_hidden, base_heads, base_head_dim,
            delta_hidden, delta_init_std
        )
    
    # === MLP FC1 ===
    if base_intermediate_layer and hasattr(base_intermediate_layer, 'dense'):
        base_fc1 = base_intermediate_layer.dense
        target_fc1 = target_intermediate.dense if target_intermediate and hasattr(target_intermediate, 'dense') else None
        
        _copy_mlp_fc1_hierarchical(
            delta_mlp.fc1, base_fc1, target_fc1,
            base_hidden, base_intermediate,
            delta_hidden, delta_init_std
        )
    
    # === MLP FC2 ===
    if base_output and hasattr(base_output, 'dense'):
        base_fc2 = base_output.dense
        target_fc2 = target_output.dense if target_output and hasattr(target_output, 'dense') else None
        
        _copy_mlp_fc2_hierarchical(
            delta_mlp.fc2, base_fc2, target_fc2,
            base_hidden, base_intermediate,
            delta_hidden, delta_init_std
        )
    
    # === LayerNorms ===
    # LN1 (attention)
    if hasattr(base_attn, 'output') and hasattr(base_attn.output, 'LayerNorm'):
        base_ln = base_attn.output.LayerNorm
        if hasattr(delta_layer.ln1, 'base_ln'):
            copy_dim = min(delta_hidden, base_hidden)
            delta_layer.ln1.base_ln.weight[:copy_dim].copy_(base_ln.weight[:copy_dim])
            delta_layer.ln1.base_ln.bias[:copy_dim].copy_(base_ln.bias[:copy_dim])
            if delta_hidden > base_hidden:
                nn.init.ones_(delta_layer.ln1.base_ln.weight[base_hidden:])
                nn.init.zeros_(delta_layer.ln1.base_ln.bias[base_hidden:])
    
    # LN2 (MLP)
    if base_output and hasattr(base_output, 'LayerNorm'):
        base_ln = base_output.LayerNorm
        if hasattr(delta_layer.ln2, 'base_ln'):
            copy_dim = min(delta_hidden, base_hidden)
            delta_layer.ln2.base_ln.weight[:copy_dim].copy_(base_ln.weight[:copy_dim])
            delta_layer.ln2.base_ln.bias[:copy_dim].copy_(base_ln.bias[:copy_dim])
            if delta_hidden > base_hidden:
                nn.init.ones_(delta_layer.ln2.base_ln.weight[base_hidden:])
                nn.init.zeros_(delta_layer.ln2.base_ln.bias[base_hidden:])


def _copy_qkv_hierarchical(
    delta_proj: nn.Linear,
    base_proj: nn.Linear,
    target_proj: Optional[nn.Linear],
    base_hidden: int,
    base_heads: int,
    base_head_dim: int,
    delta_hidden: int,
    delta_init_std: float
):
    """Copy Q/K/V projection with dimension expansion."""
    # Base weights shape: (base_hidden, base_hidden)
    # Delta weights shape: (delta_hidden, delta_hidden)
    
    base_dim = base_heads * base_head_dim  # Output dimension for base heads
    copy_in = min(delta_hidden, base_hidden)
    copy_out = min(delta_proj.weight.shape[0], base_dim)
    
    # Copy base slice
    delta_proj.weight[:copy_out, :copy_in].copy_(
        base_proj.weight[:copy_out, :copy_in]
    )
    delta_proj.bias[:copy_out].copy_(base_proj.bias[:copy_out])
    
    # Initialize delta dimensions
    # Delta heads (rows beyond base_dim)
    if delta_proj.weight.shape[0] > base_dim:
        if target_proj is not None:
            # Copy from target model for delta heads
            target_out = target_proj.weight.shape[0]
            copy_delta_out = min(delta_proj.weight.shape[0] - base_dim, target_out - base_dim)
            if copy_delta_out > 0:
                delta_proj.weight[base_dim:base_dim+copy_delta_out, :copy_in].copy_(
                    target_proj.weight[base_dim:base_dim+copy_delta_out, :copy_in]
                )
                delta_proj.bias[base_dim:base_dim+copy_delta_out].copy_(
                    target_proj.bias[base_dim:base_dim+copy_delta_out]
                )
        else:
            nn.init.xavier_uniform_(delta_proj.weight[base_dim:, :copy_in], gain=delta_init_std)
            nn.init.zeros_(delta_proj.bias[base_dim:])
    
    # Expanded input dimension (columns beyond base_hidden)
    if delta_hidden > base_hidden:
        if target_proj is not None and target_proj.weight.shape[1] >= delta_hidden:
            delta_proj.weight[:, base_hidden:delta_hidden].copy_(
                target_proj.weight[:delta_proj.weight.shape[0], base_hidden:delta_hidden]
            )
        else:
            nn.init.xavier_uniform_(delta_proj.weight[:, base_hidden:], gain=delta_init_std)


def _copy_output_proj_hierarchical(
    delta_o: nn.Linear,
    base_o: nn.Linear,
    target_o: Optional[nn.Linear],
    base_hidden: int,
    base_heads: int,
    base_head_dim: int,
    delta_hidden: int,
    delta_init_std: float
):
    """Copy output projection with dimension expansion."""
    base_in_dim = base_heads * base_head_dim
    copy_in = min(delta_o.weight.shape[1], base_in_dim)
    copy_out = min(delta_hidden, base_hidden)
    
    # Copy base slice
    delta_o.weight[:copy_out, :copy_in].copy_(base_o.weight[:copy_out, :copy_in])
    delta_o.bias[:copy_out].copy_(base_o.bias[:copy_out])
    
    # Delta columns (input from delta heads)
    if delta_o.weight.shape[1] > base_in_dim:
        if target_o is not None:
            target_in = target_o.weight.shape[1]
            copy_delta_in = min(delta_o.weight.shape[1] - base_in_dim, target_in - base_in_dim)
            if copy_delta_in > 0:
                delta_o.weight[:copy_out, base_in_dim:base_in_dim+copy_delta_in].copy_(
                    target_o.weight[:copy_out, base_in_dim:base_in_dim+copy_delta_in]
                )
        else:
            nn.init.xavier_uniform_(delta_o.weight[:copy_out, base_in_dim:], gain=delta_init_std)
    
    # Delta rows (expanded hidden dim)
    if delta_hidden > base_hidden:
        if target_o is not None and target_o.weight.shape[0] >= delta_hidden:
            delta_o.weight[base_hidden:delta_hidden, :].copy_(
                target_o.weight[base_hidden:delta_hidden, :delta_o.weight.shape[1]]
            )
            delta_o.bias[base_hidden:delta_hidden].copy_(
                target_o.bias[base_hidden:delta_hidden]
            )
        else:
            nn.init.xavier_uniform_(delta_o.weight[base_hidden:, :], gain=delta_init_std)
            nn.init.zeros_(delta_o.bias[base_hidden:])


def _copy_mlp_fc1_hierarchical(
    delta_fc1: nn.Linear,
    base_fc1: nn.Linear,
    target_fc1: Optional[nn.Linear],
    base_hidden: int,
    base_intermediate: int,
    delta_hidden: int,
    delta_init_std: float
):
    """Copy MLP fc1 (hidden -> intermediate) with expansion."""
    delta_intermediate = delta_fc1.weight.shape[0]
    
    copy_in = min(delta_hidden, base_hidden)
    copy_out = min(delta_intermediate, base_intermediate)
    
    # Copy base slice
    delta_fc1.weight[:copy_out, :copy_in].copy_(base_fc1.weight[:copy_out, :copy_in])
    delta_fc1.bias[:copy_out].copy_(base_fc1.bias[:copy_out])
    
    # Delta intermediate channels (rows beyond base)
    if delta_intermediate > base_intermediate:
        if target_fc1 is not None:
            target_inter = target_fc1.weight.shape[0]
            copy_delta = min(delta_intermediate - base_intermediate, target_inter - base_intermediate)
            if copy_delta > 0:
                delta_fc1.weight[base_intermediate:base_intermediate+copy_delta, :copy_in].copy_(
                    target_fc1.weight[base_intermediate:base_intermediate+copy_delta, :copy_in]
                )
                delta_fc1.bias[base_intermediate:base_intermediate+copy_delta].copy_(
                    target_fc1.bias[base_intermediate:base_intermediate+copy_delta]
                )
        else:
            nn.init.xavier_uniform_(delta_fc1.weight[base_intermediate:, :copy_in], gain=delta_init_std)
            nn.init.zeros_(delta_fc1.bias[base_intermediate:])
    
    # Expanded hidden dim (columns beyond base)
    if delta_hidden > base_hidden:
        if target_fc1 is not None and target_fc1.weight.shape[1] >= delta_hidden:
            delta_fc1.weight[:, base_hidden:delta_hidden].copy_(
                target_fc1.weight[:delta_intermediate, base_hidden:delta_hidden]
            )
        else:
            nn.init.xavier_uniform_(delta_fc1.weight[:, base_hidden:], gain=delta_init_std)


def _copy_mlp_fc2_hierarchical(
    delta_fc2: nn.Linear,
    base_fc2: nn.Linear,
    target_fc2: Optional[nn.Linear],
    base_hidden: int,
    base_intermediate: int,
    delta_hidden: int,
    delta_init_std: float
):
    """Copy MLP fc2 (intermediate -> hidden) with expansion."""
    delta_intermediate = delta_fc2.weight.shape[1]
    
    copy_in = min(delta_intermediate, base_intermediate)
    copy_out = min(delta_hidden, base_hidden)
    
    # Copy base slice
    delta_fc2.weight[:copy_out, :copy_in].copy_(base_fc2.weight[:copy_out, :copy_in])
    delta_fc2.bias[:copy_out].copy_(base_fc2.bias[:copy_out])
    
    # Delta intermediate input (columns beyond base)
    if delta_intermediate > base_intermediate:
        if target_fc2 is not None:
            target_inter = target_fc2.weight.shape[1]
            copy_delta = min(delta_intermediate - base_intermediate, target_inter - base_intermediate)
            if copy_delta > 0:
                delta_fc2.weight[:copy_out, base_intermediate:base_intermediate+copy_delta].copy_(
                    target_fc2.weight[:copy_out, base_intermediate:base_intermediate+copy_delta]
                )
        else:
            nn.init.xavier_uniform_(delta_fc2.weight[:copy_out, base_intermediate:], gain=delta_init_std)
    
    # Expanded hidden output (rows beyond base)
    if delta_hidden > base_hidden:
        if target_fc2 is not None and target_fc2.weight.shape[0] >= delta_hidden:
            delta_fc2.weight[base_hidden:delta_hidden, :].copy_(
                target_fc2.weight[base_hidden:delta_hidden, :delta_intermediate]
            )
            delta_fc2.bias[base_hidden:delta_hidden].copy_(
                target_fc2.bias[base_hidden:delta_hidden]
            )
        else:
            nn.init.xavier_uniform_(delta_fc2.weight[base_hidden:, :], gain=delta_init_std)
            nn.init.zeros_(delta_fc2.bias[base_hidden:])


def _copy_layer_from_target(
    delta_layer: nn.Module,
    target_layer: nn.Module,
    target_cfg: RexBERTConfig,
    delta_hidden: int,
    delta_init_std: float
):
    """Copy entire layer from target model (for layers beyond base)."""
    target_attn = target_layer.attention
    target_attn_self = target_attn.self if hasattr(target_attn, 'self') else target_attn
    
    target_hidden = target_cfg.hidden_size
    copy_dim = min(delta_hidden, target_hidden)
    
    # Q, K, V
    for proj_name, target_attr in [('q', 'query'), ('k', 'key'), ('v', 'value')]:
        delta_proj = getattr(delta_layer.attn, proj_name)
        target_proj = getattr(target_attn_self, target_attr)
        
        delta_proj.weight[:copy_dim, :copy_dim].copy_(target_proj.weight[:copy_dim, :copy_dim])
        delta_proj.bias[:copy_dim].copy_(target_proj.bias[:copy_dim])
        
        if delta_hidden > target_hidden:
            nn.init.xavier_uniform_(delta_proj.weight[target_hidden:, :], gain=delta_init_std)
            nn.init.xavier_uniform_(delta_proj.weight[:, target_hidden:], gain=delta_init_std)
            nn.init.zeros_(delta_proj.bias[target_hidden:])
    
    # Output projection
    if hasattr(target_attn, 'output'):
        target_o = target_attn.output.dense
        delta_layer.attn.o.weight[:copy_dim, :copy_dim].copy_(target_o.weight[:copy_dim, :copy_dim])
        delta_layer.attn.o.bias[:copy_dim].copy_(target_o.bias[:copy_dim])
    
    # MLP
    target_inter = target_layer.intermediate.dense if hasattr(target_layer, 'intermediate') else None
    target_out = target_layer.output.dense if hasattr(target_layer, 'output') else None
    
    if target_inter:
        target_intermediate = target_inter.weight.shape[0]
        delta_intermediate = delta_layer.mlp.fc1.weight.shape[0]
        copy_inter = min(delta_intermediate, target_intermediate)
        
        delta_layer.mlp.fc1.weight[:copy_inter, :copy_dim].copy_(target_inter.weight[:copy_inter, :copy_dim])
        delta_layer.mlp.fc1.bias[:copy_inter].copy_(target_inter.bias[:copy_inter])
    
    if target_out:
        delta_layer.mlp.fc2.weight[:copy_dim, :copy_inter].copy_(target_out.weight[:copy_dim, :copy_inter])
        delta_layer.mlp.fc2.bias[:copy_dim].copy_(target_out.bias[:copy_dim])


def _init_layer_random(delta_layer: nn.Module, delta_init_std: float):
    """Initialize layer with random weights."""
    for name, param in delta_layer.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param, gain=delta_init_std)
        elif 'bias' in name:
            nn.init.zeros_(param)


def _print_model_summary(
    delta_model: nn.Module,
    base_cfg: RexBERTConfig,
    target_cfg: Optional[RexBERTConfig]
):
    """Print summary of initialized model."""
    total_params = sum(p.numel() for p in delta_model.parameters())
    
    # Estimate base params (rough)
    base_params = (
        base_cfg.hidden_size * 30522 +  # word embeddings
        base_cfg.hidden_size * 512 +     # position embeddings
        base_cfg.num_layers * (
            4 * base_cfg.hidden_size * base_cfg.hidden_size +  # attention
            2 * base_cfg.hidden_size * base_cfg.intermediate_size  # MLP
        )
    )
    delta_params = total_params - base_params
    
    print(f"\n  Model Summary:")
    print(f"    Total parameters: {total_params:,}")
    print(f"    Base (from {base_cfg.name}): ~{base_params:,}")
    print(f"    Delta (added): ~{delta_params:,}")
    print(f"    Delta ratio: {delta_params/total_params:.1%}")


# Legacy function for backward compatibility
def load_rexbert_to_delta(
    delta_model: nn.Module,
    rexbert_path: str = "thebajajra/RexBERT-base",
    base_ratio: float = 0.5,
    base_heads: int = 6,
    delta_init_std: float = 0.02,
    verbose: bool = True
) -> nn.Module:
    """
    Legacy function - loads RexBERT weights into base slices.
    
    For new code, use init_hierarchical_delta() instead.
    """
    return init_hierarchical_delta(
        delta_model=delta_model,
        base_model_path=rexbert_path,
        target_model_path=None,
        delta_init_std=delta_init_std,
        init_delta_from_target=False,
        verbose=verbose
    )


def verify_hierarchical_loading(
    delta_model: nn.Module,
    base_path: str,
    target_path: Optional[str] = None
) -> Dict[str, bool]:
    """Verify hierarchical weight loading."""
    base_model = AutoModel.from_pretrained(base_path)
    target_model = AutoModel.from_pretrained(target_path) if target_path else None
    
    results = {}
    
    with torch.no_grad():
        # Check embeddings
        base_emb = base_model.embeddings.word_embeddings.weight
        delta_emb = delta_model.embeddings.word_embeddings.weight
        base_hidden = base_emb.shape[1]
        delta_hidden = delta_emb.shape[1]
        copy_vocab = min(base_emb.shape[0], delta_emb.shape[0])
        copy_dim = min(base_hidden, delta_hidden)
        
        results["embeddings"] = torch.allclose(
            base_emb[:copy_vocab, :copy_dim],
            delta_emb[:copy_vocab, :copy_dim],
            atol=1e-5
        )
        
        # Check first layer Q
        base_q = base_model.encoder.layer[0].attention.self.query.weight
        delta_q = delta_model.layers[0].attn.q.weight
        copy_out = min(base_q.shape[0], delta_q.shape[0])
        copy_in = min(base_q.shape[1], delta_q.shape[1])
        
        results["layer0_q"] = torch.allclose(
            base_q[:copy_out, :copy_in],
            delta_q[:copy_out, :copy_in],
            atol=1e-5
        )
    
    print("Verification results:")
    for key, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {key}: {status}")
    
    return results
