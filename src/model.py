from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .droppath import DropPath

# Check for Flash Attention / SDPA support (PyTorch 2.0+)
HAS_FLASH_ATTN = hasattr(F, 'scaled_dot_product_attention')

# ---------------- Budget spec ----------------

@dataclass
class Budget:
    width: float      # ratio in (0,1]
    heads: int        # number of heads to keep
    depth: int        # number of layers to keep

# ---------------- Budget-aware LayerNorm ----------------

class BudgetAwareLayerNorm(nn.Module):
    """LayerNorm whose affine params are produced by a tiny hypernet from budget code.
    Fallbacks to standard LayerNorm if hidden_hyper=0 or disabled."""
    def __init__(self, hidden_size: int, eps: float = 1e-5, enable: bool = True, hyper_hidden: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.enable = enable and hyper_hidden > 0
        self.base_ln = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=not self.enable)
        if self.enable:
            self.hyper = nn.Sequential(
                nn.Linear(3, hyper_hidden), nn.GELU(),
                nn.Linear(hyper_hidden, 2*hidden_size)  # gamma, beta
            )

    def forward(self, x, budget: Optional[Budget] = None):
        x = self.base_ln(x)
        if not self.enable or budget is None:
            return x
        # budget code: (width, heads_norm, depth_norm)
        b = torch.tensor([budget.width, budget.heads_norm, budget.depth_norm], device=x.device, dtype=x.dtype)
        gamma_beta = self.hyper(b).view(2, self.hidden_size)
        gamma, beta = gamma_beta[0], gamma_beta[1]
        # broadcast to (B, T, H)
        return x * gamma + beta

# ---------------- Delta MLP ----------------

class DeltaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, base_ratio: float = 0.5, dropout: float = 0.1,
                 dds_mlp=None):
        """
        Args:
            hidden_size: Model hidden dimension
            intermediate_size: MLP intermediate dimension
            base_ratio: Fraction of channels in base slice
            dropout: Dropout probability
            dds_mlp: Optional DeltaDictionaryMLP for DDS-based delta weights
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.base_ratio = base_ratio
        self.dds_mlp = dds_mlp

        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, width_ratio: float, token_delta_mask: Optional[torch.Tensor] = None,
                return_delta: bool = False, delta_only: bool = False, use_dds: bool = False):
        """x: (B, T, H). width_ratio in (0,1].
        token_delta_mask: (B, T) bool selecting tokens where delta is allowed (or None for all).
        return_delta: if True, also returns delta-only contribution for Δ-loss.
        delta_only: if True, compute only delta contribution (no base path).
        use_dds: if True and dds_mlp is set, use DDS for delta weights."""
        B, T, H = x.shape
        inter = self.intermediate_size
        base = int(self.base_ratio * inter)
        eff = int(max(1, round(width_ratio * inter)))
        delta_take = max(0, eff - base)
        
        # If using DDS for delta weights
        if use_dds and self.dds_mlp is not None and delta_take > 0:
            return self._forward_with_dds(x, base, eff, delta_take, token_delta_mask, return_delta, delta_only)
        
        # Standard forward path
        # Slices
        x_proj = self.fc1(x)  # (B, T, inter) full projection (training-time convenience)
        x_act = self.act(x_proj)
        y_base = 0.0
        y_delta = 0.0
        if not delta_only:
            y_base = x_act[:, :, :min(base, eff)]
        if delta_take > 0:
            y_delta = x_act[:, :, base:base+delta_take]
        # Concatenate base + delta view to match eff size and project back
        if delta_only:
            y = y_delta
        else:
            if isinstance(y_base, float):  # no base (eff < base)
                y = y_delta
            elif isinstance(y_delta, float) or delta_take == 0:
                y = y_base
            else:
                y = torch.cat([y_base, y_delta], dim=-1)
        # Pad with zeros if eff < base
        if y.shape[-1] < eff:
            pad = eff - y.shape[-1]
            y = torch.cat([y, y.new_zeros(B, T, pad)], dim=-1)
        # Back projection uses matching slice of fc2
        W2 = self.fc2.weight[:, :eff]
        b2 = self.fc2.bias
        out = torch.matmul(y, W2.t()) + b2
        if token_delta_mask is not None and delta_take > 0:
            # Compute delta-only projection for mask gating
            y_d = y_delta
            W2d = self.fc2.weight[:, base:base+delta_take]
            out_delta = torch.matmul(y_d, W2d.t())
            # Apply only on masked tokens
            mask = token_delta_mask.unsqueeze(-1)  # (B, T, 1)
            out = out + torch.where(mask, out_delta, out_delta.new_zeros(out_delta.shape))
        out = self.dropout(out)
        if return_delta:
            # delta-only logits (projected to H) for Δ-loss at head
            if delta_take > 0:
                out_delta = torch.matmul(y_delta, self.fc2.weight[:, base:base+delta_take].t())
            else:
                out_delta = out.new_zeros(B, T, H)
            return out, out_delta
        return out
    
    def _forward_with_dds(self, x, base, eff, delta_take, token_delta_mask, return_delta, delta_only):
        """Forward pass using DDS-reconstructed delta weights."""
        B, T, H = x.shape
        
        # Base path through standard fc1
        x_proj_base = F.linear(x, self.fc1.weight[:base], self.fc1.bias[:base])
        x_act_base = self.act(x_proj_base)
        
        # Delta path through DDS-reconstructed weights
        delta_out = self.dds_mlp.forward_delta(x, x_act_base)
        
        if delta_only:
            out = delta_out
        else:
            # Base output
            out_base = F.linear(x_act_base, self.fc2.weight[:, :base], self.fc2.bias)
            out = out_base + delta_out
        
        if token_delta_mask is not None:
            # Gate delta contribution per token
            mask = token_delta_mask.unsqueeze(-1)
            out = torch.where(mask, out, out - delta_out)
        
        out = self.dropout(out)
        
        if return_delta:
            return out, delta_out
        return out

# ---------------- Delta Attention ----------------

class DeltaSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1, base_heads: int = 0,
                 dds_attn=None):
        """
        Args:
            hidden_size: Model hidden dimension
            num_heads: Total number of attention heads
            dropout: Dropout probability
            base_heads: Number of heads in base slice
            dds_attn: Optional DeltaDictionaryAttention for DDS-based delta weights
        """
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.base_heads = base_heads if base_heads > 0 else max(1, num_heads // 2)
        self.dds_attn = dds_attn

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def _shape(self, x):
        B, T, H = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, T, dim)
        return x

    def forward(self, x, keep_heads: int, attn_mask: Optional[torch.Tensor] = None,
                token_delta_mask: Optional[torch.Tensor] = None, return_delta: bool = False,
                delta_only: bool = False, use_dds: bool = False):
        B, T, H = x.shape
        q = self._shape(self.q(x))
        k = self._shape(self.k(x))
        v = self._shape(self.v(x))

        base_h = min(self.base_heads, keep_heads)
        delta_take = max(0, keep_heads - base_h)

        # Select slices
        if delta_only:
            h_sel = slice(base_h, base_h + delta_take)
            q_ = q[:, h_sel]
            k_ = k[:, h_sel]
            v_ = v[:, h_sel]
            y_base = 0.0
        else:
            q_base = q[:, :base_h]
            k_base = k[:, :base_h]
            v_base = v[:, :base_h]
            y_base = self._attend(q_base, k_base, v_base, attn_mask)

            if delta_take > 0:
                q_delta = q[:, base_h:base_h+delta_take]
                k_delta = k[:, base_h:base_h+delta_take]
                v_delta = v[:, base_h:base_h+delta_take]
                y_delta = self._attend(q_delta, k_delta, v_delta, attn_mask)
            else:
                y_delta = 0.0

        if delta_only:
            y = self._attend(q_, k_, v_, attn_mask)
            y_delta = y
        else:
            if isinstance(y_delta, float):
                y = y_base
            elif isinstance(y_base, float):
                y = y_delta
            else:
                # Concatenate along heads then merge
                y = torch.cat([y_base, y_delta], dim=1)

        # Merge heads
        y = y.transpose(1, 2).contiguous().view(B, T, keep_heads * self.head_dim)
        # Project back using prefix of output projection
        W = self.o.weight[:keep_heads * self.head_dim, :]
        b = self.o.bias
        out = torch.matmul(y, W) + b

        if token_delta_mask is not None and not delta_only and delta_take > 0:
            # Compute delta-only projection and gate per token
            yd = y_delta if not isinstance(y_delta, float) else None
            if yd is not None:
                yd = yd.transpose(1,2).contiguous().view(B, T, delta_take * self.head_dim)
                Wd = self.o.weight[base_h * self.head_dim: (base_h + delta_take) * self.head_dim, :]
                out_delta = torch.matmul(yd, Wd)
                mask = token_delta_mask.unsqueeze(-1)
                out = out + torch.where(mask, out_delta, out_delta.new_zeros(out_delta.shape))
        out = self.proj_drop(out)

        if return_delta:
            if delta_only:
                delta_out = out
            else:
                if delta_take > 0 and not isinstance(y_delta, float):
                    yd = y_delta.transpose(1,2).contiguous().view(B, T, delta_take * self.head_dim)
                    Wd = self.o.weight[base_h * self.head_dim: (base_h + delta_take) * self.head_dim, :]
                    delta_out = torch.matmul(yd, Wd)
                else:
                    delta_out = out.new_zeros(out.shape)
            return out, delta_out
        return out

    def _attend(self, q, k, v, attn_mask=None, use_flash=True):
        # q,k,v: (B, h, T, d)
        # Use Flash Attention / SDPA when available (PyTorch 2.0+)
        if use_flash and HAS_FLASH_ATTN and self.training:
            # scaled_dot_product_attention expects: (B, h, T, d)
            # attn_mask for SDPA should be boolean or additive
            dropout_p = self.attn_drop.p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,  # Bidirectional encoder
            )
            return out
        
        # Fallback to manual attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q * scale) @ k.transpose(-2, -1)  # (B, h, T, T)
        if attn_mask is not None:
            attn = attn + attn_mask  # assume mask is additive with -inf for blocked
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # (B, h, T, d)
        return out

# ---------------- Transformer Layer ----------------

class DeltaTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, base_ratio=0.5, base_heads=0,
                 dropout=0.1, attn_dropout=0.1, ln_eps=1e-5,
                 budget_cond_ln=True, ln_hyper_hidden=64, drop_path_prob=0.0,
                 dds_mlp=None, dds_attn=None):
        super().__init__()
        self.ln1 = BudgetAwareLayerNorm(hidden_size, ln_eps, enable=budget_cond_ln, hyper_hidden=ln_hyper_hidden)
        self.attn = DeltaSelfAttention(hidden_size, num_heads, dropout=attn_dropout, base_heads=base_heads,
                                       dds_attn=dds_attn)
        self.drop_path1 = DropPath(drop_path_prob)
        self.ln2 = BudgetAwareLayerNorm(hidden_size, ln_eps, enable=budget_cond_ln, hyper_hidden=ln_hyper_hidden)
        self.mlp = DeltaMLP(hidden_size, intermediate_size, base_ratio=base_ratio, dropout=dropout,
                           dds_mlp=dds_mlp)
        self.drop_path2 = DropPath(drop_path_prob)
        
        # Store layer index for survival schedule
        self.layer_idx = 0

    def forward(self, x, budget: Budget, attention_mask=None,
                token_delta_mask: Optional[torch.Tensor] = None,
                return_delta: bool = False, delta_only: bool = False,
                use_dds: bool = False):
        # Self-attention
        h = self.ln1(x, budget)
        attn_out = self.attn(h, keep_heads=budget.heads, attn_mask=attention_mask,
                             token_delta_mask=token_delta_mask, return_delta=return_delta,
                             delta_only=delta_only, use_dds=use_dds)
        if return_delta:
            attn_out, attn_delta = attn_out
        x = x + self.drop_path1(attn_out)

        # MLP
        h2 = self.ln2(x, budget)
        mlp_out = self.mlp(h2, width_ratio=budget.width, token_delta_mask=token_delta_mask,
                           return_delta=return_delta, delta_only=delta_only, use_dds=use_dds)
        if return_delta:
            mlp_out, mlp_delta = mlp_out
        x = x + self.drop_path2(mlp_out)

        if return_delta:
            # delta contribution aggregated at hidden level
            delta_hidden = attn_delta + mlp_delta
            return x, delta_hidden
        return x

# ---------------- Embeddings ----------------

class SimpleEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings=512, dropout=0.1, pad_token_id=0):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.ln = nn.LayerNorm(hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.word_embeddings(input_ids) + self.position_embeddings(pos)
        x = self.ln(x)
        x = self.dropout(x)
        return x

# ---------------- Encoder ----------------

class DeltaEncoder(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                 intermediate_size=3072, dropout=0.1, attn_dropout=0.1, ln_eps=1e-5,
                 base_ratio=0.5, base_heads=6, drop_path=0.0, depth_floor=0,
                 budget_cond_ln=True, ln_hyper_hidden=64, pad_token_id=0, max_position_embeddings=512,
                 enable_dds=False, dds_num_atoms=16, dds_rank=64):
        super().__init__()
        self.embeddings = SimpleEmbeddings(vocab_size, hidden_size, max_position_embeddings, dropout, pad_token_id)
        
        # DDS Manager (optional)
        self.enable_dds = enable_dds
        self.dds_manager = None
        if enable_dds:
            from .dds import DDSManager
            self.dds_manager = DDSManager(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_attention_heads,
                num_layers=num_hidden_layers,
                base_ratio=base_ratio,
                base_heads=base_heads,
                num_atoms=dds_num_atoms,
                rank=dds_rank,
            )
        
        # Distribute droppath over depth linearly (stochastic depth schedule)
        dps = torch.linspace(0, drop_path, steps=num_hidden_layers).tolist()
        self.layers = nn.ModuleList([
            DeltaTransformerLayer(
                hidden_size, num_attention_heads, intermediate_size,
                base_ratio=base_ratio, base_heads=base_heads,
                dropout=dropout, attn_dropout=attn_dropout, ln_eps=ln_eps,
                budget_cond_ln=budget_cond_ln, ln_hyper_hidden=ln_hyper_hidden,
                drop_path_prob=dps[i],
                dds_mlp=self.dds_manager.mlp_deltas[i] if enable_dds else None,
                dds_attn=self.dds_manager.attn_deltas[i] if enable_dds else None,
            ) for i in range(num_hidden_layers)
        ])
        
        # Set layer indices for survival schedule
        for i, layer in enumerate(self.layers):
            layer.layer_idx = i
        
        self.ln_f = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.num_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.base_heads = base_heads
        self.base_ratio = base_ratio
        self.depth_floor = depth_floor

        # LM head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.tie_weights()
        
        # Gradient checkpointing (disabled by default)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory-efficient training."""
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def tie_weights(self):
        self.lm_head.weight = self.embeddings.word_embeddings.weight

    def budget_norms(self, budget: Budget):
        # Cached normalized codes in [0,1]
        budget.heads_norm = budget.heads / self.num_attention_heads if self.num_attention_heads > 0 else 1.0
        budget.depth_norm = budget.depth / self.num_layers

    def forward(self, input_ids, attention_mask=None, budget: Optional[Budget] = None,
                token_delta_mask: Optional[torch.Tensor] = None, return_hidden=False,
                return_delta=False, delta_only=False, use_dds: Optional[bool] = None,
                tap_layers: Optional[List[int]] = None):
        """
        Forward pass through the encoder.
        
        Args:
            input_ids: Input token IDs (B, T)
            attention_mask: Attention mask (B, T), 1 for valid, 0 for padding
            budget: Budget specification (width, heads, depth)
            token_delta_mask: Per-token mask for delta computation
            return_hidden: Return final hidden states
            return_delta: Return delta logits for residualization loss
            delta_only: Compute only delta contribution
            use_dds: Use DDS for delta weights (defaults to self.enable_dds)
            tap_layers: Layer indices to collect hidden states for CSCF
            
        Returns:
            Dict with 'logits', optionally 'hidden_states', 'delta_logits', 'tap_hiddens'
        """
        if budget is None:
            # default full budget
            budget = Budget(width=1.0, heads=self.layers[0].attn.num_heads, depth=self.num_layers)
        self.budget_norms(budget)
        
        if use_dds is None:
            use_dds = self.enable_dds
        
        B, T = input_ids.shape
        x = self.embeddings(input_ids)
        # Build attention mask (B, 1, 1, T) additive form if provided
        attn_bias = None
        if attention_mask is not None:
            # mask=1 for keep, 0 for pad
            mask = attention_mask[:, None, None, :]
            attn_bias = (1.0 - mask) * -1e4

        depth_keep = max(self.depth_floor, budget.depth)
        delta_hiddens = []
        tap_hiddens = [] if tap_layers else None

        for i, layer in enumerate(self.layers):
            if i >= depth_keep:
                break  # prefix depth dropout
            
            # Use gradient checkpointing if enabled (saves memory)
            if self.gradient_checkpointing and self.training and not return_delta:
                # Checkpointing doesn't work well with multiple returns, skip for return_delta
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(inputs[0], inputs[1], attention_mask=inputs[2],
                                    token_delta_mask=inputs[3], return_delta=False,
                                    delta_only=inputs[4], use_dds=inputs[5])
                    return custom_forward
                
                x = checkpoint(
                    create_custom_forward(layer),
                    x, budget, attn_bias, token_delta_mask, delta_only, use_dds,
                    use_reentrant=False
                )
            else:
                x = layer(x, budget, attention_mask=attn_bias,
                          token_delta_mask=token_delta_mask,
                          return_delta=return_delta, delta_only=delta_only,
                          use_dds=use_dds)
                if return_delta:
                    x, delta_h = x  # layer returns (x, delta_hidden)
                    delta_hiddens.append(delta_h)
            
            # Collect tap hiddens for CSCF
            if tap_layers is not None and i in tap_layers:
                tap_hiddens.append(x.clone() if return_delta else x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        out: Dict[str, Any] = {"logits": logits}
        if return_hidden:
            out["hidden_states"] = x
        if return_delta:
            # Aggregate delta contributions at the head space for Δ-loss if needed
            # Simple projection of last delta_hidden through lm_head's weight
            if len(delta_hiddens) == 0:
                out["delta_logits"] = torch.zeros_like(logits)
            else:
                # Sum delta hiddens from all layers kept
                dsum = torch.stack(delta_hiddens, dim=0).sum(0)
                out["delta_logits"] = self.lm_head(dsum)
        if tap_layers is not None:
            out["tap_hiddens"] = tap_hiddens
        return out

    @torch.no_grad()
    def forward_depth_delta_only(self, input_ids, attention_mask=None, base_budget: Optional[Budget] = None,
                                 full_budget: Optional[Budget] = None, use_dds: Optional[bool] = None):
        """Run only the delta **layers** d+1..L (and only their delta subpaths) on top of a base pass.
        Returns delta-only logits approximating the residual between full-depth and base-depth.
        """
        assert base_budget is not None and full_budget is not None
        self.budget_norms(base_budget); self.budget_norms(full_budget)
        
        if use_dds is None:
            use_dds = self.enable_dds
        
        # First run prefix depth to get hidden state at depth d
        B, T = input_ids.shape
        x = self.embeddings(input_ids)
        attn_bias = None
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :]
            attn_bias = (1.0 - mask) * -1e4

        d = max(self.depth_floor, base_budget.depth)
        for i, layer in enumerate(self.layers):
            if i >= d:
                break
            x = layer(x, base_budget, attention_mask=attn_bias, return_delta=False, delta_only=False,
                      use_dds=use_dds)

        # Now pass through layers d..L with delta_only=True under full_budget
        delta_hiddens = []
        for i, layer in enumerate(self.layers):
            if i < d:
                continue
            if i >= full_budget.depth:
                break
            x = layer(x, full_budget, attention_mask=attn_bias, return_delta=True, delta_only=True,
                      use_dds=use_dds)
            x, delta_h = x
            delta_hiddens.append(delta_h)

        if len(delta_hiddens) == 0:
            return torch.zeros(B, T, self.embeddings.word_embeddings.num_embeddings, device=x.device)
        dsum = torch.stack(delta_hiddens, dim=0).sum(0)
        logits_delta = self.lm_head(self.ln_f(dsum))
        return logits_delta
    
    def get_dds_sparsity_loss(self) -> torch.Tensor:
        """Get DDS sparsity regularization loss."""
        if self.dds_manager is None:
            return torch.tensor(0.0)
        return self.dds_manager.total_sparsity_loss()
    
    def get_dds_orthogonality_loss(self) -> torch.Tensor:
        """Get DDS orthogonality regularization loss."""
        if self.dds_manager is None:
            return torch.tensor(0.0)
        return self.dds_manager.orthogonality_loss()
    
    def export_delta_pack(self, layer_indices: Optional[List[int]] = None) -> dict:
        """Export DDS delta coefficients as upgrade pack."""
        if self.dds_manager is None:
            raise ValueError("DDS not enabled")
        return self.dds_manager.export_delta_pack(layer_indices)
    
    def load_delta_pack(self, pack: dict):
        """Load DDS delta coefficients from upgrade pack."""
        if self.dds_manager is None:
            raise ValueError("DDS not enabled")
        self.dds_manager.load_delta_pack(pack)

# convenience builder
def build_model(cfg: dict):
    m = DeltaEncoder(
        vocab_size=cfg.get('vocab_size', 30522),
        max_position_embeddings=cfg.get('max_position_embeddings', 512),
        hidden_size=cfg.get('hidden_size', 768),
        num_hidden_layers=cfg.get('num_hidden_layers', 12),
        num_attention_heads=cfg.get('num_attention_heads', 12),
        intermediate_size=cfg.get('intermediate_size', 3072),
        dropout=cfg.get('dropout', 0.1),
        attn_dropout=cfg.get('attention_dropout', 0.1),
        ln_eps=cfg.get('layer_norm_eps', 1e-5),
        base_ratio=cfg.get('base_ratio', 0.5),
        base_heads=cfg.get('base_heads', 6),
        drop_path=cfg.get('drop_path', 0.0),
        depth_floor=cfg.get('depth_floor', 0),
        budget_cond_ln=cfg.get('budget_cond_ln', True),
        ln_hyper_hidden=cfg.get('ln_hyper_hidden', 64),
        pad_token_id=cfg.get('pad_token_id', 0),
        enable_dds=cfg.get('enable_dds', False),
        dds_num_atoms=cfg.get('dds_num_atoms', 16),
        dds_rank=cfg.get('dds_rank', 64),
    )
    return m
