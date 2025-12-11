"""
Loss functions for Delta-Matryoshka++ training.

Includes:
- MLM task loss
- Knowledge Distillation (KD) loss
- Delta residualization loss
- Monotonic Upgrade Guarantee (MUG) loss
- Cross-Scale Feature Alignment (CSCF) loss
- Budget controller penalty
"""

import torch
import torch.nn.functional as F
from typing import List, Optional


def masked_ce_loss(logits, labels):
    """Cross-entropy over positions where labels != -100."""
    mask = labels.ne(-100)
    if mask.sum() == 0:
        return logits.new_zeros([])
    logits = logits[mask]
    labels = labels[mask]
    return F.cross_entropy(logits, labels)


def kd_kl_loss(student_logits, teacher_logits, T=2.0):
    """KL divergence teacher -> student over masked positions (logits are [N, V])."""
    # Use log_softmax for student; softmax for teacher
    t = F.softmax(teacher_logits / T, dim=-1)
    s = F.log_softmax(student_logits / T, dim=-1)
    return F.kl_div(s, t, reduction='batchmean') * (T * T)


def mse_loss(a, b):
    """Mean squared error loss."""
    return F.mse_loss(a, b)


def monotonic_upgrade_loss(small_logits, large_logits, margin=0.0, labels=None):
    """Penalize cases where the larger budget is worse than the smaller.
    If labels are provided, use CE difference; otherwise use negative log likelihood w.r.t argmax of teacher.
    """
    if labels is None:
        # Use max probability class of large as pseudo-label
        labels = large_logits.argmax(dim=-1)
    mask = labels.ne(-100)
    if mask.sum() == 0:
        return small_logits.new_zeros([])
    small = small_logits[mask]
    large = large_logits[mask]
    ce_small = F.cross_entropy(small, labels[mask])
    ce_large = F.cross_entropy(large, labels[mask])
    # Penalize when large is worse than small by margin
    return F.relu((ce_large - ce_small) + margin)


# ============== New Loss Functions ==============

def cscf_loss(
    h_small: List[torch.Tensor],
    h_full: List[torch.Tensor],
    align_mean: bool = True,
    align_var: bool = True,
    detach_full: bool = True
) -> torch.Tensor:
    """
    Cross-Scale Feature Alignment (CSCF) Loss.
    
    Aligns mean and variance of hidden states between small and full budget
    at specified tap layers to reduce representation drift.
    
    Args:
        h_small: List of hidden states from small budget model at tap layers
        h_full: List of hidden states from full budget model at tap layers
        align_mean: Whether to align means
        align_var: Whether to align variances (diagonal covariance approx)
        detach_full: Whether to detach full model hiddens (treat as target)
        
    Returns:
        CSCF loss scalar
    """
    if len(h_small) == 0 or len(h_full) == 0:
        return torch.tensor(0.0, device=h_small[0].device if h_small else 'cpu')
    
    if len(h_small) != len(h_full):
        raise ValueError(f"Mismatched tap layers: {len(h_small)} vs {len(h_full)}")
    
    loss = torch.tensor(0.0, device=h_small[0].device)
    
    for h_s, h_f in zip(h_small, h_full):
        if detach_full:
            h_f = h_f.detach()
        
        # Flatten batch and sequence dims: (B, T, H) -> (B*T, H)
        h_s_flat = h_s.reshape(-1, h_s.shape[-1])
        h_f_flat = h_f.reshape(-1, h_f.shape[-1])
        
        if align_mean:
            # Mean alignment over all tokens
            mean_s = h_s_flat.mean(dim=0)
            mean_f = h_f_flat.mean(dim=0)
            loss = loss + F.mse_loss(mean_s, mean_f)
        
        if align_var:
            # Variance alignment (diagonal covariance for efficiency)
            var_s = h_s_flat.var(dim=0)
            var_f = h_f_flat.var(dim=0)
            loss = loss + F.mse_loss(var_s, var_f)
    
    return loss / len(h_small)


def budget_penalty(
    actual_delta_ratio: torch.Tensor,
    target_ratio: float,
    actual_depth: Optional[int] = None,
    target_depth: Optional[int] = None,
    depth_weight: float = 0.5
) -> torch.Tensor:
    """
    Budget Controller Penalty.
    
    Penalizes deviation from target delta invocation ratio and effective depth.
    Useful for controlling compute at inference time (latency SLO).
    
    Args:
        actual_delta_ratio: Actual fraction of tokens receiving delta compute
        target_ratio: Target delta ratio
        actual_depth: Actual depth used (optional)
        target_depth: Target depth (optional)
        depth_weight: Weight for depth penalty relative to ratio penalty
        
    Returns:
        Budget penalty scalar
    """
    # Ratio penalty: deviation from target
    ratio_penalty = (actual_delta_ratio - target_ratio).abs()
    
    # Depth penalty: penalize undershooting target depth
    depth_penalty = torch.tensor(0.0, device=ratio_penalty.device)
    if actual_depth is not None and target_depth is not None:
        # Penalize if actual_depth < target_depth
        depth_penalty = F.relu(torch.tensor(target_depth - actual_depth, dtype=torch.float32))
    
    return ratio_penalty + depth_weight * depth_penalty


def delta_energy_score(
    delta_logits: torch.Tensor,
    token_delta_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute delta energy score for OOD detection.
    
    Uses magnitude of delta corrections as uncertainty signal.
    Higher energy suggests model relies more on delta corrections,
    potentially indicating OOD inputs.
    
    Args:
        delta_logits: Delta-only logits from model
        token_delta_mask: Optional mask of which tokens received delta
        
    Returns:
        Per-sample energy scores (B,)
    """
    # L2 norm of delta logits per token
    energy_per_token = delta_logits.norm(dim=-1)  # (B, T)
    
    if token_delta_mask is not None:
        # Average only over tokens that received delta
        energy_per_token = energy_per_token * token_delta_mask.float()
        energy = energy_per_token.sum(dim=-1) / (token_delta_mask.float().sum(dim=-1) + 1e-8)
    else:
        # Average over all tokens
        energy = energy_per_token.mean(dim=-1)
    
    return energy  # (B,)


def packing_regularization(
    model,
    reg_type: str = 'spectral',
    target_layers: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Packing regularization to encourage information concentration in prefix.
    
    Args:
        model: DeltaEncoder model
        reg_type: Type of regularization ('spectral' or 'l2_base')
        target_layers: Which layers to regularize (None = all)
        
    Returns:
        Regularization loss
    """
    if target_layers is None:
        target_layers = list(range(len(model.layers)))
    
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    
    for i in target_layers:
        layer = model.layers[i]
        
        if reg_type == 'spectral':
            # Encourage base weights to have larger spectral norm than delta
            # This pushes information into base slice
            base_channels = int(layer.mlp.base_ratio * layer.mlp.intermediate_size)
            
            W_base = layer.mlp.fc1.weight[:base_channels]
            W_delta = layer.mlp.fc1.weight[base_channels:]
            
            if W_delta.shape[0] > 0:
                # Spectral norm approximation via power iteration
                spec_base = _spectral_norm_estimate(W_base)
                spec_delta = _spectral_norm_estimate(W_delta)
                # Penalize if delta has larger spectral norm
                loss = loss + F.relu(spec_delta - spec_base)
        
        elif reg_type == 'l2_base':
            # Encourage larger L2 norm in base weights
            base_channels = int(layer.mlp.base_ratio * layer.mlp.intermediate_size)
            
            norm_base = layer.mlp.fc1.weight[:base_channels].norm()
            norm_delta = layer.mlp.fc1.weight[base_channels:].norm()
            
            if norm_delta > 0:
                loss = loss + F.relu(norm_delta - norm_base)
    
    return loss / len(target_layers)


def _spectral_norm_estimate(W: torch.Tensor, n_iters: int = 1) -> torch.Tensor:
    """Estimate spectral norm via power iteration."""
    if W.numel() == 0:
        return torch.tensor(0.0, device=W.device)
    
    # Random vector
    v = torch.randn(W.shape[1], device=W.device)
    v = v / v.norm()
    
    for _ in range(n_iters):
        u = W @ v
        u = u / (u.norm() + 1e-8)
        v = W.T @ u
        v = v / (v.norm() + 1e-8)
    
    return (W @ v).norm()


def multi_budget_mug_loss(
    logits_dict: dict,
    labels: torch.Tensor,
    margin: float = 0.0
) -> torch.Tensor:
    """
    Multi-budget Monotonic Upgrade Guarantee loss.
    
    Checks monotonicity across all budget pairs where one dominates the other.
    
    Args:
        logits_dict: Dict mapping budget tuple (w, h, d) to logits
        labels: Ground truth labels
        margin: Margin for monotonicity violation
        
    Returns:
        MUG loss summed over all valid pairs
    """
    budgets = list(logits_dict.keys())
    loss = torch.tensor(0.0, device=labels.device)
    count = 0
    
    for i, b1 in enumerate(budgets):
        for j, b2 in enumerate(budgets):
            if i >= j:
                continue
            
            w1, h1, d1 = b1
            w2, h2, d2 = b2
            
            # Check if b1 dominates b2 (smaller in all dims)
            if w1 <= w2 and h1 <= h2 and d1 <= d2:
                # b2 should be at least as good as b1
                loss = loss + monotonic_upgrade_loss(
                    logits_dict[b1], logits_dict[b2], 
                    margin=margin, labels=labels
                )
                count += 1
            elif w2 <= w1 and h2 <= h1 and d2 <= d1:
                # b1 should be at least as good as b2
                loss = loss + monotonic_upgrade_loss(
                    logits_dict[b2], logits_dict[b1],
                    margin=margin, labels=labels
                )
                count += 1
    
    if count > 0:
        loss = loss / count
    
    return loss


def compute_ece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15
) -> torch.Tensor:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        logits: Model logits (N, C) or (B, T, C)
        labels: Ground truth labels (N,) or (B, T)
        n_bins: Number of bins for calibration
        
    Returns:
        ECE scalar
    """
    # Flatten if needed
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1)
    
    # Filter out padding
    mask = labels.ne(-100)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    
    logits = logits[mask]
    labels = labels[mask]
    
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels).float()
    
    # Bin boundaries
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    
    ece = torch.tensor(0.0, device=logits.device)
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()
        
        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece = ece + prop_in_bin * (avg_confidence - avg_accuracy).abs()
    
    return ece
