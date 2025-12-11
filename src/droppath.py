"""
Drop Path (Stochastic Depth) with Survival Schedule support.

Implements prefix depth dropout where deeper layers have higher drop probability,
with the schedule controlled by a phase-dependent alpha parameter.
"""

import torch
from torch import nn
from typing import Optional, Callable


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth). If drop_prob=0, identity.
    Scales surviving paths by 1/(1-drop_prob) to preserve expectation."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self._dynamic_drop_prob = None  # For dynamic scheduling

    def set_dynamic_drop_prob(self, prob: float):
        """Set dynamic drop probability (for phase-aware training)."""
        self._dynamic_drop_prob = prob
    
    def clear_dynamic_drop_prob(self):
        """Clear dynamic drop probability, revert to static."""
        self._dynamic_drop_prob = None

    def forward(self, x):
        drop_prob = self._dynamic_drop_prob if self._dynamic_drop_prob is not None else self.drop_prob
        
        if drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # broadcast over batch
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        output = x / keep_prob * random_tensor
        return output


def survival_prob(layer_idx: int, total_layers: int, alpha: float) -> float:
    """
    Compute survival probability for a layer using linear decay.
    
    Deeper layers have lower survival probability (higher drop rate).
    This packs information into early layers during the packing phase.
    
    Args:
        layer_idx: Index of the layer (0-indexed)
        total_layers: Total number of layers
        alpha: Drop rate scaling factor [0, 1]
               alpha=0 -> all layers survive
               alpha=0.5 -> last layer has 50% drop rate
               alpha=1 -> last layer always dropped
               
    Returns:
        Survival probability in [0, 1]
    """
    if total_layers <= 1:
        return 1.0
    
    # Linear decay: p_keep(l) = 1 - alpha * (l / (L-1))
    # Layer 0 always has p_keep = 1
    # Layer L-1 has p_keep = 1 - alpha
    drop_rate = alpha * (layer_idx / (total_layers - 1))
    return 1.0 - drop_rate


def survival_prob_cosine(layer_idx: int, total_layers: int, alpha: float) -> float:
    """
    Compute survival probability using cosine schedule.
    
    Provides smoother transition than linear, with more gradual
    increase in drop rate for early layers.
    
    Args:
        layer_idx: Index of the layer (0-indexed)
        total_layers: Total number of layers
        alpha: Maximum drop rate for last layer
        
    Returns:
        Survival probability in [0, 1]
    """
    import math
    
    if total_layers <= 1:
        return 1.0
    
    # Cosine schedule: drop rate increases gradually then faster
    progress = layer_idx / (total_layers - 1)
    drop_rate = alpha * (1 - math.cos(math.pi * progress)) / 2
    return 1.0 - drop_rate


class SurvivalSchedule:
    """
    Manages survival probabilities for prefix depth dropout.
    
    Provides phase-aware scheduling where alpha can be annealed
    across training phases.
    """
    
    def __init__(
        self,
        num_layers: int,
        initial_alpha: float = 0.5,
        schedule_type: str = 'linear'
    ):
        """
        Args:
            num_layers: Total number of layers in the model
            initial_alpha: Initial drop rate scaling factor
            schedule_type: 'linear' or 'cosine'
        """
        self.num_layers = num_layers
        self.alpha = initial_alpha
        self.schedule_type = schedule_type
        
        self._survival_fn = survival_prob if schedule_type == 'linear' else survival_prob_cosine
    
    def set_alpha(self, alpha: float):
        """Update alpha (e.g., for phase transitions)."""
        self.alpha = max(0.0, min(1.0, alpha))
    
    def get_survival_probs(self) -> list:
        """Get survival probabilities for all layers."""
        return [
            self._survival_fn(i, self.num_layers, self.alpha)
            for i in range(self.num_layers)
        ]
    
    def get_drop_probs(self) -> list:
        """Get drop probabilities for all layers."""
        return [1.0 - p for p in self.get_survival_probs()]
    
    def apply_to_model(self, model):
        """
        Apply current survival schedule to model's DropPath modules.
        
        Args:
            model: DeltaEncoder model with layers containing DropPath
        """
        drop_probs = self.get_drop_probs()
        
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'drop_path1'):
                layer.drop_path1.set_dynamic_drop_prob(drop_probs[i])
            if hasattr(layer, 'drop_path2'):
                layer.drop_path2.set_dynamic_drop_prob(drop_probs[i])
    
    def clear_from_model(self, model):
        """Clear dynamic drop probs from model, reverting to static."""
        for layer in model.layers:
            if hasattr(layer, 'drop_path1'):
                layer.drop_path1.clear_dynamic_drop_prob()
            if hasattr(layer, 'drop_path2'):
                layer.drop_path2.clear_dynamic_drop_prob()


def sample_depth_budget(
    num_layers: int,
    alpha: float,
    min_depth: int = 1,
    schedule_type: str = 'linear'
) -> int:
    """
    Sample a depth budget using survival probabilities.
    
    Samples which layer to stop at based on cumulative survival.
    
    Args:
        num_layers: Total number of layers
        alpha: Drop rate scaling factor
        min_depth: Minimum depth to use
        schedule_type: 'linear' or 'cosine'
        
    Returns:
        Sampled depth (number of layers to execute)
    """
    import random
    
    survival_fn = survival_prob if schedule_type == 'linear' else survival_prob_cosine
    
    # Sample each layer's survival independently
    depth = min_depth
    for i in range(min_depth, num_layers):
        p_survive = survival_fn(i, num_layers, alpha)
        if random.random() < p_survive:
            depth = i + 1
        else:
            break
    
    return depth


class PrefixDepthDropout:
    """
    Prefix Depth Dropout manager.
    
    Samples depth budgets and manages which layers to execute,
    ensuring only prefix layers are kept (no skipping).
    """
    
    def __init__(
        self,
        num_layers: int,
        depth_floor: int = 1,
        initial_alpha: float = 0.5,
        schedule_type: str = 'linear'
    ):
        """
        Args:
            num_layers: Total number of layers
            depth_floor: Minimum depth (never drop below this)
            initial_alpha: Initial drop rate scaling
            schedule_type: 'linear' or 'cosine'
        """
        self.num_layers = num_layers
        self.depth_floor = max(1, depth_floor)
        self.alpha = initial_alpha
        self.schedule_type = schedule_type
    
    def set_alpha(self, alpha: float):
        """Update alpha for phase transitions."""
        self.alpha = max(0.0, min(1.0, alpha))
    
    def sample_depth(self) -> int:
        """Sample a depth budget."""
        return sample_depth_budget(
            num_layers=self.num_layers,
            alpha=self.alpha,
            min_depth=self.depth_floor,
            schedule_type=self.schedule_type
        )
    
    def get_expected_depth(self) -> float:
        """Get expected depth under current alpha."""
        survival_fn = survival_prob if self.schedule_type == 'linear' else survival_prob_cosine
        
        # Expected depth = sum of survival probs
        expected = self.depth_floor
        for i in range(self.depth_floor, self.num_layers):
            expected += survival_fn(i, self.num_layers, self.alpha)
        
        return expected
    
    def get_depth_distribution(self) -> list:
        """Get probability distribution over depths."""
        survival_fn = survival_prob if self.schedule_type == 'linear' else survival_prob_cosine
        
        probs = [0.0] * (self.num_layers + 1)
        
        # P(depth = k) = P(survive layers 0..k-1) * P(drop layer k)
        cumulative_survive = 1.0
        for k in range(self.depth_floor, self.num_layers):
            p_drop_k = 1.0 - survival_fn(k, self.num_layers, self.alpha)
            probs[k] = cumulative_survive * p_drop_k
            cumulative_survive *= survival_fn(k, self.num_layers, self.alpha)
        
        # P(depth = num_layers) = survived all layers
        probs[self.num_layers] = cumulative_survive
        
        # Depth < depth_floor has prob 0
        for k in range(self.depth_floor):
            probs[k] = 0.0
        
        return probs


def anneal_alpha(
    current_step: int,
    total_steps: int,
    start_alpha: float,
    end_alpha: float,
    warmup_steps: int = 0,
    schedule: str = 'linear'
) -> float:
    """
    Anneal alpha from start to end over training.
    
    Args:
        current_step: Current training step
        total_steps: Total training steps
        start_alpha: Starting alpha value
        end_alpha: Ending alpha value
        warmup_steps: Steps to hold start_alpha before annealing
        schedule: 'linear' or 'cosine'
        
    Returns:
        Current alpha value
    """
    import math
    
    if current_step < warmup_steps:
        return start_alpha
    
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    
    if schedule == 'linear':
        return start_alpha + (end_alpha - start_alpha) * progress
    elif schedule == 'cosine':
        return end_alpha + (start_alpha - end_alpha) * (1 + math.cos(math.pi * progress)) / 2
    else:
        return start_alpha + (end_alpha - start_alpha) * progress
