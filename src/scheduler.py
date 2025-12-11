"""
Three-Phase Training Scheduler for Delta-Matryoshka++

Phase 1 (Packing): Strong prefix depth dropout, emphasize small budgets
Phase 2 (Residualization): Enable delta learning and routing, anneal dropout  
Phase 3 (Calibration): MUG loss, CSCF, low learning rate
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class PhaseConfig:
    """Configuration for a single training phase."""
    # Depth dropout
    alpha_depth_drop: float = 0.5
    
    # Loss weights
    beta_delta: float = 0.3          # Delta residualization weight
    rho_mug: float = 0.0             # Monotonic upgrade guarantee weight
    zeta_cscf: float = 0.0           # Cross-scale feature alignment weight
    eta_dds_sparsity: float = 0.0    # DDS sparsity regularization weight
    gamma_packing: float = 0.0       # Packing regularization weight
    
    # Learning rate
    lr_scale: float = 1.0
    
    # Budget sampling
    oversample_min: bool = True       # Oversample smallest budget
    enable_tce: bool = False          # Enable token-conditional delta
    target_delta_ratio: float = 0.35  # Target delta compute ratio
    
    # Routing
    enable_layer_routing: bool = False
    
    # Misc
    freeze_base: bool = False         # Freeze base weights (only train delta)


@dataclass 
class PhaseSchedulerConfig:
    """Configuration for the three-phase scheduler."""
    total_steps: int = 100000
    phase_ratios: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])
    
    # Phase 1: Packing
    packing: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        alpha_depth_drop=0.5,
        beta_delta=0.3,
        rho_mug=0.0,
        zeta_cscf=0.0,
        eta_dds_sparsity=1e-4,
        gamma_packing=1e-3,
        lr_scale=1.0,
        oversample_min=True,
        enable_tce=False,
        target_delta_ratio=0.0,
    ))
    
    # Phase 2: Residualization + Routing  
    residualization: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        alpha_depth_drop=0.3,
        beta_delta=0.7,
        rho_mug=0.05,
        zeta_cscf=0.1,
        eta_dds_sparsity=1e-4,
        gamma_packing=0.0,
        lr_scale=1.0,
        oversample_min=False,
        enable_tce=True,
        target_delta_ratio=0.35,
    ))
    
    # Phase 3: Calibration + Deploy
    calibration: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        alpha_depth_drop=0.2,
        beta_delta=0.5,
        rho_mug=0.1,
        zeta_cscf=0.2,
        eta_dds_sparsity=5e-5,
        gamma_packing=0.0,
        lr_scale=0.3,
        oversample_min=False,
        enable_tce=True,
        target_delta_ratio=0.35,
    ))


class PhaseScheduler:
    """
    Three-phase training scheduler.
    
    Manages hyperparameter transitions across training phases:
    1. Packing: Pack information into prefix (base) model
    2. Residualization: Learn delta corrections with routing
    3. Calibration: Fine-tune for monotonicity and calibration
    """
    
    def __init__(self, config: Optional[PhaseSchedulerConfig] = None):
        """
        Args:
            config: Scheduler configuration (uses defaults if None)
        """
        self.config = config or PhaseSchedulerConfig()
        self._compute_boundaries()
    
    def _compute_boundaries(self):
        """Compute step boundaries between phases."""
        ratios = self.config.phase_ratios
        total = self.config.total_steps
        
        self.phase_boundaries = [0]
        cumsum = 0
        for r in ratios[:-1]:
            cumsum += int(r * total)
            self.phase_boundaries.append(cumsum)
        self.phase_boundaries.append(total)
    
    def get_phase(self, step: int) -> int:
        """
        Get current phase index (0, 1, or 2).
        
        Args:
            step: Current training step
            
        Returns:
            Phase index
        """
        for i, boundary in enumerate(self.phase_boundaries[1:]):
            if step < boundary:
                return i
        return len(self.phase_boundaries) - 2
    
    def get_phase_name(self, step: int) -> str:
        """Get human-readable phase name."""
        phase = self.get_phase(step)
        return ['packing', 'residualization', 'calibration'][phase]
    
    def get_phase_config(self, step: int) -> PhaseConfig:
        """
        Get configuration for current phase.
        
        Args:
            step: Current training step
            
        Returns:
            PhaseConfig for current phase
        """
        phase = self.get_phase(step)
        configs = [
            self.config.packing,
            self.config.residualization,
            self.config.calibration
        ]
        return configs[phase]
    
    def get_config(self, step: int) -> Dict:
        """
        Get configuration dict for current step.
        
        Includes smooth interpolation within phases for some parameters.
        
        Args:
            step: Current training step
            
        Returns:
            Dict with all configuration values
        """
        phase = self.get_phase(step)
        phase_config = self.get_phase_config(step)
        
        # Get progress within current phase for interpolation
        phase_start = self.phase_boundaries[phase]
        phase_end = self.phase_boundaries[phase + 1]
        phase_progress = (step - phase_start) / max(1, phase_end - phase_start)
        
        # Interpolate alpha_depth_drop within phase (gradual annealing)
        if phase < 2:
            next_phase_config = [
                self.config.residualization,
                self.config.calibration
            ][phase]
            alpha = self._interpolate(
                phase_config.alpha_depth_drop,
                next_phase_config.alpha_depth_drop,
                phase_progress,
                schedule='cosine'
            )
        else:
            alpha = phase_config.alpha_depth_drop
        
        return {
            'phase': phase,
            'phase_name': self.get_phase_name(step),
            'phase_progress': phase_progress,
            'alpha_depth_drop': alpha,
            'beta_delta': phase_config.beta_delta,
            'rho_mug': phase_config.rho_mug,
            'zeta_cscf': phase_config.zeta_cscf,
            'eta_dds_sparsity': phase_config.eta_dds_sparsity,
            'gamma_packing': phase_config.gamma_packing,
            'lr_scale': phase_config.lr_scale,
            'oversample_min': phase_config.oversample_min,
            'enable_tce': phase_config.enable_tce,
            'target_delta_ratio': phase_config.target_delta_ratio,
            'enable_layer_routing': phase_config.enable_layer_routing,
            'freeze_base': phase_config.freeze_base,
        }
    
    def _interpolate(
        self, 
        start: float, 
        end: float, 
        progress: float,
        schedule: str = 'linear'
    ) -> float:
        """Interpolate between values."""
        progress = max(0.0, min(1.0, progress))
        
        if schedule == 'linear':
            return start + (end - start) * progress
        elif schedule == 'cosine':
            return end + (start - end) * (1 + math.cos(math.pi * progress)) / 2
        else:
            return start + (end - start) * progress
    
    def get_lr_multiplier(self, step: int) -> float:
        """Get learning rate multiplier for current step."""
        return self.get_phase_config(step).lr_scale
    
    def should_log_phase_transition(self, step: int) -> bool:
        """Check if we just entered a new phase."""
        return step in self.phase_boundaries
    
    def get_phase_summary(self, step: int) -> str:
        """Get summary string for logging."""
        cfg = self.get_config(step)
        return (
            f"Phase {cfg['phase']+1}/3 ({cfg['phase_name']}): "
            f"alpha={cfg['alpha_depth_drop']:.2f}, "
            f"beta_delta={cfg['beta_delta']:.2f}, "
            f"rho_mug={cfg['rho_mug']:.3f}, "
            f"zeta_cscf={cfg['zeta_cscf']:.2f}, "
            f"lr_scale={cfg['lr_scale']:.2f}"
        )


class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine decay.
    
    Integrates with PhaseScheduler for phase-aware LR scaling.
    """
    
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        phase_scheduler: Optional[PhaseScheduler] = None
    ):
        """
        Args:
            base_lr: Base learning rate
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr_ratio: Minimum LR as ratio of base_lr
            phase_scheduler: Optional phase scheduler for LR scaling
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = base_lr * min_lr_ratio
        self.phase_scheduler = phase_scheduler
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for current step."""
        # Warmup
        if step < self.warmup_steps:
            lr = self.base_lr * (step + 1) / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
        
        # Apply phase scaling
        if self.phase_scheduler is not None:
            lr *= self.phase_scheduler.get_lr_multiplier(step)
        
        return lr
    
    def step(self, optimizer, step: int):
        """Update optimizer learning rate."""
        lr = self.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class BudgetSampler:
    """
    Budget sampler with phase-aware sampling strategies.
    """
    
    def __init__(
        self,
        width_budgets: List[float],
        head_budgets: List[int],
        depth_budgets: List[int],
        num_samples: int = 4
    ):
        """
        Args:
            width_budgets: List of width ratios
            head_budgets: List of head counts
            depth_budgets: List of depth values
            num_samples: Number of budgets to sample per step
        """
        self.width_budgets = sorted(width_budgets)
        self.head_budgets = sorted(head_budgets)
        self.depth_budgets = sorted(depth_budgets)
        self.num_samples = num_samples
        
        # Precompute min/max budgets
        self.min_budget = (self.width_budgets[0], self.head_budgets[0], self.depth_budgets[0])
        self.max_budget = (self.width_budgets[-1], self.head_budgets[-1], self.depth_budgets[-1])
        
        # All possible budgets
        self.all_budgets = [
            (w, h, d) 
            for w in self.width_budgets 
            for h in self.head_budgets 
            for d in self.depth_budgets
        ]
    
    def sample(
        self, 
        oversample_min: bool = True,
        include_endpoints: bool = True,
        temperature: float = 1.0
    ) -> List[Tuple[float, int, int]]:
        """
        Sample budgets for one training step.
        
        Args:
            oversample_min: Include min budget with higher probability
            include_endpoints: Always include min and max budgets
            temperature: Sampling temperature (lower = more uniform)
            
        Returns:
            List of (width, heads, depth) tuples
        """
        import random
        
        samples = []
        
        # Always include endpoints
        if include_endpoints:
            samples.append(self.min_budget)
            samples.append(self.max_budget)
        
        # Sample remaining budgets
        remaining = self.num_samples - len(samples)
        pool = [b for b in self.all_budgets if b not in samples]
        
        if remaining > 0 and pool:
            if oversample_min:
                # Bias toward smaller budgets
                weights = []
                for w, h, d in pool:
                    # Lower weight for larger budgets
                    score = (1 - w) + (1 - h / self.head_budgets[-1]) + (1 - d / self.depth_budgets[-1])
                    weights.append(math.exp(score / temperature))
                
                total = sum(weights)
                weights = [w / total for w in weights]
                
                indices = random.choices(range(len(pool)), weights=weights, k=remaining)
                samples.extend([pool[i] for i in indices])
            else:
                # Uniform sampling
                random.shuffle(pool)
                samples.extend(pool[:remaining])
        
        return samples
    
    def sample_depth_aware(
        self,
        depth_alpha: float,
        num_layers: int,
        oversample_min: bool = True
    ) -> List[Tuple[float, int, int]]:
        """
        Sample budgets with depth determined by survival schedule.
        
        Args:
            depth_alpha: Alpha for depth survival schedule
            num_layers: Total number of layers
            oversample_min: Oversample smallest budget
            
        Returns:
            List of (width, heads, depth) tuples
        """
        from .droppath import sample_depth_budget
        import random
        
        samples = []
        
        # Min budget always included
        samples.append(self.min_budget)
        
        # Max budget always included  
        samples.append(self.max_budget)
        
        # Sample remaining with dynamic depth
        for _ in range(self.num_samples - 2):
            w = random.choice(self.width_budgets)
            h = random.choice(self.head_budgets)
            
            # Sample depth using survival schedule
            d = sample_depth_budget(
                num_layers=num_layers,
                alpha=depth_alpha,
                min_depth=self.depth_budgets[0]
            )
            # Clamp to valid depth budgets
            d = min(d, self.depth_budgets[-1])
            
            samples.append((w, h, d))
        
        # Oversample min if requested
        if oversample_min and random.random() < 0.3:
            samples[random.randint(2, len(samples)-1)] = self.min_budget
        
        return samples

