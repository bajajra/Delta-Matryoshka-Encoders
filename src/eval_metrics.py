"""
Evaluation metrics for Delta-Matryoshka++ models.

Includes:
- MLM Perplexity per budget
- AUTC (Area Under Trade-off Curve): accuracy vs FLOPs
- Monotonicity Violations
- Delta Pack Size
- Calibration (ECE/NLL)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from .model import DeltaEncoder, Budget


@dataclass
class BudgetMetrics:
    """Metrics for a single budget configuration."""
    width: float
    heads: int
    depth: int
    
    # Performance
    mlm_loss: float = 0.0
    mlm_accuracy: float = 0.0
    perplexity: float = 0.0
    
    # Compute
    flops: int = 0
    params: int = 0
    throughput: float = 0.0  # tokens/sec
    
    # Calibration
    ece: float = 0.0
    nll: float = 0.0
    
    # Delta-specific
    delta_energy: float = 0.0
    delta_ratio: float = 0.0


def estimate_flops(
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    seq_len: int,
    depth: int,
    width_ratio: float = 1.0,
    head_count: Optional[int] = None
) -> int:
    """
    Estimate FLOPs for a forward pass through the encoder.
    
    This is a rough estimate based on dominant operations.
    """
    if head_count is None:
        head_count = num_heads
    
    eff_intermediate = int(width_ratio * intermediate_size)
    head_dim = hidden_size // num_heads
    eff_hidden = head_count * head_dim
    
    # Per layer FLOPs (rough estimate)
    # Attention: Q,K,V projections + attention scores + output projection
    attn_flops = (
        3 * seq_len * hidden_size * eff_hidden +  # Q, K, V projections
        seq_len * seq_len * head_count * head_dim +  # Attention scores
        seq_len * seq_len * head_count * head_dim +  # Attention @ V
        seq_len * eff_hidden * hidden_size  # Output projection
    )
    
    # MLP: fc1 + fc2
    mlp_flops = (
        seq_len * hidden_size * eff_intermediate +  # fc1
        seq_len * eff_intermediate * hidden_size     # fc2
    )
    
    layer_flops = attn_flops + mlp_flops
    
    # Total for all layers
    total_flops = depth * layer_flops
    
    # Embeddings (rough)
    total_flops += seq_len * hidden_size
    
    return int(total_flops)


def estimate_params(
    vocab_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    depth: int,
    width_ratio: float = 1.0,
    head_count: Optional[int] = None
) -> int:
    """Estimate parameter count for a sub-model."""
    if head_count is None:
        head_count = num_heads
    
    eff_intermediate = int(width_ratio * intermediate_size)
    head_dim = hidden_size // num_heads
    eff_hidden = head_count * head_dim
    
    # Embeddings
    embed_params = vocab_size * hidden_size + 512 * hidden_size  # word + position
    
    # Per layer
    attn_params = (
        4 * hidden_size * eff_hidden +  # Q, K, V, O
        4 * hidden_size  # biases
    )
    mlp_params = (
        hidden_size * eff_intermediate +
        eff_intermediate * hidden_size +
        eff_intermediate + hidden_size  # biases
    )
    ln_params = 4 * hidden_size  # 2 LayerNorms
    layer_params = attn_params + mlp_params + ln_params
    
    total_params = embed_params + depth * layer_params
    
    return int(total_params)


class MetricsTracker:
    """
    Tracks and computes evaluation metrics across budget configurations.
    """
    
    def __init__(
        self,
        model: DeltaEncoder,
        budgets: List[Tuple[float, int, int]],
        device: torch.device
    ):
        """
        Args:
            model: DeltaEncoder model
            budgets: List of (width, heads, depth) tuples to evaluate
            device: Device for computation
        """
        self.model = model
        self.budgets = budgets
        self.device = device
        
        # Metrics storage
        self.metrics: Dict[Tuple, BudgetMetrics] = {}
        self.running_stats = defaultdict(lambda: defaultdict(list))
        
        # Initialize metrics for each budget
        for b in budgets:
            self.metrics[b] = BudgetMetrics(width=b[0], heads=b[1], depth=b[2])
    
    @torch.no_grad()
    def evaluate_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Evaluate a single batch across all budgets.
        
        Args:
            input_ids: (B, T) input tokens
            attention_mask: (B, T) attention mask
            labels: (B, T) MLM labels (-100 for non-masked)
        """
        self.model.eval()
        
        for budget_tuple in self.budgets:
            w, h, d = budget_tuple
            budget = Budget(width=w, heads=h, depth=d)
            
            # Forward pass
            out = self.model(
                input_ids, attention_mask, budget=budget,
                return_delta=True, return_hidden=True
            )
            logits = out["logits"]
            delta_logits = out.get("delta_logits", torch.zeros_like(logits))
            
            # Compute metrics for masked positions
            mask = labels.ne(-100)
            if mask.sum() == 0:
                continue
            
            logits_masked = logits[mask]
            labels_masked = labels[mask]
            delta_masked = delta_logits[mask]
            
            # MLM loss and accuracy
            loss = F.cross_entropy(logits_masked, labels_masked)
            preds = logits_masked.argmax(dim=-1)
            acc = (preds == labels_masked).float().mean()
            
            # Calibration
            probs = F.softmax(logits_masked, dim=-1)
            confidences = probs.max(dim=-1).values
            
            # Delta energy
            delta_energy = delta_masked.norm(dim=-1).mean()
            
            # Store running stats
            self.running_stats[budget_tuple]["loss"].append(loss.item())
            self.running_stats[budget_tuple]["acc"].append(acc.item())
            self.running_stats[budget_tuple]["confidence"].append(confidences.mean().item())
            self.running_stats[budget_tuple]["delta_energy"].append(delta_energy.item())
    
    def compute_final_metrics(self) -> Dict[Tuple, BudgetMetrics]:
        """Compute final metrics from running stats."""
        seq_len = 128  # Assumed sequence length for FLOPs
        
        for budget_tuple in self.budgets:
            w, h, d = budget_tuple
            stats = self.running_stats[budget_tuple]
            
            if not stats["loss"]:
                continue
            
            # Average metrics
            avg_loss = np.mean(stats["loss"])
            avg_acc = np.mean(stats["acc"])
            ppl = np.exp(avg_loss)
            
            # Estimate compute
            flops = estimate_flops(
                hidden_size=self.model.hidden_size,
                intermediate_size=self.model.layers[0].mlp.intermediate_size,
                num_heads=self.model.num_attention_heads,
                seq_len=seq_len,
                depth=d,
                width_ratio=w,
                head_count=h
            )
            
            params = estimate_params(
                vocab_size=self.model.embeddings.word_embeddings.num_embeddings,
                hidden_size=self.model.hidden_size,
                intermediate_size=self.model.layers[0].mlp.intermediate_size,
                num_heads=self.model.num_attention_heads,
                depth=d,
                width_ratio=w,
                head_count=h
            )
            
            self.metrics[budget_tuple] = BudgetMetrics(
                width=w,
                heads=h,
                depth=d,
                mlm_loss=avg_loss,
                mlm_accuracy=avg_acc,
                perplexity=ppl,
                flops=flops,
                params=params,
                delta_energy=np.mean(stats["delta_energy"]) if stats["delta_energy"] else 0.0,
            )
        
        return self.metrics
    
    def compute_autc(self) -> float:
        """
        Compute Area Under Trade-off Curve (AUTC).
        
        Measures the accuracy-FLOPs trade-off across all budgets.
        Higher AUTC = better trade-off (more accuracy per FLOP).
        """
        # Sort by FLOPs
        sorted_budgets = sorted(
            self.metrics.items(),
            key=lambda x: x[1].flops
        )
        
        if len(sorted_budgets) < 2:
            return 0.0
        
        # Normalize FLOPs and accuracy to [0, 1]
        flops_values = [m.flops for _, m in sorted_budgets]
        acc_values = [m.mlm_accuracy for _, m in sorted_budgets]
        
        flops_min, flops_max = min(flops_values), max(flops_values)
        acc_min, acc_max = min(acc_values), max(acc_values)
        
        if flops_max == flops_min or acc_max == acc_min:
            return 0.0
        
        flops_norm = [(f - flops_min) / (flops_max - flops_min) for f in flops_values]
        acc_norm = [(a - acc_min) / (acc_max - acc_min) for a in acc_values]
        
        # Compute area using trapezoidal rule
        autc = 0.0
        for i in range(len(sorted_budgets) - 1):
            dx = flops_norm[i + 1] - flops_norm[i]
            avg_y = (acc_norm[i] + acc_norm[i + 1]) / 2
            autc += dx * avg_y
        
        return autc
    
    def compute_monotonicity_violations(self) -> Dict[str, Any]:
        """
        Count violations of monotonic upgrade guarantee.
        
        A violation occurs when a larger budget performs worse than a smaller one.
        """
        violations = []
        total_pairs = 0
        
        for b1, m1 in self.metrics.items():
            for b2, m2 in self.metrics.items():
                if b1 == b2:
                    continue
                
                # Check if b1 dominates b2 (b1 <= b2 in all dims)
                if b1[0] <= b2[0] and b1[1] <= b2[1] and b1[2] <= b2[2]:
                    total_pairs += 1
                    
                    # b2 should have at least as good accuracy as b1
                    if m2.mlm_accuracy < m1.mlm_accuracy - 0.01:  # 1% tolerance
                        violations.append({
                            "small": b1,
                            "large": b2,
                            "small_acc": m1.mlm_accuracy,
                            "large_acc": m2.mlm_accuracy,
                            "gap": m1.mlm_accuracy - m2.mlm_accuracy
                        })
        
        return {
            "num_violations": len(violations),
            "total_pairs": total_pairs,
            "violation_rate": len(violations) / max(1, total_pairs),
            "violations": violations
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary of metrics."""
        lines = ["=" * 70]
        lines.append("Delta-Matryoshka++ Evaluation Summary")
        lines.append("=" * 70)
        
        # Per-budget metrics table
        lines.append("\nPer-Budget Metrics:")
        lines.append("-" * 70)
        lines.append(f"{'Budget':<15} {'PPL':>8} {'Acc':>8} {'FLOPs':>12} {'Params':>10}")
        lines.append("-" * 70)
        
        sorted_metrics = sorted(
            self.metrics.items(),
            key=lambda x: (x[0][2], x[0][1], x[0][0])  # Sort by depth, heads, width
        )
        
        for budget_tuple, m in sorted_metrics:
            budget_str = f"({m.width:.1f},{m.heads},{m.depth})"
            lines.append(
                f"{budget_str:<15} {m.perplexity:>8.2f} {m.mlm_accuracy:>7.1%} "
                f"{m.flops/1e9:>10.2f}G {m.params/1e6:>8.1f}M"
            )
        
        # Aggregate metrics
        lines.append("-" * 70)
        lines.append(f"\nAggregate Metrics:")
        
        autc = self.compute_autc()
        lines.append(f"  AUTC (Area Under Trade-off Curve): {autc:.4f}")
        
        mono_results = self.compute_monotonicity_violations()
        lines.append(f"  Monotonicity Violations: {mono_results['num_violations']}/{mono_results['total_pairs']} "
                    f"({mono_results['violation_rate']:.1%})")
        
        # Best/worst budgets
        if self.metrics:
            best_acc = max(self.metrics.items(), key=lambda x: x[1].mlm_accuracy)
            best_eff = max(self.metrics.items(), key=lambda x: x[1].mlm_accuracy / (x[1].flops / 1e9 + 1))
            
            lines.append(f"  Best Accuracy: {best_acc[0]} with {best_acc[1].mlm_accuracy:.1%}")
            lines.append(f"  Best Efficiency: {best_eff[0]} with {best_eff[1].mlm_accuracy:.1%} @ {best_eff[1].flops/1e9:.1f}G FLOPs")
        
        lines.append("=" * 70)
        return "\n".join(lines)


def compute_delta_pack_size(model: DeltaEncoder) -> Dict[str, int]:
    """
    Compute sizes of DDS delta packs in bytes.
    
    Args:
        model: DeltaEncoder with DDS enabled
        
    Returns:
        Dict with size breakdowns
    """
    if not model.enable_dds or model.dds_manager is None:
        return {"total_bytes": 0, "atom_bank_bytes": 0, "coefficients_bytes": 0}
    
    dds = model.dds_manager
    
    # Atom bank size (shared across all budgets)
    atom_bank_bytes = dds.atom_bank_size_bytes()
    
    # Per-layer coefficient size (the "upgrade pack")
    coeff_bytes = dds.delta_pack_size_bytes()
    
    return {
        "atom_bank_bytes": atom_bank_bytes,
        "coefficients_bytes": coeff_bytes,
        "total_bytes": atom_bank_bytes + coeff_bytes,
        "atom_bank_mb": atom_bank_bytes / (1024 * 1024),
        "coefficients_kb": coeff_bytes / 1024,
    }


def evaluate_ood_detection(
    model: DeltaEncoder,
    id_dataloader,
    ood_dataloader,
    device: torch.device,
    budget: Budget
) -> Dict[str, float]:
    """
    Evaluate OOD detection using delta energy.
    
    Args:
        model: DeltaEncoder model
        id_dataloader: In-distribution data loader
        ood_dataloader: Out-of-distribution data loader  
        device: Device for computation
        budget: Budget to evaluate
        
    Returns:
        Dict with AUROC and other OOD metrics
    """
    from sklearn.metrics import roc_auc_score
    
    model.eval()
    
    id_energies = []
    ood_energies = []
    
    with torch.no_grad():
        # ID samples
        for batch in id_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            out = model(input_ids, attention_mask, budget=budget, return_delta=True)
            delta_logits = out.get("delta_logits", torch.zeros_like(out["logits"]))
            
            # Energy per sample
            energy = delta_logits.norm(dim=-1).mean(dim=-1)  # (B,)
            id_energies.extend(energy.cpu().tolist())
        
        # OOD samples
        for batch in ood_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            out = model(input_ids, attention_mask, budget=budget, return_delta=True)
            delta_logits = out.get("delta_logits", torch.zeros_like(out["logits"]))
            
            energy = delta_logits.norm(dim=-1).mean(dim=-1)
            ood_energies.extend(energy.cpu().tolist())
    
    # Compute AUROC
    labels = [0] * len(id_energies) + [1] * len(ood_energies)
    scores = id_energies + ood_energies
    
    auroc = roc_auc_score(labels, scores)
    
    return {
        "auroc": auroc,
        "id_energy_mean": np.mean(id_energies),
        "id_energy_std": np.std(id_energies),
        "ood_energy_mean": np.mean(ood_energies),
        "ood_energy_std": np.std(ood_energies),
    }


def run_full_evaluation(
    model: DeltaEncoder,
    eval_dataloader,
    budgets: List[Tuple[float, int, int]],
    device: torch.device,
    max_batches: int = 100
) -> Dict[str, Any]:
    """
    Run full evaluation suite.
    
    Args:
        model: DeltaEncoder model
        eval_dataloader: Evaluation data loader
        budgets: List of budget tuples to evaluate
        device: Device for computation
        max_batches: Maximum batches to evaluate
        
    Returns:
        Dict with all metrics
    """
    tracker = MetricsTracker(model, budgets, device)
    
    for i, batch in enumerate(eval_dataloader):
        if i >= max_batches:
            break
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        tracker.evaluate_batch(input_ids, attention_mask, labels)
    
    # Compute final metrics
    metrics = tracker.compute_final_metrics()
    autc = tracker.compute_autc()
    mono_violations = tracker.compute_monotonicity_violations()
    
    # DDS pack sizes
    dds_sizes = compute_delta_pack_size(model)
    
    return {
        "per_budget_metrics": {str(k): vars(v) for k, v in metrics.items()},
        "autc": autc,
        "monotonicity": mono_violations,
        "dds_sizes": dds_sizes,
        "summary": tracker.get_summary()
    }

