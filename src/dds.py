"""
Delta Dictionary Sharing (DDS) Module

Implements shared low-rank atoms for delta weights, enabling tiny "upgrade packs"
where each layer only stores mixing coefficients rather than full delta weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DeltaDictionary(nn.Module):
    """
    Shared dictionary of low-rank atoms for reconstructing delta weights.
    
    Each atom is a low-rank matrix factorization: atom = U @ V
    where U: (rank, in_features) and V: (rank, out_features)
    
    Delta weight = sum_i(coeff_i * atom_i) for sparse reconstruction.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_atoms: int = 16,
        rank: int = 64,
        init_std: float = 0.02
    ):
        """
        Args:
            in_features: Input dimension for delta weights
            out_features: Output dimension for delta weights
            num_atoms: Number of shared atoms in the dictionary
            rank: Rank of each atom (controls expressiveness vs compression)
            init_std: Standard deviation for initialization
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_atoms = num_atoms
        self.rank = rank
        
        # Atom bank: factorized as U @ V for memory efficiency
        # U: (num_atoms, rank, in_features)
        # V: (num_atoms, out_features, rank)
        # Full atom would be: V @ U -> (out_features, in_features)
        self.atom_U = nn.Parameter(
            torch.randn(num_atoms, rank, in_features) * init_std
        )
        self.atom_V = nn.Parameter(
            torch.randn(num_atoms, out_features, rank) * init_std
        )
    
    def get_atom(self, idx: int) -> torch.Tensor:
        """Get a single atom as full matrix (out_features, in_features)."""
        # V[idx] @ U[idx] -> (out_features, rank) @ (rank, in_features)
        return self.atom_V[idx] @ self.atom_U[idx]
    
    def reconstruct(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct delta weight from coefficients.
        
        Args:
            coeffs: (num_atoms,) mixing coefficients
            
        Returns:
            Reconstructed weight matrix (out_features, in_features)
        """
        # Efficient: compute weighted sum of low-rank products
        # coeffs: (num_atoms,)
        # atom_V: (num_atoms, out_features, rank)
        # atom_U: (num_atoms, rank, in_features)
        
        # Weighted V: (num_atoms, out_features, rank) * (num_atoms, 1, 1)
        weighted_V = self.atom_V * coeffs.view(-1, 1, 1)
        # Sum over atoms: (out_features, rank)
        V_sum = weighted_V.sum(dim=0)
        # U mean (or could also weight): for simplicity, sum weighted
        weighted_U = self.atom_U * coeffs.view(-1, 1, 1)
        U_sum = weighted_U.sum(dim=0)
        
        # Final reconstruction: (out_features, rank) @ (rank, in_features)
        return V_sum @ U_sum
    
    def reconstruct_efficient(self, coeffs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return factorized form for efficient matmul: (V_eff, U_eff)
        where delta_weight ≈ V_eff @ U_eff
        
        This allows applying delta as: x @ U_eff.T @ V_eff.T
        which is more efficient for inference.
        """
        weighted_V = self.atom_V * coeffs.view(-1, 1, 1)
        V_eff = weighted_V.sum(dim=0)  # (out_features, rank)
        
        weighted_U = self.atom_U * coeffs.view(-1, 1, 1)
        U_eff = weighted_U.sum(dim=0)  # (rank, in_features)
        
        return V_eff, U_eff
    
    def sparsity_loss(self, coeffs: torch.Tensor) -> torch.Tensor:
        """L1 sparsity penalty on coefficients."""
        return coeffs.abs().sum()
    
    def orthogonality_loss(self) -> torch.Tensor:
        """Encourage atoms to be diverse/orthogonal."""
        # Compute pairwise similarity between atoms
        # Flatten atoms and compute cosine similarity
        U_flat = self.atom_U.view(self.num_atoms, -1)  # (num_atoms, rank * in_features)
        U_norm = F.normalize(U_flat, dim=1)
        sim = U_norm @ U_norm.T  # (num_atoms, num_atoms)
        
        # Penalize off-diagonal similarities
        eye = torch.eye(self.num_atoms, device=sim.device)
        off_diag = sim * (1 - eye)
        return off_diag.abs().mean()


class DeltaDictionaryMLP(nn.Module):
    """
    MLP delta weights reconstructed from shared dictionary.
    
    Instead of storing full delta weight matrices, stores only
    per-layer coefficients that mix shared atoms.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        base_intermediate: int,
        dds_fc1: DeltaDictionary,
        dds_fc2: DeltaDictionary,
        num_atoms: int = 16
    ):
        """
        Args:
            hidden_size: Model hidden dimension
            intermediate_size: Full MLP intermediate size
            base_intermediate: Size of base (non-delta) intermediate
            dds_fc1: Shared dictionary for fc1 delta
            dds_fc2: Shared dictionary for fc2 delta
            num_atoms: Number of atoms (must match dictionaries)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.base_intermediate = base_intermediate
        self.delta_intermediate = intermediate_size - base_intermediate
        
        # Per-layer coefficients (this is all we store per layer!)
        self.coeffs_fc1 = nn.Parameter(torch.zeros(num_atoms))
        self.coeffs_fc2 = nn.Parameter(torch.zeros(num_atoms))
        
        # References to shared dictionaries
        self.dds_fc1 = dds_fc1
        self.dds_fc2 = dds_fc2
    
    def get_delta_fc1(self) -> torch.Tensor:
        """Reconstruct delta fc1 weight: (delta_intermediate, hidden_size)."""
        return self.dds_fc1.reconstruct(self.coeffs_fc1)
    
    def get_delta_fc2(self) -> torch.Tensor:
        """Reconstruct delta fc2 weight: (hidden_size, delta_intermediate)."""
        return self.dds_fc2.reconstruct(self.coeffs_fc2)
    
    def forward_delta(self, x: torch.Tensor, x_act: torch.Tensor) -> torch.Tensor:
        """
        Compute delta MLP contribution.
        
        Args:
            x: Input hidden states (B, T, H)
            x_act: Activated intermediate (after base fc1 + activation)
            
        Returns:
            Delta output contribution (B, T, H)
        """
        if self.delta_intermediate <= 0:
            return x.new_zeros(x.shape)
        
        # Reconstruct delta weights
        delta_fc1 = self.get_delta_fc1()  # (delta_inter, hidden)
        delta_fc2 = self.get_delta_fc2()  # (hidden, delta_inter)
        
        # Delta path: x -> delta_fc1 -> act -> delta_fc2
        delta_inter = F.gelu(x @ delta_fc1.T)  # (B, T, delta_inter)
        delta_out = delta_inter @ delta_fc2.T   # (B, T, hidden)
        
        return delta_out
    
    def sparsity_loss(self) -> torch.Tensor:
        """Get L1 sparsity penalty for this layer's coefficients."""
        return (
            self.dds_fc1.sparsity_loss(self.coeffs_fc1) +
            self.dds_fc2.sparsity_loss(self.coeffs_fc2)
        )
    
    def num_delta_params(self) -> int:
        """Number of parameters in delta (just the coefficients)."""
        return self.coeffs_fc1.numel() + self.coeffs_fc2.numel()


class DeltaDictionaryAttention(nn.Module):
    """
    Attention delta weights reconstructed from shared dictionary.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        base_heads: int,
        dds_qkv: DeltaDictionary,
        dds_out: DeltaDictionary,
        num_atoms: int = 16
    ):
        """
        Args:
            hidden_size: Model hidden dimension
            num_heads: Total number of attention heads
            base_heads: Number of heads in base slice
            dds_qkv: Shared dictionary for Q/K/V delta projections
            dds_out: Shared dictionary for output projection delta
            num_atoms: Number of atoms
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.base_heads = base_heads
        self.delta_heads = num_heads - base_heads
        self.head_dim = hidden_size // num_heads
        
        # Per-layer coefficients for Q, K, V, O
        self.coeffs_q = nn.Parameter(torch.zeros(num_atoms))
        self.coeffs_k = nn.Parameter(torch.zeros(num_atoms))
        self.coeffs_v = nn.Parameter(torch.zeros(num_atoms))
        self.coeffs_o = nn.Parameter(torch.zeros(num_atoms))
        
        self.dds_qkv = dds_qkv
        self.dds_out = dds_out
    
    def get_delta_q(self) -> torch.Tensor:
        return self.dds_qkv.reconstruct(self.coeffs_q)
    
    def get_delta_k(self) -> torch.Tensor:
        return self.dds_qkv.reconstruct(self.coeffs_k)
    
    def get_delta_v(self) -> torch.Tensor:
        return self.dds_qkv.reconstruct(self.coeffs_v)
    
    def get_delta_o(self) -> torch.Tensor:
        return self.dds_out.reconstruct(self.coeffs_o)
    
    def sparsity_loss(self) -> torch.Tensor:
        """Get L1 sparsity penalty for this layer's coefficients."""
        return (
            self.dds_qkv.sparsity_loss(self.coeffs_q) +
            self.dds_qkv.sparsity_loss(self.coeffs_k) +
            self.dds_qkv.sparsity_loss(self.coeffs_v) +
            self.dds_out.sparsity_loss(self.coeffs_o)
        )
    
    def num_delta_params(self) -> int:
        """Number of parameters in delta (just the coefficients)."""
        return (
            self.coeffs_q.numel() + self.coeffs_k.numel() +
            self.coeffs_v.numel() + self.coeffs_o.numel()
        )


class DDSManager(nn.Module):
    """
    Manages shared DDS dictionaries across all layers.
    
    Creates and holds the shared atom banks, and provides
    per-layer coefficient modules.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_layers: int,
        base_ratio: float = 0.5,
        base_heads: int = 6,
        num_atoms: int = 16,
        rank: int = 64,
        init_std: float = 0.02
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_atoms = num_atoms
        
        base_intermediate = int(base_ratio * intermediate_size)
        delta_intermediate = intermediate_size - base_intermediate
        delta_heads = num_heads - base_heads
        delta_head_dim = hidden_size // num_heads * delta_heads
        
        # Shared dictionaries for MLP
        self.dds_fc1 = DeltaDictionary(
            in_features=hidden_size,
            out_features=delta_intermediate,
            num_atoms=num_atoms,
            rank=rank,
            init_std=init_std
        )
        self.dds_fc2 = DeltaDictionary(
            in_features=delta_intermediate,
            out_features=hidden_size,
            num_atoms=num_atoms,
            rank=rank,
            init_std=init_std
        )
        
        # Shared dictionaries for Attention
        self.dds_qkv = DeltaDictionary(
            in_features=hidden_size,
            out_features=delta_head_dim,
            num_atoms=num_atoms,
            rank=rank,
            init_std=init_std
        )
        self.dds_out = DeltaDictionary(
            in_features=delta_head_dim,
            out_features=hidden_size,
            num_atoms=num_atoms,
            rank=rank,
            init_std=init_std
        )
        
        # Per-layer coefficient modules
        self.mlp_deltas = nn.ModuleList([
            DeltaDictionaryMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                base_intermediate=base_intermediate,
                dds_fc1=self.dds_fc1,
                dds_fc2=self.dds_fc2,
                num_atoms=num_atoms
            )
            for _ in range(num_layers)
        ])
        
        self.attn_deltas = nn.ModuleList([
            DeltaDictionaryAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                base_heads=base_heads,
                dds_qkv=self.dds_qkv,
                dds_out=self.dds_out,
                num_atoms=num_atoms
            )
            for _ in range(num_layers)
        ])
    
    def total_sparsity_loss(self) -> torch.Tensor:
        """Sum of sparsity losses across all layers."""
        loss = torch.tensor(0.0, device=self.dds_fc1.atom_U.device)
        for mlp_delta in self.mlp_deltas:
            loss = loss + mlp_delta.sparsity_loss()
        for attn_delta in self.attn_deltas:
            loss = loss + attn_delta.sparsity_loss()
        return loss
    
    def orthogonality_loss(self) -> torch.Tensor:
        """Orthogonality regularization on atom banks."""
        return (
            self.dds_fc1.orthogonality_loss() +
            self.dds_fc2.orthogonality_loss() +
            self.dds_qkv.orthogonality_loss() +
            self.dds_out.orthogonality_loss()
        )
    
    def export_delta_pack(self, layer_indices: Optional[list] = None) -> dict:
        """
        Export delta coefficients as a compact "upgrade pack".
        
        Args:
            layer_indices: Which layers to export (None = all)
            
        Returns:
            Dictionary with coefficients and metadata
        """
        if layer_indices is None:
            layer_indices = list(range(self.num_layers))
        
        pack = {
            'num_atoms': self.num_atoms,
            'layer_indices': layer_indices,
            'mlp_coeffs': {},
            'attn_coeffs': {},
        }
        
        for idx in layer_indices:
            pack['mlp_coeffs'][idx] = {
                'fc1': self.mlp_deltas[idx].coeffs_fc1.detach().cpu(),
                'fc2': self.mlp_deltas[idx].coeffs_fc2.detach().cpu(),
            }
            pack['attn_coeffs'][idx] = {
                'q': self.attn_deltas[idx].coeffs_q.detach().cpu(),
                'k': self.attn_deltas[idx].coeffs_k.detach().cpu(),
                'v': self.attn_deltas[idx].coeffs_v.detach().cpu(),
                'o': self.attn_deltas[idx].coeffs_o.detach().cpu(),
            }
        
        return pack
    
    def load_delta_pack(self, pack: dict):
        """Load delta coefficients from an upgrade pack."""
        for idx in pack['layer_indices']:
            self.mlp_deltas[idx].coeffs_fc1.data.copy_(pack['mlp_coeffs'][idx]['fc1'])
            self.mlp_deltas[idx].coeffs_fc2.data.copy_(pack['mlp_coeffs'][idx]['fc2'])
            self.attn_deltas[idx].coeffs_q.data.copy_(pack['attn_coeffs'][idx]['q'])
            self.attn_deltas[idx].coeffs_k.data.copy_(pack['attn_coeffs'][idx]['k'])
            self.attn_deltas[idx].coeffs_v.data.copy_(pack['attn_coeffs'][idx]['v'])
            self.attn_deltas[idx].coeffs_o.data.copy_(pack['attn_coeffs'][idx]['o'])
    
    def delta_pack_size_bytes(self) -> int:
        """Estimate size of delta pack in bytes (assuming float32)."""
        # Per layer: 2 MLP coeffs + 4 Attn coeffs = 6 * num_atoms floats
        return self.num_layers * 6 * self.num_atoms * 4
    
    def atom_bank_size_bytes(self) -> int:
        """Size of shared atom banks in bytes."""
        total_params = sum(p.numel() for p in [
            self.dds_fc1.atom_U, self.dds_fc1.atom_V,
            self.dds_fc2.atom_U, self.dds_fc2.atom_V,
            self.dds_qkv.atom_U, self.dds_qkv.atom_V,
            self.dds_out.atom_U, self.dds_out.atom_V,
        ])
        return total_params * 4  # float32

