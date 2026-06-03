"""
Online statistics computation for activation normalization.

Uses Welford's online algorithm for numerically stable computation
of mean and variance in a single pass.
"""
import torch
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RunningStats:
    """
    Running statistics using Welford's online algorithm.
    
    Computes mean and variance in a numerically stable way
    without storing all data points.
    """
    count: int = 0
    mean: torch.Tensor = None
    M2: torch.Tensor = None  # Sum of squared differences from mean
    
    def update(self, x: torch.Tensor):
        """
        Update statistics with a new batch of data.
        
        Args:
            x: Tensor of shape [batch, seq_len, hidden_dim] or [n_samples, hidden_dim]
        """
        # Flatten to [n_samples, hidden_dim]
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        
        x = x.float()  # Ensure float32 for numerical stability
        
        batch_count = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        
        if self.mean is None:
            # First batch
            self.count = batch_count
            self.mean = batch_mean
            self.M2 = batch_var * batch_count
        else:
            # Combine with existing statistics (parallel algorithm)
            new_count = self.count + batch_count
            delta = batch_mean - self.mean
            
            self.mean = self.mean + delta * batch_count / new_count
            self.M2 = self.M2 + batch_var * batch_count + \
                      delta ** 2 * self.count * batch_count / new_count
            self.count = new_count
    
    @property
    def std(self) -> torch.Tensor:
        """Compute standard deviation."""
        if self.count < 2:
            return torch.ones_like(self.mean)
        return torch.sqrt(self.M2 / self.count)
    
    def get_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) tuple."""
        return self.mean, self.std


class MultiLayerStats:
    """
    Track running statistics for multiple layers simultaneously.
    """
    
    def __init__(self, layer_names: List[str]):
        self.layer_names = layer_names
        self.stats: Dict[str, RunningStats] = {name: RunningStats() for name in layer_names}
    
    def update(self, activations: Dict[str, torch.Tensor]):
        """
        Update statistics for all layers.
        
        Args:
            activations: Dict mapping layer names to activation tensors
        """
        for name, tensor in activations.items():
            if name in self.stats:
                self.stats[name].update(tensor)
    
    def get_all_stats(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Return dict mapping layer names to (mean, std) tuples."""
        return {name: stats.get_stats() for name, stats in self.stats.items()}
    
    def get_stats_list(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Return list of (mean, std) tuples in layer order."""
        return [self.stats[name].get_stats() for name in self.layer_names]