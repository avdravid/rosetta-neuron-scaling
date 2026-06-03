"""
Core neuron matching via Pearson correlation.

Implements the matching procedure from Rosetta Neurons, adapted for language models.
Uses vectorized operations for efficiency.
"""
import torch
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


def normalize_activations(
    activations: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Normalize activations to zero mean and unit variance.
    
    Args:
        activations: [batch, seq_len, hidden_dim] or [n_samples, hidden_dim]
        mean: [hidden_dim]
        std: [hidden_dim]
        eps: Small constant for numerical stability
        
    Returns:
        Normalized activations with same shape as input
    """
    return (activations - mean) / (std + eps)


def compute_correlation_matrix(
    activations1: torch.Tensor,
    activations2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Pearson correlation between all pairs of neurons.
    
    Assumes activations are already normalized (zero mean, unit variance).
    
    Args:
        activations1: [n_samples, dim1] - normalized activations from model 1
        activations2: [n_samples, dim2] - normalized activations from model 2
        
    Returns:
        Correlation matrix of shape [dim1, dim2]
    """
    n_samples = activations1.shape[0]
    
    # Pearson correlation for normalized data: corr = (1/N) * sum(x * y)
    # Using einsum for efficient batch matrix multiplication
    # 'ni,nj->ij' means: for each (i,j), sum over n: activations1[:, i] * activations2[:, j]
    correlation = torch.einsum('ni,nj->ij', activations1, activations2) / n_samples
    
    return correlation


def compute_correlation_matrix_chunked(
    activations1: torch.Tensor,
    activations2: torch.Tensor,
    chunk_size: int = 1024,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Compute correlation matrix in chunks to handle large hidden dimensions.
    
    Args:
        activations1: [n_samples, dim1]
        activations2: [n_samples, dim2]
        chunk_size: Number of neurons to process at once
        device: Device for computation
        
    Returns:
        Correlation matrix of shape [dim1, dim2]
    """
    n_samples, dim1 = activations1.shape
    _, dim2 = activations2.shape
    
    correlation = torch.zeros(dim1, dim2, device='cpu')
    
    # Process in chunks over dim1
    for i in range(0, dim1, chunk_size):
        i_end = min(i + chunk_size, dim1)
        chunk1 = activations1[:, i:i_end].to(device)
        
        # Process in chunks over dim2
        for j in range(0, dim2, chunk_size):
            j_end = min(j + chunk_size, dim2)
            chunk2 = activations2[:, j:j_end].to(device)
            
            # Compute correlation for this chunk
            corr_chunk = torch.einsum('ni,nj->ij', chunk1, chunk2) / n_samples
            correlation[i:i_end, j:j_end] = corr_chunk.cpu()
    
    return correlation


class NeuronMatcher:
    """
    Main class for computing neuron matches between two models.
    
    Follows the Rosetta Neurons approach:
    1. Extract post-activation MLP activations from both models
    2. Compute mean and std for each neuron across the dataset
    3. Normalize activations
    4. Compute Pearson correlation between all pairs of neurons
    """
    
    def __init__(
        self,
        layer_names1: List[str],
        layer_names2: List[str],
        device: str = "cuda"
    ):
        self.layer_names1 = layer_names1
        self.layer_names2 = layer_names2
        self.device = device
        
        # Will be populated during matching
        self.stats1: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None
        self.stats2: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None
        self.correlation_matrix: Optional[torch.Tensor] = None
        
        # Track layer boundaries for visualization
        self.layer_boundaries1: Optional[List[int]] = None
        self.layer_boundaries2: Optional[List[int]] = None
    
    def compute_full_correlation_matrix(
        self,
        activations1_list: List[torch.Tensor],
        activations2_list: List[torch.Tensor],
        stats1_list: List[Tuple[torch.Tensor, torch.Tensor]],
        stats2_list: List[Tuple[torch.Tensor, torch.Tensor]],
        chunk_size: int = 2048
    ) -> torch.Tensor:
        """
        Compute the full correlation matrix across all layers.
        
        Args:
            activations1_list: List of activation tensors for model 1, one per layer
            activations2_list: List of activation tensors for model 2, one per layer
            stats1_list: List of (mean, std) tuples for model 1
            stats2_list: List of (mean, std) tuples for model 2
            chunk_size: Chunk size for memory-efficient computation
            
        Returns:
            Full correlation matrix [total_neurons_1, total_neurons_2]
        """
        # Compute total neurons and layer boundaries
        dims1 = [a.shape[-1] for a in activations1_list]
        dims2 = [a.shape[-1] for a in activations2_list]
        
        total_dim1 = sum(dims1)
        total_dim2 = sum(dims2)
        
        self.layer_boundaries1 = [0] + list(torch.cumsum(torch.tensor(dims1), 0).tolist())
        self.layer_boundaries2 = [0] + list(torch.cumsum(torch.tensor(dims2), 0).tolist())
        
        print(f"Computing correlation matrix of shape [{total_dim1}, {total_dim2}]")
        print(f"Model 1: {len(activations1_list)} layers, {total_dim1} total neurons")
        print(f"Model 2: {len(activations2_list)} layers, {total_dim2} total neurons")
        
        # Initialize full correlation matrix on CPU
        full_correlation = torch.zeros(total_dim1, total_dim2)
        
        # Compute correlations block by block (layer pair by layer pair)
        row_offset = 0
        for i, (act1, (mean1, std1)) in enumerate(tqdm(
            zip(activations1_list, stats1_list), 
            total=len(activations1_list),
            desc="Computing correlations"
        )):
            # Flatten and normalize activations from model 1
            act1_flat = act1.reshape(-1, act1.shape[-1]).float()
            act1_norm = normalize_activations(act1_flat, mean1.cpu(), std1.cpu())
            
            col_offset = 0
            for j, (act2, (mean2, std2)) in enumerate(zip(activations2_list, stats2_list)):
                # Flatten and normalize activations from model 2
                act2_flat = act2.reshape(-1, act2.shape[-1]).float()
                act2_norm = normalize_activations(act2_flat, mean2.cpu(), std2.cpu())
                
                # Compute correlation for this layer pair
                corr_block = compute_correlation_matrix_chunked(
                    act1_norm, act2_norm, 
                    chunk_size=chunk_size,
                    device=self.device
                )
                
                # Store in full matrix
                full_correlation[
                    row_offset:row_offset + act1.shape[-1],
                    col_offset:col_offset + act2.shape[-1]
                ] = corr_block
                
                col_offset += act2.shape[-1]
            
            row_offset += act1.shape[-1]
        
        # Handle NaN values (from zero-variance neurons)
        nan_mask = torch.isnan(full_correlation)
        if nan_mask.any():
            print(f"Warning: {nan_mask.sum().item()} NaN values in correlation matrix, setting to 0")
            full_correlation[nan_mask] = 0
        
        self.correlation_matrix = full_correlation
        return full_correlation
    
    def find_best_matches(self, top_k: int = 5) -> Dict[str, torch.Tensor]:
        """
        Find the top-k best matching neurons for each neuron.
        
        Returns:
            Dict with:
                - 'forward_matches': [total_neurons_1, top_k] - best matches in model 2 for each model 1 neuron
                - 'forward_scores': [total_neurons_1, top_k] - corresponding correlation scores
                - 'backward_matches': [total_neurons_2, top_k] - best matches in model 1 for each model 2 neuron
                - 'backward_scores': [total_neurons_2, top_k] - corresponding correlation scores
        """
        if self.correlation_matrix is None:
            raise ValueError("Must compute correlation matrix first")
        
        # Forward: for each neuron in model 1, find best matches in model 2
        forward_scores, forward_matches = torch.topk(self.correlation_matrix, k=top_k, dim=1)
        
        # Backward: for each neuron in model 2, find best matches in model 1
        backward_scores, backward_matches = torch.topk(self.correlation_matrix.T, k=top_k, dim=1)
        
        return {
            'forward_matches': forward_matches,
            'forward_scores': forward_scores,
            'backward_matches': backward_matches,
            'backward_scores': backward_scores
        }
    
    def find_best_buddies(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find 'best buddy' pairs - neurons that are each other's best match.
        
        Returns:
            Tuple of:
                - indices1: Indices of best-buddy neurons in model 1
                - indices2: Indices of best-buddy neurons in model 2
                - scores: Correlation scores for each pair
        """
        if self.correlation_matrix is None:
            raise ValueError("Must compute correlation matrix first")
        
        # Best match in model 2 for each model 1 neuron
        best_in_2 = self.correlation_matrix.argmax(dim=1)  # [dim1]
        
        # Best match in model 1 for each model 2 neuron
        best_in_1 = self.correlation_matrix.argmax(dim=0)  # [dim2]
        
        # Find mutual best matches (best buddies)
        # For neuron i in model 1, if its best match is j in model 2,
        # and j's best match is i, then (i, j) is a best buddy pair
        indices1 = torch.arange(self.correlation_matrix.shape[0])
        is_best_buddy = best_in_1[best_in_2] == indices1
        
        buddy_indices1 = indices1[is_best_buddy]
        buddy_indices2 = best_in_2[is_best_buddy]
        buddy_scores = self.correlation_matrix[buddy_indices1, buddy_indices2]
        
        return buddy_indices1, buddy_indices2, buddy_scores