"""
Visualization utilities for neuron matching results.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Optional, Tuple
from pathlib import Path


def plot_correlation_matrix(
    correlation: torch.Tensor,
    layer_boundaries1: Optional[List[int]] = None,
    layer_boundaries2: Optional[List[int]] = None,
    title: str = "Neuron Correlation Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    show_layer_grid: bool = True,
    dpi: int = 150
) -> plt.Figure:
    """
    Plot the correlation matrix with optional layer boundaries.
    
    Args:
        correlation: [dim1, dim2] correlation matrix
        layer_boundaries1: Cumulative neuron counts for model 1 layers
        layer_boundaries2: Cumulative neuron counts for model 2 layers
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
        cmap: Colormap
        vmin, vmax: Color scale limits
        show_layer_grid: Whether to show grid lines at layer boundaries
        dpi: Resolution for saved figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot correlation matrix
    corr_np = correlation.numpy() if isinstance(correlation, torch.Tensor) else correlation
    im = ax.imshow(corr_np, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pearson Correlation', fontsize=12)
    
    # Add layer boundary lines
    if show_layer_grid and layer_boundaries1 is not None:
        for b in layer_boundaries1[1:-1]:  # Skip first (0) and last (end)
            ax.axhline(y=b - 0.5, color='black', linewidth=0.5, alpha=0.5)
    
    if show_layer_grid and layer_boundaries2 is not None:
        for b in layer_boundaries2[1:-1]:
            ax.axvline(x=b - 0.5, color='black', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel('Model 2 Neurons', fontsize=12)
    ax.set_ylabel('Model 1 Neurons', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved correlation matrix plot to {save_path}")
    
    return fig


def plot_diagonal_analysis(
    correlation: torch.Tensor,
    layer_boundaries: Optional[List[int]] = None,
    title: str = "Diagonal Correlation Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    For sanity check: analyze diagonal values when comparing same model to itself.
    
    Args:
        correlation: [dim, dim] square correlation matrix
        layer_boundaries: Cumulative neuron counts per layer
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    assert correlation.shape[0] == correlation.shape[1], "Matrix must be square for diagonal analysis"
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Histogram of diagonal values
    diagonal = torch.diag(correlation).numpy()
    axes[0].hist(diagonal, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=1.0, color='red', linestyle='--', label='Perfect correlation')
    axes[0].axvline(x=diagonal.mean(), color='green', linestyle='--', label=f'Mean: {diagonal.mean():.4f}')
    axes[0].set_xlabel('Correlation')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Diagonal Values Distribution')
    axes[0].legend()
    
    # 2. Diagonal values over neuron index
    axes[1].plot(diagonal, alpha=0.7, linewidth=0.5)
    if layer_boundaries is not None:
        for b in layer_boundaries[1:-1]:
            axes[1].axvline(x=b, color='red', linewidth=0.5, alpha=0.5)
    axes[1].set_xlabel('Neuron Index')
    axes[1].set_ylabel('Self-Correlation')
    axes[1].set_title('Diagonal Values by Neuron')
    axes[1].set_ylim(0, 1.1)
    
    # 3. Per-layer mean diagonal value
    if layer_boundaries is not None:
        layer_means = []
        layer_stds = []
        for i in range(len(layer_boundaries) - 1):
            start, end = layer_boundaries[i], layer_boundaries[i + 1]
            layer_diag = diagonal[start:end]
            layer_means.append(layer_diag.mean())
            layer_stds.append(layer_diag.std())
        
        x = range(len(layer_means))
        axes[2].bar(x, layer_means, yerr=layer_stds, capsize=3, alpha=0.7)
        axes[2].set_xlabel('Layer Index')
        axes[2].set_ylabel('Mean Self-Correlation')
        axes[2].set_title('Per-Layer Diagonal Mean')
        axes[2].set_ylim(0, 1.1)
    else:
        axes[2].text(0.5, 0.5, 'No layer boundaries provided', 
                     ha='center', va='center', transform=axes[2].transAxes)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved diagonal analysis plot to {save_path}")
    
    return fig


def plot_layer_block_correlations(
    correlation: torch.Tensor,
    layer_boundaries1: List[int],
    layer_boundaries2: List[int],
    title: str = "Layer-wise Correlation Blocks",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot mean correlation for each layer pair as a heatmap.
    
    Args:
        correlation: Full correlation matrix
        layer_boundaries1: Boundaries for model 1
        layer_boundaries2: Boundaries for model 2
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    n_layers1 = len(layer_boundaries1) - 1
    n_layers2 = len(layer_boundaries2) - 1
    
    layer_corr = torch.zeros(n_layers1, n_layers2)
    
    for i in range(n_layers1):
        for j in range(n_layers2):
            block = correlation[
                layer_boundaries1[i]:layer_boundaries1[i+1],
                layer_boundaries2[j]:layer_boundaries2[j+1]
            ]
            layer_corr[i, j] = block.mean()
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(layer_corr.numpy(), cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='Mean Correlation')
    
    ax.set_xlabel('Model 2 Layer', fontsize=12)
    ax.set_ylabel('Model 1 Layer', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add text annotations
    for i in range(n_layers1):
        for j in range(n_layers2):
            val = layer_corr[i, j].item()
            color = 'white' if abs(val) > 0.25 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   fontsize=6, color=color)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved layer block correlation plot to {save_path}")
    
    return fig


def plot_best_buddy_histogram(
    buddy_scores: torch.Tensor,
    title: str = "Best Buddy Pair Correlations",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot histogram of best buddy pair correlation scores.
    
    Args:
        buddy_scores: Correlation scores for best buddy pairs
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    scores_np = buddy_scores.numpy() if isinstance(buddy_scores, torch.Tensor) else buddy_scores
    
    ax.hist(scores_np, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=scores_np.mean(), color='red', linestyle='--', 
               label=f'Mean: {scores_np.mean():.4f}')
    ax.axvline(x=np.median(scores_np), color='green', linestyle='--',
               label=f'Median: {np.median(scores_np):.4f}')
    
    ax.set_xlabel('Correlation Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{title}\n({len(scores_np)} best buddy pairs)', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved best buddy histogram to {save_path}")
    
    return fig