"""
Activation extraction for transformer models using PyTorch hooks.

Targets post-GELU activations in MLP layers, following the Rosetta Neurons
paper's approach of extracting activations after the nonlinearity.
"""
from typing import Dict, List, Callable
import torch
import torch.nn as nn


class ActivationCache:
    """
    Manages forward hooks to capture activations from specified layers.
    
    Usage:
        cache = ActivationCache(model, layer_names)
        output = model(input)
        activations = cache.get_activations()
        cache.clear()
    """
    
    def __init__(self, model: nn.Module, layer_names: List[str]):
        """
        Args:
            model: The model to hook
            layer_names: List of layer names to capture (e.g., 'gpt_neox.layers.0.mlp.act')
        """
        self.model = model
        self.layer_names = layer_names
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        self._register_hooks()
    
    def _get_module(self, name: str) -> nn.Module:
        """Get a submodule by its full name (dot-separated)."""
        module = self.model
        for part in name.split('.'):
            module = getattr(module, part)
        return module
    
    def _make_hook(self, name: str) -> Callable:
        """Create a hook function that stores activations."""
        def hook(module: nn.Module, input: tuple, output: torch.Tensor):
            # Detach and store on CPU to save GPU memory during accumulation
            self.activations[name] = output.detach()
        return hook
    
    def _register_hooks(self):
        """Register forward hooks on all specified layers."""
        for name in self.layer_names:
            module = self._get_module(name)
            hook = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Return captured activations."""
        return self.activations
    
    def clear(self):
        """Clear stored activations to free memory."""
        self.activations = {}
    
    def remove_hooks(self):
        """Remove all hooks from the model."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __del__(self):
        """Cleanup hooks when cache is deleted."""
        self.remove_hooks()


def get_mlp_activation_layer_names(model_name: str, layer_indices: List[int]) -> List[str]:
    """
    Get the layer names for post-GELU MLP activations for a given model.
    
    Args:
        model_name: HuggingFace model name
        layer_indices: Which transformer layers to extract from
        
    Returns:
        List of layer names to hook
    """
    # Pythia / GPT-NeoX architecture
    if "pythia" in model_name.lower() or "neox" in model_name.lower():
        return [f"gpt_neox.layers.{i}.mlp.act" for i in layer_indices]
    
    # GPT-2 architecture
    elif "gpt2" in model_name.lower():
        # GPT-2 uses: c_fc -> act -> c_proj
        # We need to hook after the activation
        return [f"transformer.h.{i}.mlp.act" for i in layer_indices]
    
    # LLaMA architecture
    elif "llama" in model_name.lower() or "gemma" in model_name.lower():
        # LLaMA uses SiLU (Swish) activation: gate_proj * act(up_proj)
        # The activation is applied inline, so we hook the mlp module
        # and extract intermediate activations differently
        return [f"model.layers.{i}.mlp" for i in layer_indices]
    
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")


class PythiaMLPHook:
    """
    Specialized hook for Pythia models to extract post-GELU activations.
    
    In Pythia, the MLP structure is:
        x -> dense_h_to_4h -> gelu -> dense_4h_to_h -> output
    
    We want the activations after gelu (before dense_4h_to_h).
    The 'act' module is the GELU activation, so hooking it gives us what we want.
    """
    pass  # The generic ActivationCache works for Pythia since 'act' is a separate module
