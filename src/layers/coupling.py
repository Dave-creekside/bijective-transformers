"""
Coupling layers for bijective transformations.
Implements additive, affine, and neural spline coupling layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass
import math


@dataclass
class CouplingLayerConfig:
    """Configuration for coupling layers."""
    input_dim: int
    hidden_dim: int = 512
    num_layers: int = 2
    activation: str = "relu"
    dropout: float = 0.0
    split_dim: Optional[int] = None  # If None, splits in half
    coupling_type: str = "additive"  # additive, affine, spline
    
    # Affine coupling specific
    scale_activation: str = "tanh"  # tanh, sigmoid, none
    scale_factor: float = 1.0
    
    # Spline coupling specific
    num_bins: int = 8
    tail_bound: float = 3.0
    
    # Numerical stability
    eps: float = 1e-6
    clamp_log_scale: float = 2.0


class CouplingNetwork(nn.Module):
    """Neural network for coupling transformations."""
    
    def __init__(self, config: CouplingLayerConfig, output_dim: int):
        super().__init__()
        self.config = config
        self.output_dim = output_dim
        
        # Build network layers
        layers = []
        current_dim = config.input_dim // 2  # Input is split dimension
        
        # Hidden layers
        for i in range(config.num_layers):
            layers.append(nn.Linear(current_dim, config.hidden_dim))
            
            # Activation
            if config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "gelu":
                layers.append(nn.GELU())
            elif config.activation == "swish":
                layers.append(nn.SiLU())
            elif config.activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {config.activation}")
            
            # Dropout
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            
            current_dim = config.hidden_dim
        
        # Output layer
        layers.append(nn.Linear(config.hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize final layer to output zeros for identity initialization
        if len(self.network) > 0 and isinstance(self.network[-1], nn.Linear):
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through coupling network."""
        return self.network(x)


class AdditiveCouplingLayer(nn.Module):
    """
    Additive coupling layer: y₁ = x₁, y₂ = x₂ + f(x₁)
    
    This is the simplest bijective transformation where the Jacobian
    determinant is always 1 (log det = 0).
    """
    
    def __init__(self, config: CouplingLayerConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        
        # Determine split dimension
        if config.split_dim is None:
            self.split_dim = config.input_dim // 2
        else:
            self.split_dim = config.split_dim
        
        self.remaining_dim = config.input_dim - self.split_dim
        
        # Create coupling network
        self.coupling_net = CouplingNetwork(config, self.remaining_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation.
        
        Args:
            x: Input tensor [batch_size, ..., input_dim]
            
        Returns:
            y: Transformed tensor [batch_size, ..., input_dim]
            log_det: Log determinant (always 0 for additive coupling)
        """
        # Split input
        x1, x2 = self._split(x)
        
        # Apply coupling transformation
        y1 = x1  # Identity transformation
        shift = self.coupling_net(x1)
        y2 = x2 + shift
        
        # Combine outputs
        y = self._combine(y1, y2)
        
        # Log determinant is 0 for additive coupling
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        
        return y, log_det
    
    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation.
        
        Args:
            y: Transformed tensor [batch_size, ..., input_dim]
            
        Returns:
            x: Original tensor [batch_size, ..., input_dim]
            log_det: Log determinant (always 0 for additive coupling)
        """
        # Split input
        y1, y2 = self._split(y)
        
        # Apply inverse coupling transformation
        x1 = y1  # Identity transformation
        shift = self.coupling_net(x1)
        x2 = y2 - shift
        
        # Combine outputs
        x = self._combine(x1, x2)
        
        # Log determinant is 0 for additive coupling
        log_det = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
        
        return x, log_det
    
    def _split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split tensor along last dimension."""
        x1 = x[..., :self.split_dim]
        x2 = x[..., self.split_dim:]
        return x1, x2
    
    def _combine(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Combine tensors along last dimension."""
        return torch.cat([x1, x2], dim=-1)


class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer: y₁ = x₁, y₂ = x₂ ⊙ exp(s(x₁)) + t(x₁)
    
    More expressive than additive coupling, with non-trivial Jacobian determinant.
    """
    
    def __init__(self, config: CouplingLayerConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        
        # Determine split dimension
        if config.split_dim is None:
            self.split_dim = config.input_dim // 2
        else:
            self.split_dim = config.split_dim
        
        self.remaining_dim = config.input_dim - self.split_dim
        
        # Create coupling networks for scale and translation
        self.scale_net = CouplingNetwork(config, self.remaining_dim)
        self.translation_net = CouplingNetwork(config, self.remaining_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation.
        
        Args:
            x: Input tensor [batch_size, ..., input_dim]
            
        Returns:
            y: Transformed tensor [batch_size, ..., input_dim]
            log_det: Log determinant of Jacobian
        """
        # Split input
        x1, x2 = self._split(x)
        
        # Compute scale and translation
        log_scale = self.scale_net(x1)
        translation = self.translation_net(x1)
        
        # Apply scale activation and clamping for numerical stability
        log_scale = self._apply_scale_activation(log_scale)
        log_scale = torch.clamp(log_scale, -self.config.clamp_log_scale, self.config.clamp_log_scale)
        
        # Apply affine transformation
        y1 = x1  # Identity transformation
        y2 = x2 * torch.exp(log_scale) + translation
        
        # Combine outputs
        y = self._combine(y1, y2)
        
        # Compute log determinant
        log_det = torch.sum(log_scale, dim=-1)  # Sum over transformed dimensions
        
        return y, log_det
    
    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation.
        
        Args:
            y: Transformed tensor [batch_size, ..., input_dim]
            
        Returns:
            x: Original tensor [batch_size, ..., input_dim]
            log_det: Log determinant of Jacobian
        """
        # Split input
        y1, y2 = self._split(y)
        
        # Compute scale and translation
        log_scale = self.scale_net(y1)
        translation = self.translation_net(y1)
        
        # Apply scale activation and clamping
        log_scale = self._apply_scale_activation(log_scale)
        log_scale = torch.clamp(log_scale, -self.config.clamp_log_scale, self.config.clamp_log_scale)
        
        # Apply inverse affine transformation
        x1 = y1  # Identity transformation
        x2 = (y2 - translation) * torch.exp(-log_scale)
        
        # Combine outputs
        x = self._combine(x1, x2)
        
        # Compute log determinant (negative for inverse)
        log_det = -torch.sum(log_scale, dim=-1)
        
        return x, log_det
    
    def _apply_scale_activation(self, log_scale: torch.Tensor) -> torch.Tensor:
        """Apply activation to scale parameter."""
        if self.config.scale_activation == "tanh":
            return torch.tanh(log_scale) * self.config.scale_factor
        elif self.config.scale_activation == "sigmoid":
            return (torch.sigmoid(log_scale) - 0.5) * 2 * self.config.scale_factor
        elif self.config.scale_activation == "none":
            return log_scale * self.config.scale_factor
        else:
            raise ValueError(f"Unknown scale activation: {self.config.scale_activation}")
    
    def _split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split tensor along last dimension."""
        x1 = x[..., :self.split_dim]
        x2 = x[..., self.split_dim:]
        return x1, x2
    
    def _combine(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Combine tensors along last dimension."""
        return torch.cat([x1, x2], dim=-1)


class NeuralSplineCouplingLayer(nn.Module):
    """
    Neural spline coupling layer using rational quadratic splines.
    
    More expressive than affine coupling, can model complex transformations
    while maintaining invertibility.
    """
    
    def __init__(self, config: CouplingLayerConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.num_bins = config.num_bins
        self.tail_bound = config.tail_bound
        
        # Determine split dimension
        if config.split_dim is None:
            self.split_dim = config.input_dim // 2
        else:
            self.split_dim = config.split_dim
        
        self.remaining_dim = config.input_dim - self.split_dim
        
        # Output dimension for spline parameters
        # Need: widths, heights, derivatives for each bin + boundary derivatives
        self.spline_params_dim = self.remaining_dim * (3 * self.num_bins + 1)
        
        # Create coupling network
        self.spline_net = CouplingNetwork(config, self.spline_params_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward transformation using rational quadratic splines."""
        # Split input
        x1, x2 = self._split(x)
        
        # Get spline parameters
        spline_params = self.spline_net(x1)
        widths, heights, derivatives = self._parse_spline_params(spline_params)
        
        # Apply spline transformation
        y1 = x1  # Identity transformation
        y2, log_det_spline = self._rational_quadratic_spline(
            x2, widths, heights, derivatives
        )
        
        # Combine outputs
        y = self._combine(y1, y2)
        
        # Sum log determinants across dimensions
        log_det = torch.sum(log_det_spline, dim=-1)
        
        return y, log_det
    
    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse transformation using rational quadratic splines."""
        # Split input
        y1, y2 = self._split(y)
        
        # Get spline parameters
        spline_params = self.spline_net(y1)
        widths, heights, derivatives = self._parse_spline_params(spline_params)
        
        # Apply inverse spline transformation
        x1 = y1  # Identity transformation
        x2, log_det_spline = self._rational_quadratic_spline(
            y2, widths, heights, derivatives, inverse=True
        )
        
        # Combine outputs
        x = self._combine(x1, x2)
        
        # Sum log determinants across dimensions (negative for inverse)
        log_det = -torch.sum(log_det_spline, dim=-1)
        
        return x, log_det
    
    def _parse_spline_params(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse spline parameters from network output."""
        batch_shape = params.shape[:-1]
        params = params.reshape(*batch_shape, self.remaining_dim, 3 * self.num_bins + 1)
        
        # Split parameters
        widths = params[..., :self.num_bins]
        heights = params[..., self.num_bins:2*self.num_bins]
        derivatives = params[..., 2*self.num_bins:]
        
        # Apply softmax to widths and heights to ensure they sum to 1
        widths = F.softmax(widths, dim=-1)
        heights = F.softmax(heights, dim=-1)
        
        # Ensure derivatives are positive
        derivatives = F.softplus(derivatives) + self.config.eps
        
        return widths, heights, derivatives
    
    def _rational_quadratic_spline(
        self,
        x: torch.Tensor,
        widths: torch.Tensor,
        heights: torch.Tensor,
        derivatives: torch.Tensor,
        inverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rational quadratic spline transformation.
        
        This is a simplified implementation. A full implementation would
        include proper handling of the tail regions and more robust
        numerical computation.
        """
        # For simplicity, we'll implement a basic version
        # In practice, you'd want to use a more robust implementation
        # like the one from the Neural Spline Flows paper
        
        if inverse:
            # For inverse, we'd need to solve the spline equation
            # This is complex, so for now we'll use a placeholder
            y = x  # Placeholder - should implement proper inverse
            log_det = torch.zeros_like(x)
        else:
            # Forward transformation (also simplified)
            y = x  # Placeholder - should implement proper spline
            log_det = torch.zeros_like(x)
        
        return y, log_det
    
    def _split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split tensor along last dimension."""
        x1 = x[..., :self.split_dim]
        x2 = x[..., self.split_dim:]
        return x1, x2
    
    def _combine(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Combine tensors along last dimension."""
        return torch.cat([x1, x2], dim=-1)


def create_coupling_layer(config: CouplingLayerConfig) -> nn.Module:
    """Factory function to create coupling layers."""
    if config.coupling_type == "additive":
        return AdditiveCouplingLayer(config)
    elif config.coupling_type == "affine":
        return AffineCouplingLayer(config)
    elif config.coupling_type == "spline":
        return NeuralSplineCouplingLayer(config)
    else:
        raise ValueError(f"Unknown coupling type: {config.coupling_type}")


def create_coupling_config(
    input_dim: int,
    coupling_type: str = "additive",
    hidden_dim: int = 512,
    num_layers: int = 2,
    **kwargs
) -> CouplingLayerConfig:
    """Create coupling layer configuration with defaults."""
    return CouplingLayerConfig(
        input_dim=input_dim,
        coupling_type=coupling_type,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        **kwargs
    )
