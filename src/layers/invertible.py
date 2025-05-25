"""
Invertible layers for bijective transformers.
Implements RevNet-style residual connections and invertible layer normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
import math


@dataclass
class InvertibleConfig:
    """Configuration for invertible layers."""
    embed_dim: int
    split_dim: Optional[int] = None  # If None, splits in half
    coupling_type: str = "additive"  # additive, affine
    activation: str = "gelu"
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
    # Memory optimization
    use_checkpointing: bool = False
    
    # Numerical stability
    eps: float = 1e-6
    clamp_log_scale: float = 2.0


class InvertibleResidualConnection(nn.Module):
    """
    Invertible residual connection using RevNet-style coupling.
    
    Forward:  x1, x2 = split(x)
              y1 = x1 + F(x2)  
              y2 = x2 + G(y1)
              y = concat(y1, y2)
    
    Inverse:  y1, y2 = split(y)
              x2 = y2 - G(y1)
              x1 = y1 - F(x2)
              x = concat(x1, x2)
    """
    
    def __init__(
        self,
        config: InvertibleConfig,
        F_function: nn.Module,
        G_function: nn.Module
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        
        # Determine split dimension
        if config.split_dim is None:
            self.split_dim = config.embed_dim // 2
        else:
            self.split_dim = config.split_dim
        
        self.remaining_dim = config.embed_dim - self.split_dim
        
        # Store the coupling functions
        self.F_function = F_function
        self.G_function = G_function
        
        # Ensure functions have correct output dimensions
        self._validate_functions()
    
    def _validate_functions(self):
        """Validate that F and G functions have correct dimensions."""
        # Test with dummy input
        dummy_input = torch.randn(1, self.remaining_dim)
        
        try:
            f_output = self.F_function(dummy_input)
            assert f_output.shape[-1] == self.split_dim, \
                f"F function output dim {f_output.shape[-1]} != split_dim {self.split_dim}"
        except Exception as e:
            raise ValueError(f"F function validation failed: {e}")
        
        try:
            dummy_input_g = torch.randn(1, self.split_dim)
            g_output = self.G_function(dummy_input_g)
            assert g_output.shape[-1] == self.remaining_dim, \
                f"G function output dim {g_output.shape[-1]} != remaining_dim {self.remaining_dim}"
        except Exception as e:
            raise ValueError(f"G function validation failed: {e}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through invertible residual connection.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            
        Returns:
            y: Output tensor [batch_size, seq_len, embed_dim]
            log_det: Log determinant (always 0 for additive coupling)
        """
        # Split input
        x1, x2 = self._split(x)
        
        # Apply RevNet coupling
        y1 = x1 + self.F_function(x2)
        y2 = x2 + self.G_function(y1)
        
        # Combine outputs
        y = self._combine(y1, y2)
        
        # Log determinant is 0 for additive coupling
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        
        return y, log_det
    
    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass through invertible residual connection.
        
        Args:
            y: Output tensor [batch_size, seq_len, embed_dim]
            
        Returns:
            x: Input tensor [batch_size, seq_len, embed_dim]
            log_det: Log determinant (always 0 for additive coupling)
        """
        # Split output
        y1, y2 = self._split(y)
        
        # Apply inverse RevNet coupling
        x2 = y2 - self.G_function(y1)
        x1 = y1 - self.F_function(x2)
        
        # Combine inputs
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


class InvertibleLayerNorm(nn.Module):
    """
    Invertible layer normalization.
    
    Applies standard layer normalization but ensures invertibility
    by storing the normalization parameters for exact inverse.
    """
    
    def __init__(self, embed_dim: int, eps: float = 1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward layer normalization.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            
        Returns:
            y: Normalized tensor
            log_det: Log determinant of transformation
        """
        # Compute normalization statistics
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        
        # Apply normalization
        x_normalized = (x - mean) / std
        y = self.weight * x_normalized + self.bias
        
        # Compute log determinant
        # For layer norm: log|det(J)| = sum(log(weight/std))
        log_det = torch.sum(torch.log(torch.abs(self.weight) / std), dim=-1)
        
        return y, log_det
    
    def inverse(self, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Inverse layer normalization.
        
        Args:
            y: Normalized tensor
            mean: Original mean (stored from forward pass)
            std: Original std (stored from forward pass)
            
        Returns:
            x: Original tensor
        """
        # Reverse affine transformation
        x_normalized = (y - self.bias) / self.weight
        
        # Reverse normalization
        x = x_normalized * std + mean
        
        return x


class CouplingFunction(nn.Module):
    """
    Neural network function for use in invertible residual connections.
    Designed to be used as F or G functions in RevNet coupling.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build network
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            # Activation
            if activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "swish":
                layers.append(nn.SiLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize to output zeros for identity initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for identity initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize final layer to zeros
        if len(self.network) > 0 and isinstance(self.network[-1], nn.Linear):
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through coupling function."""
        return self.network(x)


class InvertibleFeedForward(nn.Module):
    """
    Invertible feed-forward network using coupling layers.
    Replaces standard transformer feed-forward with bijective version.
    """
    
    def __init__(self, config: InvertibleConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        
        # Determine split dimensions
        if config.split_dim is None:
            self.split_dim = config.embed_dim // 2
        else:
            self.split_dim = config.split_dim
        
        self.remaining_dim = config.embed_dim - self.split_dim
        
        # Create coupling functions
        hidden_dim = config.embed_dim * 4  # Standard transformer scaling
        
        self.F_function = CouplingFunction(
            input_dim=self.remaining_dim,
            output_dim=self.split_dim,
            hidden_dim=hidden_dim,
            activation=config.activation,
            dropout=config.dropout
        )
        
        self.G_function = CouplingFunction(
            input_dim=self.split_dim,
            output_dim=self.remaining_dim,
            hidden_dim=hidden_dim,
            activation=config.activation,
            dropout=config.dropout
        )
        
        # Create invertible residual connection
        self.invertible_residual = InvertibleResidualConnection(
            config, self.F_function, self.G_function
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through invertible feed-forward."""
        return self.invertible_residual.forward(x)
    
    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass through invertible feed-forward."""
        return self.invertible_residual.inverse(y)


def create_invertible_config(
    embed_dim: int,
    coupling_type: str = "additive",
    activation: str = "gelu",
    dropout: float = 0.1,
    **kwargs
) -> InvertibleConfig:
    """Create invertible configuration with defaults."""
    return InvertibleConfig(
        embed_dim=embed_dim,
        coupling_type=coupling_type,
        activation=activation,
        dropout=dropout,
        **kwargs
    )


def create_coupling_functions(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 512,
    **kwargs
) -> Tuple[CouplingFunction, CouplingFunction]:
    """Create a pair of coupling functions for RevNet residuals."""
    F_function = CouplingFunction(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        **kwargs
    )
    
    G_function = CouplingFunction(
        input_dim=output_dim,
        output_dim=input_dim,
        hidden_dim=hidden_dim,
        **kwargs
    )
    
    return F_function, G_function
