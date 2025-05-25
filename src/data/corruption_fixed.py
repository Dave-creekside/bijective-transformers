"""
Device-aware text corruption methods for discrete diffusion.
Implements various noise processes: masking, substitution, deletion.
FIXED: All components are device-aware to prevent device mismatch errors.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math


@dataclass
class CorruptionConfig:
    """Configuration for text corruption parameters."""
    mask_prob: float = 0.15
    substitute_prob: float = 0.1
    delete_prob: float = 0.05
    mask_token_id: int = 50256  # GPT-2 mask token
    vocab_size: int = 50257
    random_token_prob: float = 0.1
    keep_original_prob: float = 0.1
    max_deletions: float = 0.2


class NoiseScheduler:
    """
    Device-aware noise scheduling for discrete diffusion process.
    Controls how corruption probability changes over timesteps.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: str = "linear",
        min_noise: float = 0.01,
        max_noise: float = 0.99,
        device: Optional[torch.device] = None
    ):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.device = device or torch.device('cpu')
        
        self.noise_schedule = self._create_schedule()
    
    def _create_schedule(self) -> torch.Tensor:
        """Create noise schedule based on schedule type."""
        if self.schedule_type == "linear":
            schedule = torch.linspace(self.min_noise, self.max_noise, self.num_timesteps)
        elif self.schedule_type == "cosine":
            # Cosine schedule from DDPM
            steps = torch.arange(self.num_timesteps + 1, dtype=torch.float32)
            alpha_bar = torch.cos(((steps / self.num_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            schedule = torch.clamp(betas, self.min_noise, self.max_noise)
        elif self.schedule_type == "sqrt":
            # Square root schedule
            steps = torch.arange(self.num_timesteps, dtype=torch.float32)
            schedule = self.min_noise + (self.max_noise - self.min_noise) * torch.sqrt(steps / (self.num_timesteps - 1))
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return schedule.to(self.device)
    
    def to(self, device: torch.device) -> 'NoiseScheduler':
        """Move scheduler to specified device."""
        self.device = device
        self.noise_schedule = self.noise_schedule.to(device)
        return self
    
    def get_noise_level(self, timestep: Union[int, torch.Tensor]) -> torch.Tensor:
        """Get noise level for given timestep(s)."""
        if isinstance(timestep, int):
            return self.noise_schedule[timestep]
        else:
            # Ensure timestep is on same device as noise_schedule
            timestep = timestep.to(self.noise_schedule.device)
            return self.noise_schedule[timestep]
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for training."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)


class TextCorruptor:
    """
    Device-aware text corruption for discrete diffusion training.
    Applies various noise processes to clean text.
    """
    
    def __init__(self, config: CorruptionConfig, noise_scheduler: NoiseScheduler):
        self.config = config
        self.noise_scheduler = noise_scheduler
    
    def to(self, device: torch.device) -> 'TextCorruptor':
        """Move corruptor to specified device."""
        self.noise_scheduler.to(device)
        return self
    
    def corrupt_sequence(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Corrupt input sequence based on timesteps.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            timesteps: Diffusion timesteps [batch_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            corrupted_ids: Corrupted token IDs
            corruption_mask: Mask indicating which tokens were corrupted
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Ensure noise scheduler is on same device
        if self.noise_scheduler.device != device:
            self.noise_scheduler.to(device)
        
        # Get noise levels for each timestep
        noise_levels = self.noise_scheduler.get_noise_level(timesteps)  # [batch_size]
        
        # Create corruption mask based on attention mask
        if attention_mask is None:
            valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            valid_mask = attention_mask.bool()
        
        # Initialize corrupted sequence
        corrupted_ids = input_ids.clone()
        corruption_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        for i in range(batch_size):
            noise_level = noise_levels[i].item()
            seq_mask = valid_mask[i]
            
            # Apply corruption to this sequence
            corrupted_seq, corrupt_mask = self._corrupt_single_sequence(
                input_ids[i], noise_level, seq_mask
            )
            
            corrupted_ids[i] = corrupted_seq
            corruption_mask[i] = corrupt_mask
        
        return corrupted_ids, corruption_mask
    
    def _corrupt_single_sequence(
        self,
        input_ids: torch.Tensor,
        noise_level: float,
        valid_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Corrupt a single sequence based on noise level."""
        seq_len = input_ids.shape[0]
        device = input_ids.device
        
        # Scale corruption probabilities by noise level
        mask_prob = self.config.mask_prob * noise_level
        substitute_prob = self.config.substitute_prob * noise_level
        delete_prob = self.config.delete_prob * noise_level
        
        # Only corrupt valid tokens
        valid_indices = torch.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return input_ids, torch.zeros_like(input_ids, dtype=torch.bool)
        
        corrupted_ids = input_ids.clone()
        corruption_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # Apply masking corruption
        if mask_prob > 0:
            mask_indices = self._sample_corruption_indices(valid_indices, mask_prob, device)
            corrupted_ids, mask_applied = self._apply_masking(
                corrupted_ids, mask_indices
            )
            corruption_mask |= mask_applied
        
        # Apply substitution corruption
        if substitute_prob > 0:
            # Don't substitute already masked tokens
            available_indices = valid_indices[~corruption_mask[valid_indices]]
            if len(available_indices) > 0:
                sub_indices = self._sample_corruption_indices(available_indices, substitute_prob, device)
                corrupted_ids, sub_applied = self._apply_substitution(
                    corrupted_ids, sub_indices
                )
                corruption_mask |= sub_applied
        
        # Apply deletion corruption
        if delete_prob > 0:
            # Don't delete already corrupted tokens
            available_indices = valid_indices[~corruption_mask[valid_indices]]
            if len(available_indices) > 0:
                max_deletions = int(len(valid_indices) * self.config.max_deletions)
                del_indices = self._sample_corruption_indices(
                    available_indices, delete_prob, device, max_count=max_deletions
                )
                corrupted_ids, del_applied = self._apply_deletion(
                    corrupted_ids, del_indices
                )
                corruption_mask |= del_applied
        
        return corrupted_ids, corruption_mask
    
    def _sample_corruption_indices(
        self,
        valid_indices: torch.Tensor,
        prob: float,
        device: torch.device,
        max_count: Optional[int] = None
    ) -> torch.Tensor:
        """Sample indices for corruption based on probability."""
        if len(valid_indices) == 0:
            return torch.tensor([], dtype=torch.long, device=device)
        
        # Sample corruption mask - ensure on correct device
        corruption_probs = torch.full((len(valid_indices),), prob, device=device)
        corruption_decisions = torch.bernoulli(corruption_probs).bool()
        
        # Apply max count limit if specified
        if max_count is not None and corruption_decisions.sum() > max_count:
            # Randomly select max_count indices
            corrupt_indices = torch.where(corruption_decisions)[0]
            selected = torch.randperm(len(corrupt_indices), device=device)[:max_count]
            corruption_decisions.fill_(False)
            corruption_decisions[corrupt_indices[selected]] = True
        
        return valid_indices[corruption_decisions]
    
    def _apply_masking(
        self,
        input_ids: torch.Tensor,
        mask_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply masking corruption."""
        corrupted_ids = input_ids.clone()
        mask_applied = torch.zeros_like(input_ids, dtype=torch.bool)
        device = input_ids.device
        
        if len(mask_indices) == 0:
            return corrupted_ids, mask_applied
        
        # For each token to mask, decide what to do
        for idx in mask_indices:
            rand = torch.rand(1, device=device).item()  # Ensure random tensor on correct device
            
            if rand < self.config.keep_original_prob:
                # Keep original token
                pass
            elif rand < self.config.keep_original_prob + self.config.random_token_prob:
                # Replace with random token
                random_token = torch.randint(0, self.config.vocab_size, (1,), device=device)
                corrupted_ids[idx] = random_token
                mask_applied[idx] = True
            else:
                # Replace with mask token
                corrupted_ids[idx] = self.config.mask_token_id
                mask_applied[idx] = True
        
        return corrupted_ids, mask_applied
    
    def _apply_substitution(
        self,
        input_ids: torch.Tensor,
        sub_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply substitution corruption."""
        corrupted_ids = input_ids.clone()
        sub_applied = torch.zeros_like(input_ids, dtype=torch.bool)
        device = input_ids.device
        
        if len(sub_indices) == 0:
            return corrupted_ids, sub_applied
        
        # Replace with random tokens from vocabulary - ensure on correct device
        random_tokens = torch.randint(
            0, self.config.vocab_size, (len(sub_indices),), device=device
        )
        corrupted_ids[sub_indices] = random_tokens
        sub_applied[sub_indices] = True
        
        return corrupted_ids, sub_applied
    
    def _apply_deletion(
        self,
        input_ids: torch.Tensor,
        del_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply deletion corruption (mark as special deletion token)."""
        corrupted_ids = input_ids.clone()
        del_applied = torch.zeros_like(input_ids, dtype=torch.bool)
        
        if len(del_indices) == 0:
            return corrupted_ids, del_applied
        
        # For simplicity, we'll mark deleted tokens with a special token
        # In practice, you might want to actually remove them and handle variable lengths
        deletion_token_id = min(self.config.vocab_size - 1, self.config.mask_token_id)  # Use mask token for deletion
        corrupted_ids[del_indices] = deletion_token_id
        del_applied[del_indices] = True
        
        return corrupted_ids, del_applied


def ensure_device_compatibility(model, corruptor: TextCorruptor) -> torch.device:
    """
    Ensure all components are on same device as model.
    
    Args:
        model: PyTorch model
        corruptor: TextCorruptor instance
        
    Returns:
        Device that everything is now on
    """
    device = next(model.parameters()).device
    corruptor.to(device)
    return device


def create_corruption_config(
    mask_prob: float = 0.15,
    substitute_prob: float = 0.1,
    delete_prob: float = 0.05,
    vocab_size: int = 50257,
    **kwargs
) -> CorruptionConfig:
    """Create corruption configuration with defaults."""
    return CorruptionConfig(
        mask_prob=mask_prob,
        substitute_prob=substitute_prob,
        delete_prob=delete_prob,
        vocab_size=vocab_size,
        **kwargs
    )


def create_noise_scheduler(
    num_timesteps: int = 1000,
    schedule_type: str = "linear",
    device: Optional[torch.device] = None,
    **kwargs
) -> NoiseScheduler:
    """Create device-aware noise scheduler with defaults."""
    return NoiseScheduler(
        num_timesteps=num_timesteps,
        schedule_type=schedule_type,
        device=device,
        **kwargs
    )


def create_device_aware_corruptor(
    config: CorruptionConfig,
    noise_scheduler: NoiseScheduler,
    device: Optional[torch.device] = None
) -> TextCorruptor:
    """
    Create a device-aware text corruptor.
    
    Args:
        config: Corruption configuration
        noise_scheduler: Noise scheduler
        device: Target device (if None, uses CPU)
        
    Returns:
        Device-aware TextCorruptor
    """
    if device is not None:
        noise_scheduler.to(device)
    
    corruptor = TextCorruptor(config, noise_scheduler)
    
    if device is not None:
        corruptor.to(device)
    
    return corruptor
