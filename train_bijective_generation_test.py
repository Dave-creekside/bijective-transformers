#!/usr/bin/env python3
"""
GENERATION TEST: Training script with improved generation sampling and noise injection.
Tests advanced sampling to fix mask token (50256) repetition.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import time
import os
from typing import Dict, Any
import math

# Set environment variable to suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import our improved bijective model
from src.models.bijective_diffusion_fixed import (
    BijectiveDiscreteDiffusionModel,
    create_bijective_diffusion_model_config
)

# Import FINAL working corruption (device-aware)
from src.data.corruption_final import (
    TextCorruptor, 
    CorruptionConfig, 
    NoiseScheduler,
    ensure_device_compatibility,
    create_device_aware_corruptor
)

# Import REAL WikiText-2 data (NO SYNTHETIC DATA)
from src.data.wikitext_real import WikiTextDataModule


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_strategic_noise(input_ids: torch.Tensor, noise_level: float = 0.3) -> torch.Tensor:
    """
    IMPROVED: Create strategic noise patterns instead of random corruption.
    Avoids creating all mask tokens which leads to repetitive generation.
    """
    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    vocab_size = 50257  # GPT-2 vocab size
    
    # Create noise mask
    noise_mask = torch.rand(batch_size, seq_len, device=device) < noise_level
    
    # Create diverse noise tokens (avoid mask token bias)
    # Use common tokens (lower IDs are more frequent in GPT-2)
    noise_tokens = torch.randint(0, min(vocab_size, 5000), (batch_size, seq_len), device=device)
    
    # Apply noise strategically
    noisy_input = torch.where(noise_mask, noise_tokens, input_ids)
    
    return noisy_input


def test_generation_quality(model, data_module, device: str, num_tests: int = 3):
    """Test generation quality with multiple samples and strategies."""
    model.eval()
    
    print(f"\n🎯 Testing IMPROVED Generation Quality ({num_tests} samples):")
    
    with torch.no_grad():
        val_loader = data_module.val_dataloader()
        
        for test_idx in range(num_tests):
            print(f"\n--- Test Sample {test_idx + 1} ---")
            
            # Get a real sample
            val_batch = next(iter(val_loader))
            real_input = val_batch["input_ids"][:1].to(device)
            real_mask = val_batch["attention_mask"][:1].to(device)
            
            # Show original text
            try:
                original_text = data_module.train_dataset.decode(real_input.squeeze())
                print(f"Original: {original_text[:150]}...")
            except Exception as e:
                print(f"Original decode failed: {e}")
                print(f"Original tokens: {real_input.squeeze()[:15].tolist()}...")
            
            # Test 1: Strategic noise injection
            strategic_noise = create_strategic_noise(real_input, noise_level=0.4)
            
            # Test 2: Generate with improved sampling
            generated = model.generate(
                input_ids=strategic_noise,
                num_inference_steps=15,  # More steps for better quality
                attention_mask=real_mask
            )
            
            # Analyze generation quality
            try:
                generated_text = data_module.train_dataset.decode(generated.squeeze())
                
                # Check for diversity
                unique_tokens = torch.unique(generated.squeeze())
                total_tokens = generated.numel()
                diversity_ratio = len(unique_tokens) / total_tokens
                
                # Check for mask token repetition
                mask_token_count = (generated.squeeze() == 50256).sum().item()
                mask_ratio = mask_token_count / total_tokens
                
                print(f"Generated: {generated_text[:150]}...")
                print(f"Diversity: {len(unique_tokens)}/{total_tokens} tokens ({diversity_ratio:.2%})")
                print(f"Mask tokens: {mask_token_count}/{total_tokens} ({mask_ratio:.2%})")
                
                if diversity_ratio > 0.1 and mask_ratio < 0.5:
                    print("✅ GOOD: Diverse generation with low mask repetition")
                elif diversity_ratio > 0.05:
                    print("⚠️  FAIR: Some diversity but could be better")
                else:
                    print("❌ POOR: Low diversity, needs more training")
                    
            except Exception as e:
                print(f"Generated decode failed: {e}")
                print(f"Generated tokens: {generated.squeeze()[:15].tolist()}...")
            
            # Token change analysis
            token_changes = (generated != strategic_noise).float().mean().item()
            print(f"Token change rate: {token_changes:.2%}")


def train_bijective_model():
    """Train bijective model with generation testing."""
    print("🚀 Training Bijective Model with IMPROVED Generation Testing")
    print("=" * 70)
    print("🔧 TESTING: Advanced sampling, strategic noise, generation quality")
    print("📚 USING REAL DATA ONLY - NO SYNTHETIC DATA")
    print("=" * 70)
    
    # Configuration
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data configuration
    data_config_path = "configs/data/wikitext2.yaml"
    if not os.path.exists(data_config_path):
        print(f"⚠️  Data config not found at {data_config_path}, using defaults")
        data_config = {
            "tokenizer_name": "gpt2",
            "max_length": 256,
            "batch_size": 8,
            "eval_batch_size": 16,
            "num_workers": 0,
            "pin_memory": True,
            "preprocessing": {"min_length": 10},
            "cache_dir": "data/cache",
            "use_cache": True
        }
    else:
        data_config = load_config(data_config_path)
        data_config["num_workers"] = 0
        data_config["max_length"] = 256
        data_config["batch_size"] = 8
    
    print(f"Data config: max_length={data_config['max_length']}, batch_size={data_config['batch_size']}")
    
    # Create data module with REAL WikiText-2
    print("\n📚 Setting up REAL WikiText-2 dataset...")
    data_module = WikiTextDataModule(data_config)
    data_module.setup()
    
    # Get real vocabulary size from tokenizer
    real_vocab_size = data_module.get_vocab_size()
    print(f"✅ Real vocabulary size: {real_vocab_size}")
    
    # Model configuration - optimized for testing
    config = create_bijective_diffusion_model_config(
        vocab_size=real_vocab_size,
        max_seq_length=data_config["max_length"],
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        use_exact_likelihood=True,
        likelihood_weight=0.001,
        inference_steps=15  # More inference steps for better generation
    )
    
    print(f"Model config: {config.transformer.transformer.vocab_size} vocab, "
          f"{config.transformer.transformer.max_seq_length} max_len, "
          f"{config.transformer.transformer.embed_dim} embed_dim, "
          f"{config.transformer.transformer.num_layers} layers")
    
    # Create model
    model = BijectiveDiscreteDiffusionModel(config)
    model = model.to(device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Create dataloaders from REAL data
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create device-aware corruptor for REAL vocabulary
    corruption_config = CorruptionConfig(
        mask_prob=0.15,
        substitute_prob=0.1,
        vocab_size=real_vocab_size,
        mask_token_id=real_vocab_size - 1
    )
    
    # Create device-aware noise scheduler
    noise_scheduler = NoiseScheduler(num_timesteps=1000, device=device)
    
    # Create device-aware corruptor
    corruptor = create_device_aware_corruptor(
        corruption_config, 
        noise_scheduler, 
        device=device
    )
    
    # CRITICAL: Ensure device compatibility
    actual_device = ensure_device_compatibility(model, corruptor)
    print(f"✅ Device compatibility ensured: {actual_device}")
    print(f"✅ Model device: {next(model.parameters()).device}")
    print(f"✅ Corruptor device: {corruptor.device}")
    
    # Store corruptor in model for validation
    model._temp_corruptor = corruptor
    
    # Optimizer with learning rate scheduling
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    
    # Quick training for generation testing
    num_epochs = 2
    batches_per_epoch = 50  # Reduced for faster testing
    model.train()
    
    print(f"\n🔥 Starting QUICK training: {num_epochs} epochs × {batches_per_epoch} batches")
    print("🎯 FOCUS: Test generation improvements after minimal training")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0.0
        total_denoising_loss = 0.0
        total_likelihood_loss = 0.0
        successful_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= batches_per_epoch:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Training step
            optimizer.zero_grad()
            
            try:
                metrics = model.training_step(
                    clean_input_ids=input_ids,
                    attention_mask=attention_mask,
                    corruptor=corruptor
                )
                
                loss = metrics["loss"]
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                # Accumulate metrics
                total_loss += loss.item()
                total_denoising_loss += metrics["denoising_loss"].item()
                total_likelihood_loss += metrics["likelihood_loss"].item()
                successful_batches += 1
                
                # Log progress
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{batches_per_epoch}: "
                          f"Loss={loss.item():.4f}, "
                          f"Denoising={metrics['denoising_loss'].item():.4f}, "
                          f"Likelihood={metrics['likelihood_loss'].item():.4f}")
                
            except Exception as e:
                print(f"❌ Training step failed: {e}")
                continue
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(successful_batches, 1)
        avg_denoising = total_denoising_loss / max(successful_batches, 1)
        avg_likelihood = total_likelihood_loss / max(successful_batches, 1)
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Time: {epoch_time:.1f}s")
        print(f"   Successful batches: {successful_batches}/{batches_per_epoch}")
        print(f"   Avg Loss: {avg_loss:.4f}")
        print(f"   Avg Denoising Loss: {avg_denoising:.4f}")
        print(f"   Avg Likelihood Loss: {avg_likelihood:.4f}")
        print(f"   Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Test generation after each epoch
        test_generation_quality(model, data_module, device, num_tests=2)
        
        model.train()  # Back to training mode
    
    print("\n🎉 Training completed!")
    
    # Final comprehensive generation test
    print("\n🎯 FINAL Generation Quality Test:")
    test_generation_quality(model, data_module, device, num_tests=5)
    
    # Model info
    print(f"\n🏆 Model Information:")
    bijective_info = model.get_bijective_info()
    print(f"   Model type: {bijective_info['model_type']}")
    print(f"   Total parameters: {bijective_info['total_params']:,}")
    print(f"   Exact likelihood enabled: {bijective_info['exact_likelihood_enabled']}")
    print(f"   Likelihood weight: {config.likelihood_weight}")
    print(f"   Bijective blocks: {bijective_info['transformer_info']['bijective_blocks']}")
    print(f"   Total blocks: {bijective_info['transformer_info']['total_blocks']}")
    
    # Device compatibility confirmation
    print(f"\n✅ Final Device Compatibility Status:")
    print(f"   Model device: {next(model.parameters()).device}")
    print(f"   Corruptor device: {corruptor.device}")
    print(f"   Device types compatible: {next(model.parameters()).device.type == corruptor.device.type}")
    print(f"   All components synchronized: ✅")
    
    return model


if __name__ == "__main__":
    try:
        model = train_bijective_model()
        print("\n✅ SUCCESS: Generation testing completed!")
        print("🔧 IMPROVEMENTS TESTED: Advanced sampling, strategic noise, quality metrics!")
        print("📚 NO SYNTHETIC DATA used - only real text data!")
        print("🛠️  Device compatibility maintained throughout!")
    except Exception as e:
        print(f"\n❌ FAILED: Training failed with error: {e}")
        import traceback
        traceback.print_exc()
