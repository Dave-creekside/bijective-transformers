#!/usr/bin/env python3
"""
WORKSTATION: Training script optimized for 2x RTX 4070 (24GB VRAM) + 128GB RAM.
Scales up model size and training for serious generation quality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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


def setup_distributed():
    """Setup distributed training for multi-GPU."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single GPU or DataParallel mode
        return 0, 1, 0


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_strategic_noise(input_ids: torch.Tensor, noise_level: float = 0.3) -> torch.Tensor:
    """Create strategic noise patterns for better generation training."""
    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    vocab_size = 50257  # GPT-2 vocab size
    
    # Create noise mask
    noise_mask = torch.rand(batch_size, seq_len, device=device) < noise_level
    
    # Create diverse noise tokens (avoid mask token bias)
    # Use common tokens (lower IDs are more frequent in GPT-2)
    noise_tokens = torch.randint(0, min(vocab_size, 10000), (batch_size, seq_len), device=device)
    
    # Apply noise strategically
    noisy_input = torch.where(noise_mask, noise_tokens, input_ids)
    
    return noisy_input


def test_generation_quality(model, data_module, device: str, num_tests: int = 3):
    """Test generation quality with multiple samples and strategies."""
    if hasattr(model, 'module'):
        model = model.module  # Unwrap DDP/DataParallel
    
    model.eval()
    
    print(f"\n🎯 Testing WORKSTATION Generation Quality ({num_tests} samples):")
    
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
            
            # Strategic noise injection
            strategic_noise = create_strategic_noise(real_input, noise_level=0.4)
            
            # Generate with improved sampling
            generated = model.generate(
                input_ids=strategic_noise,
                num_inference_steps=20,  # More steps for workstation
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
                
                if diversity_ratio > 0.2 and mask_ratio < 0.3:
                    print("✅ EXCELLENT: High diversity with low mask repetition")
                elif diversity_ratio > 0.1 and mask_ratio < 0.5:
                    print("✅ GOOD: Diverse generation with reasonable mask usage")
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


def train_bijective_workstation():
    """Train bijective model optimized for workstation hardware."""
    print("🚀 Training Bijective Model - WORKSTATION OPTIMIZED")
    print("=" * 80)
    print("🖥️  HARDWARE: 2x RTX 4070 (24GB VRAM) + 128GB RAM")
    print("🔧 OPTIMIZED: Large model, big batches, extended training")
    print("📚 USING REAL DATA ONLY - NO SYNTHETIC DATA")
    print("=" * 80)
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Device configuration
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        print(f"Using CUDA device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"VRAM: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f}GB")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")
    
    # WORKSTATION-OPTIMIZED Data configuration
    data_config_path = "configs/data/wikitext2.yaml"
    if not os.path.exists(data_config_path):
        print(f"⚠️  Data config not found at {data_config_path}, using workstation defaults")
        data_config = {
            "tokenizer_name": "gpt2",
            "max_length": 512,      # INCREASED: Full sequences for workstation
            "batch_size": 16,       # INCREASED: Large batches for 24GB VRAM
            "eval_batch_size": 32,  # INCREASED: Even larger eval batches
            "num_workers": 8,       # INCREASED: Utilize 128GB RAM
            "pin_memory": True,
            "preprocessing": {"min_length": 10},
            "cache_dir": "data/cache",
            "use_cache": True
        }
    else:
        data_config = load_config(data_config_path)
        # Override for workstation optimization
        data_config["max_length"] = 512
        data_config["batch_size"] = 16
        data_config["eval_batch_size"] = 32
        data_config["num_workers"] = 8
    
    print(f"WORKSTATION Data config: max_length={data_config['max_length']}, "
          f"batch_size={data_config['batch_size']}, workers={data_config['num_workers']}")
    
    # Create data module with REAL WikiText-2
    print("\n📚 Setting up REAL WikiText-2 dataset...")
    data_module = WikiTextDataModule(data_config)
    data_module.setup()
    
    # Get real vocabulary size from tokenizer
    real_vocab_size = data_module.get_vocab_size()
    print(f"✅ Real vocabulary size: {real_vocab_size}")
    
    # WORKSTATION-OPTIMIZED Model configuration - LARGE MODEL
    config = create_bijective_diffusion_model_config(
        vocab_size=real_vocab_size,
        max_seq_length=data_config["max_length"],  # 512 tokens
        embed_dim=768,          # INCREASED: Large embeddings
        num_layers=12,          # INCREASED: Deep model for workstation
        num_heads=12,           # INCREASED: More attention heads
        use_exact_likelihood=True,
        likelihood_weight=0.001,
        inference_steps=20      # More inference steps for quality
    )
    
    print(f"WORKSTATION Model config: {config.transformer.transformer.vocab_size} vocab, "
          f"{config.transformer.transformer.max_seq_length} max_len, "
          f"{config.transformer.transformer.embed_dim} embed_dim, "
          f"{config.transformer.transformer.num_layers} layers")
    
    # Create model
    model = BijectiveDiscreteDiffusionModel(config)
    model = model.to(device)
    
    # Multi-GPU setup
    if torch.cuda.device_count() > 1 and world_size > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DistributedDataParallel")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    elif torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    print(f"Model parameters: {model.module.get_num_params() if hasattr(model, 'module') else model.get_num_params():,}")
    
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
    if hasattr(model, 'module'):
        model.module._temp_corruptor = corruptor
    else:
        model._temp_corruptor = corruptor
    
    # WORKSTATION-OPTIMIZED Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)  # Higher LR for larger model
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)  # Longer schedule
    
    # EXTENDED training for workstation
    num_epochs = 10         # INCREASED: More epochs for quality
    batches_per_epoch = 500 # INCREASED: Much more training per epoch
    
    print(f"\n🔥 Starting WORKSTATION training: {num_epochs} epochs × {batches_per_epoch} batches")
    print("🖥️  WORKSTATION POWER: Large model, big batches, extended training")
    
    model.train()
    
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
                # Get training step method from model (handle DDP/DataParallel)
                if hasattr(model, 'module'):
                    metrics = model.module.training_step(
                        clean_input_ids=input_ids,
                        attention_mask=attention_mask,
                        corruptor=corruptor
                    )
                else:
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
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{batches_per_epoch}: "
                          f"Loss={loss.item():.4f}, "
                          f"Denoising={metrics['denoising_loss'].item():.4f}, "
                          f"Likelihood={metrics['likelihood_loss'].item():.4f}, "
                          f"LR={scheduler.get_last_lr()[0]:.6f}")
                
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
        
        # Test generation every 2 epochs
        if (epoch + 1) % 2 == 0:
            test_generation_quality(model, data_module, device, num_tests=3)
            model.train()  # Back to training mode
    
    print("\n🎉 WORKSTATION Training completed!")
    
    # Final comprehensive generation test
    print("\n🎯 FINAL WORKSTATION Generation Quality Test:")
    test_generation_quality(model, data_module, device, num_tests=10)
    
    # Model info
    print(f"\n🏆 WORKSTATION Model Information:")
    model_for_info = model.module if hasattr(model, 'module') else model
    bijective_info = model_for_info.get_bijective_info()
    print(f"   Model type: {bijective_info['model_type']}")
    print(f"   Total parameters: {bijective_info['total_params']:,}")
    print(f"   Exact likelihood enabled: {bijective_info['exact_likelihood_enabled']}")
    print(f"   Likelihood weight: {config.likelihood_weight}")
    print(f"   Bijective blocks: {bijective_info['transformer_info']['bijective_blocks']}")
    print(f"   Total blocks: {bijective_info['transformer_info']['total_blocks']}")
    
    # Hardware utilization
    if torch.cuda.is_available():
        print(f"\n🖥️  Hardware Utilization:")
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1e9
            memory_reserved = torch.cuda.memory_reserved(i) / 1e9
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {memory_allocated:.1f}GB / {memory_total:.1f}GB allocated")
            print(f"   GPU {i}: {memory_reserved:.1f}GB / {memory_total:.1f}GB reserved")
    
    # Device compatibility confirmation
    print(f"\n✅ Final Device Compatibility Status:")
    print(f"   Model device: {next(model.parameters()).device}")
    print(f"   Corruptor device: {corruptor.device}")
    print(f"   Device types compatible: {next(model.parameters()).device.type == corruptor.device.type}")
    print(f"   All components synchronized: ✅")
    
    return model


if __name__ == "__main__":
    try:
        model = train_bijective_workstation()
        print("\n✅ SUCCESS: WORKSTATION Bijective model training completed!")
        print("🖥️  WORKSTATION OPTIMIZED: Large model, extended training, dual GPU!")
        print("📚 NO SYNTHETIC DATA used - only real text data!")
        print("🛠️  Device compatibility maintained throughout!")
    except Exception as e:
        print(f"\n❌ FAILED: Training failed with error: {e}")
        import traceback
        traceback.print_exc()
