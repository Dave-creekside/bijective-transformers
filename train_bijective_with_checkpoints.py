#!/usr/bin/env python3
"""
CHECKPOINT-ENABLED: Training script with comprehensive save/load functionality.
Supports resuming training, automatic checkpointing, and model export.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import time
import os
import argparse
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

# Import checkpoint manager
from src.utils.checkpoint import create_checkpoint_manager


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_perplexity(model, dataloader, device: str = "cpu") -> float:
    """Compute perplexity on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 5:  # Limit for speed
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            try:
                # Use training_step for consistency
                metrics = model.training_step(
                    clean_input_ids=input_ids,
                    attention_mask=attention_mask,
                    corruptor=model._temp_corruptor
                )
                
                loss = metrics["denoising_loss"]
                valid_tokens = attention_mask.sum().item()
                
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens
                
            except Exception as e:
                print(f"Validation batch {batch_idx} failed: {e}")
                continue
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 10))
    
    return perplexity


def test_generation_quality(model, data_module, device: str, num_tests: int = 3):
    """Test generation quality with multiple samples."""
    model.eval()
    
    print(f"\n🎯 Testing Generation Quality ({num_tests} samples):")
    
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
                print(f"Original: {original_text[:100]}...")
            except Exception as e:
                print(f"Original tokens: {real_input.squeeze()[:10].tolist()}...")
            
            # Generate
            generated = model.generate(
                input_ids=real_input,
                num_inference_steps=10,
                attention_mask=real_mask
            )
            
            # Analyze generation quality
            try:
                generated_text = data_module.train_dataset.decode(generated.squeeze())
                unique_tokens = torch.unique(generated.squeeze())
                total_tokens = generated.numel()
                diversity_ratio = len(unique_tokens) / total_tokens
                
                mask_token_count = (generated.squeeze() == 50256).sum().item()
                mask_ratio = mask_token_count / total_tokens
                
                print(f"Generated: {generated_text[:100]}...")
                print(f"Diversity: {len(unique_tokens)}/{total_tokens} ({diversity_ratio:.2%})")
                print(f"Mask tokens: {mask_token_count}/{total_tokens} ({mask_ratio:.2%})")
                
                if diversity_ratio > 0.1 and mask_ratio < 0.5:
                    print("✅ GOOD: Diverse generation")
                elif diversity_ratio > 0.05:
                    print("⚠️  FAIR: Some diversity")
                else:
                    print("❌ POOR: Low diversity, needs more training")
                    
            except Exception as e:
                print(f"Generated tokens: {generated.squeeze()[:10].tolist()}...")
            
            token_changes = (generated != real_input).float().mean().item()
            print(f"Token change rate: {token_changes:.2%}")


def train_bijective_model(
    resume_from: str = None,
    export_final: bool = True,
    checkpoint_every: int = 2,
    max_epochs: int = 10
):
    """Train bijective model with comprehensive checkpoint management."""
    print("🚀 Training Bijective Model with CHECKPOINT MANAGEMENT")
    print("=" * 70)
    print("💾 FEATURES: Auto-save, resume training, model export")
    print("📚 USING REAL DATA ONLY - NO SYNTHETIC DATA")
    print("=" * 70)
    
    # Configuration
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create checkpoint manager
    checkpoint_manager = create_checkpoint_manager(
        checkpoint_dir="models/checkpoints",
        export_dir="models/exports",
        max_checkpoints=5,
        save_every_n_epochs=checkpoint_every
    )
    
    # Load data configuration
    data_config_path = "configs/data/wikitext2.yaml"
    if not os.path.exists(data_config_path):
        print(f"⚠️  Data config not found, using defaults")
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
    
    # Create data module
    print("\n📚 Setting up REAL WikiText-2 dataset...")
    data_module = WikiTextDataModule(data_config)
    data_module.setup()
    
    real_vocab_size = data_module.get_vocab_size()
    print(f"✅ Real vocabulary size: {real_vocab_size}")
    
    # Model configuration
    config = create_bijective_diffusion_model_config(
        vocab_size=real_vocab_size,
        max_seq_length=data_config["max_length"],
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        use_exact_likelihood=True,
        likelihood_weight=0.001,
        inference_steps=10
    )
    
    print(f"Model config: {config.transformer.transformer.vocab_size} vocab, "
          f"{config.transformer.transformer.max_seq_length} max_len, "
          f"{config.transformer.transformer.embed_dim} embed_dim, "
          f"{config.transformer.transformer.num_layers} layers")
    
    # Create model
    model = BijectiveDiscreteDiffusionModel(config)
    model = model.to(device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # Initialize training state
    start_epoch = 0
    best_loss = float('inf')
    
    # Resume from checkpoint if specified
    if resume_from:
        if resume_from == "latest":
            resume_from = checkpoint_manager.get_latest_checkpoint()
        elif resume_from == "best":
            resume_from = checkpoint_manager.get_best_checkpoint()
        
        if resume_from and os.path.exists(resume_from):
            print(f"\n📂 Resuming training from checkpoint...")
            start_epoch, prev_loss, loaded_config = checkpoint_manager.load_checkpoint(
                model, optimizer, scheduler, resume_from, device
            )
            start_epoch += 1  # Start from next epoch
            print(f"✅ Resumed from epoch {start_epoch-1}, loss {prev_loss:.4f}")
        else:
            print(f"⚠️  Checkpoint not found: {resume_from}, starting fresh")
    
    # Create dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create corruptor
    corruption_config = CorruptionConfig(
        mask_prob=0.15,
        substitute_prob=0.1,
        vocab_size=real_vocab_size,
        mask_token_id=real_vocab_size - 1
    )
    
    noise_scheduler = NoiseScheduler(num_timesteps=1000, device=device)
    corruptor = create_device_aware_corruptor(corruption_config, noise_scheduler, device=device)
    
    # Ensure device compatibility
    actual_device = ensure_device_compatibility(model, corruptor)
    print(f"✅ Device compatibility ensured: {actual_device}")
    
    # Store corruptor in model
    model._temp_corruptor = corruptor
    
    # Training loop
    batches_per_epoch = 100
    model.train()
    
    print(f"\n🔥 Starting training: epochs {start_epoch} to {max_epochs}")
    print(f"💾 Auto-save every {checkpoint_every} epochs")
    
    for epoch in range(start_epoch, max_epochs):
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
            
            optimizer.zero_grad()
            
            try:
                metrics = model.training_step(
                    clean_input_ids=input_ids,
                    attention_mask=attention_mask,
                    corruptor=corruptor
                )
                
                loss = metrics["loss"]
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                total_denoising_loss += metrics["denoising_loss"].item()
                total_likelihood_loss += metrics["likelihood_loss"].item()
                successful_batches += 1
                
                if batch_idx % 20 == 0:
                    print(f"Epoch {epoch+1}/{max_epochs}, Batch {batch_idx}/{batches_per_epoch}: "
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
        
        # Validation
        try:
            val_perplexity = compute_perplexity(model, val_loader, device)
            print(f"   Validation Perplexity: {val_perplexity:.2f}")
        except Exception as e:
            print(f"   Validation failed: {e}")
            val_perplexity = None
        
        # Save checkpoint if needed
        if checkpoint_manager.should_save_checkpoint(epoch + 1):
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                loss=avg_loss,
                config=config,
                validation_loss=val_perplexity,
                additional_data={
                    'denoising_loss': avg_denoising,
                    'likelihood_loss': avg_likelihood,
                    'epoch_time': epoch_time,
                    'successful_batches': successful_batches
                }
            )
            
            # Test generation on checkpoint epochs
            test_generation_quality(model, data_module, device, num_tests=2)
        
        model.train()  # Back to training mode
    
    print("\n🎉 Training completed!")
    
    # Final generation test
    print("\n🎯 FINAL Generation Quality Test:")
    test_generation_quality(model, data_module, device, num_tests=5)
    
    # Export final model
    if export_final:
        print("\n📦 Exporting final model...")
        export_path = checkpoint_manager.export_model(
            model=model,
            config=config,
            export_name=f"bijective_diffusion_final_epoch_{max_epochs}"
        )
        print(f"✅ Model exported to: {export_path}")
    
    # Training summary
    print(f"\n🏆 Training Summary:")
    summary = checkpoint_manager.get_training_summary()
    for key, value in summary.items():
        if key != "model_info":
            print(f"   {key}: {value}")
    
    # List all checkpoints
    print(f"\n💾 Available Checkpoints:")
    checkpoints = checkpoint_manager.list_checkpoints()
    for cp in checkpoints:
        print(f"   Epoch {cp['epoch']:3d}: {cp['path']} ({cp['size_mb']:.1f}MB)")
    
    best_checkpoint = checkpoint_manager.get_best_checkpoint()
    if best_checkpoint:
        print(f"   🏆 Best: {best_checkpoint}")
    
    return model, checkpoint_manager


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Train bijective diffusion model with checkpoints")
    parser.add_argument("--resume", type=str, default=None, 
                       help="Resume from checkpoint (path, 'latest', or 'best')")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs to train")
    parser.add_argument("--checkpoint-every", type=int, default=2,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--no-export", action="store_true",
                       help="Don't export final model")
    
    args = parser.parse_args()
    
    try:
        model, checkpoint_manager = train_bijective_model(
            resume_from=args.resume,
            export_final=not args.no_export,
            checkpoint_every=args.checkpoint_every,
            max_epochs=args.epochs
        )
        
        print("\n✅ SUCCESS: Training with checkpoints completed!")
        print("💾 All checkpoints and exports saved!")
        print("📚 NO SYNTHETIC DATA used - only real text data!")
        print("🛠️  Device compatibility maintained throughout!")
        
    except Exception as e:
        print(f"\n❌ FAILED: Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
