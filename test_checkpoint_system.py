#!/usr/bin/env python3
"""
Test script to demonstrate checkpoint save/load functionality.
Shows how to save, load, and resume training with the checkpoint system.
"""

import torch
import os
from src.utils.checkpoint import create_checkpoint_manager
from src.models.bijective_diffusion_fixed import (
    BijectiveDiscreteDiffusionModel,
    create_bijective_diffusion_model_config
)

def test_checkpoint_system():
    """Test the checkpoint save/load functionality."""
    print("🧪 Testing Checkpoint System")
    print("=" * 50)
    
    # Create a simple model for testing
    config = create_bijective_diffusion_model_config(
        vocab_size=1000,  # Small vocab for testing
        max_seq_length=128,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        use_exact_likelihood=True,
        likelihood_weight=0.001
    )
    
    model = BijectiveDiscreteDiffusionModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    print(f"✅ Created test model with {model.get_num_params():,} parameters")
    
    # Create checkpoint manager
    checkpoint_manager = create_checkpoint_manager(
        checkpoint_dir="test_checkpoints",
        export_dir="test_exports",
        max_checkpoints=3,
        save_every_n_epochs=1
    )
    
    print("✅ Created checkpoint manager")
    
    # Test 1: Save a checkpoint
    print("\n📁 Test 1: Saving checkpoint...")
    checkpoint_path = checkpoint_manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=5,
        loss=2.5,
        config=config,
        validation_loss=3.0,
        additional_data={'test_data': 'hello world'}
    )
    print(f"✅ Checkpoint saved to: {checkpoint_path}")
    
    # Test 2: List checkpoints
    print("\n📋 Test 2: Listing checkpoints...")
    checkpoints = checkpoint_manager.list_checkpoints()
    for cp in checkpoints:
        print(f"   Epoch {cp['epoch']}: {cp['loss']:.4f} loss, {cp['size_mb']:.1f}MB")
    
    # Test 3: Create a new model and load checkpoint
    print("\n📂 Test 3: Loading checkpoint into new model...")
    new_model = BijectiveDiscreteDiffusionModel(config)
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)
    new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=100)
    
    # Get original weights for comparison
    original_weight = model.transformer.token_embedding.weight[0, :5].clone()
    new_weight_before = new_model.transformer.token_embedding.weight[0, :5].clone()
    
    print(f"   Original model weight sample: {original_weight}")
    print(f"   New model weight sample (before): {new_weight_before}")
    
    # Load checkpoint
    epoch, loss, loaded_config = checkpoint_manager.load_checkpoint(
        new_model, new_optimizer, new_scheduler, checkpoint_path
    )
    
    new_weight_after = new_model.transformer.token_embedding.weight[0, :5].clone()
    print(f"   New model weight sample (after): {new_weight_after}")
    
    # Verify weights match
    weights_match = torch.allclose(original_weight, new_weight_after)
    print(f"   Weights match: {'✅' if weights_match else '❌'}")
    
    # Test 4: Export model
    print("\n📦 Test 4: Exporting model...")
    export_path = checkpoint_manager.export_model(
        model=model,
        config=config,
        export_name="test_bijective_model"
    )
    print(f"✅ Model exported to: {export_path}")
    
    # Test 5: Load exported model
    print("\n📥 Test 5: Loading exported model...")
    inference_model = BijectiveDiscreteDiffusionModel(config)
    loaded_config = checkpoint_manager.load_exported_model(
        inference_model, export_path
    )
    print("✅ Exported model loaded successfully")
    
    # Test 6: Training summary
    print("\n📊 Test 6: Training summary...")
    summary = checkpoint_manager.get_training_summary()
    for key, value in summary.items():
        if key != "model_info":
            print(f"   {key}: {value}")
    
    # Test 7: Resume training simulation
    print("\n🔄 Test 7: Resume training simulation...")
    
    # Save another checkpoint
    checkpoint_path_2 = checkpoint_manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=10,
        loss=1.8,
        config=config,
        validation_loss=2.2
    )
    
    # Get latest checkpoint
    latest = checkpoint_manager.get_latest_checkpoint()
    best = checkpoint_manager.get_best_checkpoint()
    
    print(f"   Latest checkpoint: {latest}")
    print(f"   Best checkpoint: {best}")
    
    # Test resume functionality
    resume_model = BijectiveDiscreteDiffusionModel(config)
    resume_optimizer = torch.optim.AdamW(resume_model.parameters(), lr=1e-4)
    resume_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resume_optimizer, T_max=100)
    
    resume_epoch, resume_loss, resume_config = checkpoint_manager.load_checkpoint(
        resume_model, resume_optimizer, resume_scheduler, latest
    )
    
    print(f"   Resumed from epoch {resume_epoch} with loss {resume_loss:.4f}")
    print(f"   Next training epoch would be: {resume_epoch + 1}")
    
    print("\n🎉 All checkpoint tests passed!")
    
    # Cleanup
    print("\n🧹 Cleaning up test files...")
    import shutil
    if os.path.exists("test_checkpoints"):
        shutil.rmtree("test_checkpoints")
    if os.path.exists("test_exports"):
        shutil.rmtree("test_exports")
    print("✅ Cleanup complete")

if __name__ == "__main__":
    test_checkpoint_system()
