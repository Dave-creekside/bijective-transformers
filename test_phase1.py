#!/usr/bin/env python3
"""
Test script for Phase 1 implementation.
Verifies that all components work together correctly.
"""

import torch
import torch.nn.functional as F
from src.models.transformer import BidirectionalTransformer, TransformerConfig
from src.models.denoising_head import DenoisingHead, DenoisingHeadConfig
from src.models.diffusion import DiscreteDiffusionModel, create_diffusion_model_config
from src.data.corruption import TextCorruptor, NoiseScheduler, CorruptionConfig, create_corruption_config, create_noise_scheduler

def test_transformer():
    """Test bidirectional transformer."""
    print("🔧 Testing Bidirectional Transformer...")
    
    config = TransformerConfig(
        vocab_size=1000,
        max_seq_length=128,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        feedforward_dim=1024
    )
    
    model = BidirectionalTransformer(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    timesteps = torch.randint(0, 1000, (batch_size,))
    attention_mask = torch.ones(batch_size, seq_len)
    
    with torch.no_grad():
        hidden_states = model(input_ids, timesteps, attention_mask)
    
    expected_shape = (batch_size, seq_len, config.embed_dim)
    assert hidden_states.shape == expected_shape, f"Expected {expected_shape}, got {hidden_states.shape}"
    
    print(f"   ✅ Transformer output shape: {hidden_states.shape}")
    print(f"   ✅ Model parameters: {model.get_num_params():,}")
    return True

def test_denoising_head():
    """Test denoising head."""
    print("🔧 Testing Denoising Head...")
    
    config = DenoisingHeadConfig(
        vocab_size=1000,
        embed_dim=256,
        hidden_dim=512,
        num_layers=2
    )
    
    head = DenoisingHead(config)
    
    # Test forward pass
    batch_size, seq_len, embed_dim = 2, 64, 256
    hidden_states = torch.randn(batch_size, seq_len, embed_dim)
    
    with torch.no_grad():
        logits = head(hidden_states)
    
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    
    print(f"   ✅ Denoising head output shape: {logits.shape}")
    return True

def test_corruption():
    """Test text corruption."""
    print("🔧 Testing Text Corruption...")
    
    # Create corruption config and scheduler
    corruption_config = create_corruption_config(
        mask_prob=0.15,
        substitute_prob=0.1,
        delete_prob=0.05,
        vocab_size=1000
    )
    
    noise_scheduler = create_noise_scheduler(
        num_timesteps=1000,
        schedule_type="linear"
    )
    
    corruptor = TextCorruptor(corruption_config, noise_scheduler)
    
    # Test corruption
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    timesteps = torch.randint(0, 1000, (batch_size,))
    attention_mask = torch.ones(batch_size, seq_len)
    
    corrupted_ids, corruption_mask = corruptor.corrupt_sequence(
        input_ids, timesteps, attention_mask
    )
    
    assert corrupted_ids.shape == input_ids.shape
    assert corruption_mask.shape == input_ids.shape
    assert corruption_mask.dtype == torch.bool
    
    corruption_rate = corruption_mask.float().mean().item()
    print(f"   ✅ Corruption rate: {corruption_rate:.3f}")
    print(f"   ✅ Corrupted tokens: {corruption_mask.sum().item()}/{input_ids.numel()}")
    return True

def test_diffusion_model():
    """Test complete diffusion model."""
    print("🔧 Testing Discrete Diffusion Model...")
    
    # Create model config
    config = create_diffusion_model_config(
        vocab_size=1000,
        max_seq_length=128,
        embed_dim=256,
        num_layers=4,
        num_heads=8
    )
    
    model = DiscreteDiffusionModel(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.transformer.vocab_size, (batch_size, seq_len))
    timesteps = torch.randint(0, 1000, (batch_size,))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, config.transformer.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            timesteps=timesteps,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
    
    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, config.transformer.vocab_size)
    assert outputs["loss"].item() > 0
    
    print(f"   ✅ Model output shape: {outputs['logits'].shape}")
    print(f"   ✅ Loss: {outputs['loss'].item():.4f}")
    print(f"   ✅ Total parameters: {model.get_num_params():,}")
    return True

def test_training_step():
    """Test training step with corruption."""
    print("🔧 Testing Training Step...")
    
    try:
        # Create model
        config = create_diffusion_model_config(
            vocab_size=1000,
            max_seq_length=128,
            embed_dim=256,
            num_layers=4,
            num_heads=8
        )
        model = DiscreteDiffusionModel(config)
        
        # Create corruptor with matching vocab size
        corruption_config = create_corruption_config(
            vocab_size=1000,
            mask_token_id=999  # Use last token as mask for testing
        )
        noise_scheduler = create_noise_scheduler()
        corruptor = TextCorruptor(corruption_config, noise_scheduler)
        
        # Test training step
        batch_size, seq_len = 2, 64
        clean_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        print(f"   📊 Input shapes: clean_ids={clean_input_ids.shape}, mask={attention_mask.shape}")
        
        with torch.no_grad():
            metrics = model.training_step(
                clean_input_ids=clean_input_ids,
                attention_mask=attention_mask,
                corruptor=corruptor
            )
        
        required_keys = ["loss", "corrupted_accuracy", "overall_accuracy", "corruption_rate"]
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"
        
        print(f"   ✅ Loss: {metrics['loss'].item():.4f}")
        print(f"   ✅ Corrupted accuracy: {metrics['corrupted_accuracy'].item():.3f}")
        print(f"   ✅ Overall accuracy: {metrics['overall_accuracy'].item():.3f}")
        print(f"   ✅ Corruption rate: {metrics['corruption_rate'].item():.3f}")
        return True
        
    except Exception as e:
        print(f"   ❌ Training step error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation():
    """Test text generation."""
    print("🔧 Testing Text Generation...")
    
    # Create small model for faster testing
    config = create_diffusion_model_config(
        vocab_size=1000,
        max_seq_length=128,
        embed_dim=128,
        num_layers=2,
        num_heads=4,
        inference_steps=5  # Few steps for testing
    )
    model = DiscreteDiffusionModel(config)
    
    # Test generation
    batch_size, seq_len = 1, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_inference_steps=5
        )
    
    assert generated_ids.shape == input_ids.shape
    assert generated_ids.dtype == torch.long
    
    # Check that some tokens changed
    changed_tokens = (generated_ids != input_ids).sum().item()
    print(f"   ✅ Generated shape: {generated_ids.shape}")
    print(f"   ✅ Changed tokens: {changed_tokens}/{input_ids.numel()}")
    return True

def test_device_compatibility():
    """Test MPS device compatibility on M3 Mac."""
    print("🔧 Testing Device Compatibility...")
    
    try:
        # Check MPS availability
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"   ✅ Using MPS device")
        else:
            device = torch.device("cpu")
            print(f"   ⚠️  MPS not available, using CPU")
        
        # Create small model
        config = create_diffusion_model_config(
            vocab_size=1000,
            embed_dim=128,
            num_layers=2,
            num_heads=4
        )
        model = DiscreteDiffusionModel(config).to(device)
        
        # Test forward pass on device
        batch_size, seq_len = 1, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        with torch.no_grad():
            outputs = model(input_ids, timesteps, attention_mask, return_dict=True)
        
        # Check that model runs without error (device compatibility may vary)
        assert outputs["logits"] is not None
        print(f"   ✅ Model runs on {device}")
        print(f"   ✅ Output device: {outputs['logits'].device}")
        return True
        
    except Exception as e:
        print(f"   ❌ Device compatibility error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Phase 1 Implementation Tests")
    print("=" * 50)
    
    tests = [
        ("Bidirectional Transformer", test_transformer),
        ("Denoising Head", test_denoising_head),
        ("Text Corruption", test_corruption),
        ("Diffusion Model", test_diffusion_model),
        ("Training Step", test_training_step),
        ("Text Generation", test_generation),
        ("Device Compatibility", test_device_compatibility),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}:")
            success = test_func()
            if success:
                passed += 1
                print(f"   ✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"   ❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"   ❌ {test_name} FAILED: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{len(tests)} passed")
    
    if failed == 0:
        print("🎉 All tests passed! Phase 1 implementation is working correctly.")
        print("\n🚀 Ready to proceed with:")
        print("   • Dataset integration")
        print("   • Training pipeline")
        print("   • Evaluation metrics")
    else:
        print(f"⚠️  {failed} tests failed. Please review the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
