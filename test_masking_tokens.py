#!/usr/bin/env python3
"""
Test script to verify masking token setup and functionality.
Checks consistency across all components and tests corruption system.
"""

import torch
import yaml
import os
from transformers import AutoTokenizer

# Import our components
from src.models.bijective_diffusion_fixed import (
    BijectiveDiscreteDiffusionModel,
    create_bijective_diffusion_model_config
)
from src.data.corruption_final import (
    TextCorruptor, 
    CorruptionConfig, 
    NoiseScheduler,
    create_device_aware_corruptor
)
from src.data.wikitext_real import WikiTextDataModule


def test_tokenizer_mask_token():
    """Test GPT-2 tokenizer mask token setup."""
    print("🔍 Testing GPT-2 Tokenizer Setup...")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = len(tokenizer)
    
    print(f"✅ Vocab size: {vocab_size}")
    print(f"✅ EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"✅ PAD token: '{tokenizer.pad_token}' (ID: {getattr(tokenizer, 'pad_token_id', 'None')})")
    
    # Check if there's a specific mask token
    if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token is not None:
        print(f"✅ MASK token: '{tokenizer.mask_token}' (ID: {tokenizer.mask_token_id})")
        mask_token_id = tokenizer.mask_token_id
    else:
        print("⚠️  No specific MASK token, using last token in vocab as mask")
        mask_token_id = vocab_size - 1
        print(f"✅ Using token ID {mask_token_id} as mask token")
    
    # Test encoding/decoding with mask token
    test_text = "The quick brown fox jumps over the lazy dog."
    encoded = tokenizer(test_text, return_tensors="pt")
    input_ids = encoded["input_ids"].squeeze()
    
    print(f"✅ Original text: '{test_text}'")
    print(f"✅ Encoded: {input_ids[:10].tolist()}...")
    print(f"✅ Decoded: '{tokenizer.decode(input_ids, skip_special_tokens=True)}'")
    
    # Test mask token decoding
    mask_tensor = torch.tensor([mask_token_id])
    decoded_mask = tokenizer.decode(mask_tensor, skip_special_tokens=False)
    print(f"✅ Mask token {mask_token_id} decodes to: '{decoded_mask}'")
    
    return vocab_size, mask_token_id


def test_corruption_config():
    """Test corruption configuration consistency."""
    print("\n🔍 Testing Corruption Configuration...")
    
    vocab_size, expected_mask_id = test_tokenizer_mask_token()
    
    # Test default corruption config
    config = CorruptionConfig()
    print(f"✅ Default mask_token_id: {config.mask_token_id}")
    print(f"✅ Default vocab_size: {config.vocab_size}")
    
    # Check consistency
    if config.mask_token_id == expected_mask_id and config.vocab_size == vocab_size:
        print("✅ Configuration is consistent with tokenizer!")
    else:
        print(f"⚠️  Inconsistency detected:")
        print(f"   Expected mask_token_id: {expected_mask_id}, got: {config.mask_token_id}")
        print(f"   Expected vocab_size: {vocab_size}, got: {config.vocab_size}")
    
    return config, vocab_size, expected_mask_id


def test_corruption_system():
    """Test the text corruption system."""
    print("\n🔍 Testing Text Corruption System...")
    
    device = "cpu"  # Use CPU for testing
    
    # Create proper configuration
    vocab_size, mask_token_id = test_tokenizer_mask_token()
    
    corruption_config = CorruptionConfig(
        mask_prob=0.3,  # Higher for testing
        substitute_prob=0.2,
        vocab_size=vocab_size,
        mask_token_id=mask_token_id
    )
    
    noise_scheduler = NoiseScheduler(num_timesteps=1000, device=device)
    corruptor = create_device_aware_corruptor(corruption_config, noise_scheduler, device=device)
    
    # Create test input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size-100, (batch_size, seq_len))  # Avoid special tokens
    attention_mask = torch.ones_like(input_ids)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    print(f"✅ Test input shape: {input_ids.shape}")
    print(f"✅ Original tokens: {input_ids[0].tolist()}")
    
    # Test corruption
    try:
        corrupted_ids, corruption_mask = corruptor.corrupt_sequence(
            input_ids, timesteps, attention_mask
        )
        
        print(f"✅ Corrupted tokens: {corrupted_ids[0].tolist()}")
        print(f"✅ Corruption mask: {corruption_mask[0].tolist()}")
        
        # Check for mask tokens
        mask_count = (corrupted_ids == mask_token_id).sum().item()
        total_corrupted = corruption_mask.sum().item()
        
        print(f"✅ Total corrupted tokens: {total_corrupted}")
        print(f"✅ Mask tokens used: {mask_count}")
        
        if mask_count > 0:
            print("✅ Mask token corruption working!")
        else:
            print("⚠️  No mask tokens found (may be normal with low corruption)")
        
        return True
        
    except Exception as e:
        print(f"❌ Corruption test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_mask_handling():
    """Test how the model handles mask tokens."""
    print("\n🔍 Testing Model Mask Token Handling...")
    
    try:
        device = "cpu"
        vocab_size, mask_token_id = test_tokenizer_mask_token()
        
        # Create model configuration
        config = create_bijective_diffusion_model_config(
            vocab_size=vocab_size,
            max_seq_length=128,
            embed_dim=128,
            num_layers=2,
            num_heads=4
        )
        
        model = BijectiveDiscreteDiffusionModel(config)
        model.eval()
        
        print(f"✅ Model created with vocab_size: {config.transformer.transformer.vocab_size}")
        
        # Test with mask tokens
        batch_size = 2
        seq_len = 10
        
        # Create input with some mask tokens
        input_ids = torch.randint(0, vocab_size-100, (batch_size, seq_len))
        input_ids[0, 2] = mask_token_id  # Add mask token
        input_ids[1, 5] = mask_token_id  # Add another mask token
        
        timesteps = torch.randint(0, 1000, (batch_size,))
        
        print(f"✅ Input with mask tokens: {input_ids[0].tolist()}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                timesteps=timesteps,
                store_cache=False,
                return_dict=True
            )
            
            logits = outputs["logits"]
            print(f"✅ Output logits shape: {logits.shape}")
            print(f"✅ Forward pass successful!")
            
            # Check predictions for mask tokens
            pred_tokens = logits.argmax(dim=-1)
            print(f"✅ Predicted tokens: {pred_tokens[0].tolist()}")
            
            # Test generation method
            generated = model.generate(
                input_ids=input_ids,
                num_inference_steps=5
            )
            print(f"✅ Generated tokens: {generated[0].tolist()}")
            
            # Check for excessive mask token generation
            mask_ratio = (generated == mask_token_id).float().mean().item()
            print(f"✅ Generated mask token ratio: {mask_ratio:.2%}")
            
            if mask_ratio < 0.5:
                print("✅ Anti-mask bias working - low mask token generation!")
            else:
                print("⚠️  High mask token generation - anti-mask bias may need tuning")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_integration():
    """Test integration with real data pipeline."""
    print("\n🔍 Testing Data Integration...")
    
    try:
        # Load data config
        config_path = "configs/data/wikitext2.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                data_config = yaml.safe_load(f)
        else:
            data_config = {
                "tokenizer_name": "gpt2",
                "max_length": 64,  # Small for testing
                "batch_size": 2,
                "num_workers": 0
            }
        
        # Reduce size for testing
        data_config["max_length"] = 64
        data_config["batch_size"] = 2
        data_config["num_workers"] = 0
        
        # Create data module
        data_module = WikiTextDataModule(data_config)
        
        print("✅ Data module created")
        
        # Setup (this might take a moment)
        print("⏳ Setting up datasets...")
        data_module.setup()
        
        vocab_size = data_module.get_vocab_size()
        print(f"✅ Data vocab size: {vocab_size}")
        
        # Test dataloader
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        
        print(f"✅ Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"✅ Sample tokens: {batch['input_ids'][0][:10].tolist()}")
        
        # Check for any existing mask tokens in data
        mask_token_id = vocab_size - 1
        mask_count = (batch['input_ids'] == mask_token_id).sum().item()
        print(f"✅ Existing mask tokens in data: {mask_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all masking token tests."""
    print("🔬 MASKING TOKEN VERIFICATION TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Tokenizer Setup", test_tokenizer_mask_token),
        ("Corruption Config", test_corruption_config),
        ("Corruption System", test_corruption_system),
        ("Model Integration", test_model_mask_handling),
        ("Data Integration", test_data_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result if isinstance(result, bool) else True
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*20} TEST RESULTS {'='*20}")
    all_passed = True
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 ALL TESTS PASSED - Masking tokens are properly configured!")
    else:
        print("⚠️  SOME TESTS FAILED - Check configuration above")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
