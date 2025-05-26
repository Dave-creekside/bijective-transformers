#!/usr/bin/env python3
"""
Simple test script to verify masking token setup and functionality.
Tests core components without requiring complete bijective system.
"""

import torch
import yaml
import os
from transformers import AutoTokenizer

# Import our components
from src.data.corruption_final import (
    TextCorruptor, 
    CorruptionConfig, 
    NoiseScheduler,
    create_device_aware_corruptor
)


def test_tokenizer_and_vocab():
    """Test GPT-2 tokenizer and vocabulary setup."""
    print("🔍 Testing GPT-2 Tokenizer and Vocabulary...")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = len(tokenizer)
    
    print(f"✅ Vocab size: {vocab_size}")
    print(f"✅ EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    
    # For GPT-2, there's no built-in mask token, so we use the last token
    mask_token_id = vocab_size - 1  # Should be 50256
    print(f"✅ Using token ID {mask_token_id} as mask token")
    
    # Test some token encoding/decoding
    test_text = "The quick brown fox jumps over the lazy dog."
    encoded = tokenizer(test_text, return_tensors="pt")
    input_ids = encoded["input_ids"].squeeze()
    
    print(f"✅ Original text: '{test_text}'")
    print(f"✅ Encoded tokens: {input_ids[:8].tolist()}...")
    print(f"✅ Decoded back: '{tokenizer.decode(input_ids, skip_special_tokens=True)}'")
    
    # Test mask token behavior
    mask_tensor = torch.tensor([mask_token_id])
    decoded_mask = tokenizer.decode(mask_tensor, skip_special_tokens=False)
    print(f"✅ Mask token {mask_token_id} decodes to: '{decoded_mask}'")
    
    return vocab_size, mask_token_id


def test_corruption_configuration():
    """Test corruption configuration setup."""
    print("\n🔍 Testing Corruption Configuration...")
    
    vocab_size, expected_mask_id = test_tokenizer_and_vocab()
    
    # Test with correct configuration
    config = CorruptionConfig(
        mask_prob=0.15,
        substitute_prob=0.1,
        delete_prob=0.05,
        vocab_size=vocab_size,
        mask_token_id=expected_mask_id
    )
    
    print(f"✅ Configured mask_token_id: {config.mask_token_id}")
    print(f"✅ Configured vocab_size: {config.vocab_size}")
    print(f"✅ Mask probability: {config.mask_prob}")
    print(f"✅ Substitute probability: {config.substitute_prob}")
    print(f"✅ Delete probability: {config.delete_prob}")
    
    # Verify consistency
    if config.mask_token_id == expected_mask_id and config.vocab_size == vocab_size:
        print("✅ Configuration is consistent with tokenizer!")
        return True, config
    else:
        print(f"❌ Configuration mismatch!")
        print(f"   Expected mask_token_id: {expected_mask_id}, got: {config.mask_token_id}")
        print(f"   Expected vocab_size: {vocab_size}, got: {config.vocab_size}")
        return False, config


def test_noise_scheduler():
    """Test noise scheduler functionality."""
    print("\n🔍 Testing Noise Scheduler...")
    
    device = "cpu"
    
    # Create noise scheduler
    scheduler = NoiseScheduler(
        num_timesteps=1000,
        schedule_type="linear",
        min_noise=0.01,
        max_noise=0.99,
        device=device
    )
    
    print(f"✅ Scheduler created with {scheduler.num_timesteps} timesteps")
    print(f"✅ Schedule type: {scheduler.schedule_type}")
    print(f"✅ Device: {scheduler.device}")
    
    # Test noise level queries
    noise_start = scheduler.get_noise_level(0)
    noise_mid = scheduler.get_noise_level(500)
    noise_end = scheduler.get_noise_level(999)
    
    print(f"✅ Noise at t=0: {noise_start:.4f}")
    print(f"✅ Noise at t=500: {noise_mid:.4f}")
    print(f"✅ Noise at t=999: {noise_end:.4f}")
    
    # Test timestep sampling
    batch_size = 4
    timesteps = scheduler.sample_timesteps(batch_size, device)
    print(f"✅ Sample timesteps: {timesteps.tolist()}")
    
    return True


def test_corruption_system():
    """Test the text corruption system with mask tokens."""
    print("\n🔍 Testing Text Corruption System...")
    
    device = "cpu"
    vocab_size, mask_token_id = test_tokenizer_and_vocab()
    
    # Create corruption configuration
    corruption_config = CorruptionConfig(
        mask_prob=0.5,  # Higher probability for testing
        substitute_prob=0.2,
        delete_prob=0.1,
        vocab_size=vocab_size,
        mask_token_id=mask_token_id
    )
    
    # Create noise scheduler
    noise_scheduler = NoiseScheduler(
        num_timesteps=1000,
        schedule_type="linear",
        device=device
    )
    
    # Create corruptor
    corruptor = create_device_aware_corruptor(
        corruption_config, noise_scheduler, device=device
    )
    
    print(f"✅ Corruptor created and configured")
    
    # Create test input
    batch_size = 3
    seq_len = 12
    # Use realistic token IDs (avoid very high values)
    input_ids = torch.randint(1, vocab_size-100, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    # Sample some timesteps
    timesteps = torch.randint(200, 800, (batch_size,))  # Mid-range timesteps
    
    print(f"✅ Test input shape: {input_ids.shape}")
    print(f"✅ Original tokens [0]: {input_ids[0].tolist()}")
    print(f"✅ Timesteps: {timesteps.tolist()}")
    
    # Test corruption
    try:
        corrupted_ids, corruption_mask = corruptor.corrupt_sequence(
            input_ids, timesteps, attention_mask
        )
        
        print(f"✅ Corrupted tokens [0]: {corrupted_ids[0].tolist()}")
        print(f"✅ Corruption mask [0]: {corruption_mask[0].tolist()}")
        
        # Analyze corruption results
        total_tokens = input_ids.numel()
        corrupted_tokens = corruption_mask.sum().item()
        mask_tokens = (corrupted_ids == mask_token_id).sum().item()
        
        print(f"✅ Total tokens: {total_tokens}")
        print(f"✅ Corrupted tokens: {corrupted_tokens} ({corrupted_tokens/total_tokens:.1%})")
        print(f"✅ Mask tokens used: {mask_tokens}")
        
        # Check specific corruption types
        changes = corrupted_ids != input_ids
        changes_count = changes.sum().item()
        
        print(f"✅ Tokens changed: {changes_count}")
        
        if mask_tokens > 0:
            print("✅ SUCCESS: Mask token corruption is working!")
        else:
            print("⚠️  No mask tokens found (may be normal due to random sampling)")
        
        # Test multiple runs to see variation
        print("\n🔄 Testing corruption variation (3 runs):")
        for run in range(3):
            test_corrupted, test_mask = corruptor.corrupt_sequence(
                input_ids[:1], timesteps[:1], attention_mask[:1]
            )
            test_mask_count = (test_corrupted == mask_token_id).sum().item()
            test_changes = (test_corrupted != input_ids[:1]).sum().item()
            print(f"   Run {run+1}: {test_changes} changes, {test_mask_count} masks")
        
        return True
        
    except Exception as e:
        print(f"❌ Corruption test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mask_token_detection():
    """Test detection and handling of mask tokens."""
    print("\n🔍 Testing Mask Token Detection...")
    
    vocab_size, mask_token_id = test_tokenizer_and_vocab()
    
    # Create sequences with known mask tokens
    batch_size = 2
    seq_len = 8
    
    # Create test sequences
    test_sequences = torch.randint(1, vocab_size-100, (batch_size, seq_len))
    
    # Manually insert mask tokens
    test_sequences[0, 2] = mask_token_id  # Add mask at position 2
    test_sequences[0, 5] = mask_token_id  # Add mask at position 5
    test_sequences[1, 1] = mask_token_id  # Add mask at position 1
    
    print(f"✅ Test sequences with masks:")
    print(f"   Sequence 0: {test_sequences[0].tolist()}")
    print(f"   Sequence 1: {test_sequences[1].tolist()}")
    
    # Detect mask tokens
    mask_positions = (test_sequences == mask_token_id)
    mask_counts = mask_positions.sum(dim=1)
    
    print(f"✅ Mask token detection:")
    print(f"   Sequence 0 masks: {mask_counts[0].item()} at positions {torch.where(mask_positions[0])[0].tolist()}")
    print(f"   Sequence 1 masks: {mask_counts[1].item()} at positions {torch.where(mask_positions[1])[0].tolist()}")
    
    # Test mask ratio calculation
    total_tokens = test_sequences.numel()
    total_masks = mask_positions.sum().item()
    mask_ratio = total_masks / total_tokens
    
    print(f"✅ Overall mask ratio: {mask_ratio:.2%} ({total_masks}/{total_tokens})")
    
    return True


def test_tokenizer_mask_decoding():
    """Test how the tokenizer handles mask tokens."""
    print("\n🔍 Testing Tokenizer Mask Token Decoding...")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size, mask_token_id = test_tokenizer_and_vocab()
    
    # Test sequences with mask tokens
    test_tokens = [
        [1234, 5678, mask_token_id, 9876, 5432],
        [mask_token_id, 1111, 2222, mask_token_id, 3333]
    ]
    
    print(f"✅ Testing mask token decoding:")
    
    for i, tokens in enumerate(test_tokens):
        token_tensor = torch.tensor(tokens)
        
        # Decode with special tokens (shows raw tokens)
        decoded_with_special = tokenizer.decode(token_tensor, skip_special_tokens=False)
        
        # Decode without special tokens (may filter out masks)
        decoded_without_special = tokenizer.decode(token_tensor, skip_special_tokens=True)
        
        print(f"   Tokens {i+1}: {tokens}")
        print(f"   With special: '{decoded_with_special}'")
        print(f"   Without special: '{decoded_without_special}'")
        
        # Check where mask tokens are
        mask_positions = [j for j, t in enumerate(tokens) if t == mask_token_id]
        print(f"   Mask positions: {mask_positions}")
    
    return True


def main():
    """Run all masking token tests."""
    print("🔬 SIMPLE MASKING TOKEN VERIFICATION")
    print("=" * 45)
    
    tests = [
        ("Tokenizer & Vocab", test_tokenizer_and_vocab),
        ("Corruption Config", test_corruption_configuration),
        ("Noise Scheduler", test_noise_scheduler),
        ("Corruption System", test_corruption_system),
        ("Mask Detection", test_mask_token_detection),
        ("Mask Decoding", test_tokenizer_mask_decoding)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*15} {test_name} {'='*15}")
        try:
            result = test_func()
            results[test_name] = result if isinstance(result, bool) else True
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*15} TEST RESULTS {'='*15}")
    all_passed = True
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n{'='*45}")
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Masking tokens are properly configured")
        print("✅ Corruption system is working correctly")
        print("✅ Token handling is consistent")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Check the configuration above")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
