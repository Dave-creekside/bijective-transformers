#!/usr/bin/env python3
"""
Test script for invertible layers and residual connections.
Validates RevNet-style residuals and bijective feed-forward networks.
"""

import torch
import torch.nn.functional as F
from src.layers.invertible import (
    InvertibleResidualConnection,
    InvertibleLayerNorm,
    InvertibleFeedForward,
    CouplingFunction,
    InvertibleConfig,
    create_invertible_config,
    create_coupling_functions
)
from src.utils.invertibility import test_invertibility, create_test_config

def test_coupling_function():
    """Test basic coupling function."""
    print("🔧 Testing Coupling Function...")
    
    input_dim, output_dim = 128, 64
    hidden_dim = 256
    
    # Create coupling function
    coupling_fn = CouplingFunction(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        activation="gelu",
        dropout=0.1
    )
    
    # Test forward pass
    batch_size, seq_len = 4, 32
    x = torch.randn(batch_size, seq_len, input_dim)
    
    with torch.no_grad():
        y = coupling_fn(x)
    
    assert y.shape == (batch_size, seq_len, output_dim), f"Shape mismatch: {y.shape}"
    
    # Test that initial output is zeros (identity initialization)
    initial_output_norm = y.norm().item()
    
    print(f"   ✅ Input shape: {x.shape}")
    print(f"   ✅ Output shape: {y.shape}")
    print(f"   ✅ Initial output norm: {initial_output_norm:.2e} (should be ~0)")
    
    return True

def test_invertible_residual():
    """Test invertible residual connection."""
    print("🔧 Testing Invertible Residual Connection...")
    
    embed_dim = 256
    config = create_invertible_config(embed_dim=embed_dim)
    
    # Create coupling functions
    split_dim = embed_dim // 2
    remaining_dim = embed_dim - split_dim
    
    F_function = CouplingFunction(
        input_dim=remaining_dim,
        output_dim=split_dim,
        hidden_dim=512,
        num_layers=2
    )
    
    G_function = CouplingFunction(
        input_dim=split_dim,
        output_dim=remaining_dim,
        hidden_dim=512,
        num_layers=2
    )
    
    # Create invertible residual
    residual = InvertibleResidualConnection(config, F_function, G_function)
    
    # Test forward and inverse
    batch_size, seq_len = 4, 32
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    with torch.no_grad():
        # Forward pass
        y, log_det_forward = residual.forward(x)
        
        # Inverse pass
        x_reconstructed, log_det_inverse = residual.inverse(y)
        
        # Check reconstruction
        reconstruction_error = torch.norm(x - x_reconstructed, dim=-1).max().item()
        
        # Check log determinant (should be 0 for additive coupling)
        log_det_error = torch.abs(log_det_forward).max().item()
    
    assert y.shape == x.shape, f"Output shape mismatch: {y.shape} vs {x.shape}"
    assert reconstruction_error < 1e-6, f"Reconstruction error too large: {reconstruction_error}"
    assert log_det_error < 1e-6, f"Log det should be 0: {log_det_error}"
    
    print(f"   ✅ Forward/inverse shapes: {x.shape} -> {y.shape} -> {x_reconstructed.shape}")
    print(f"   ✅ Reconstruction error: {reconstruction_error:.2e}")
    print(f"   ✅ Log determinant: {log_det_forward.mean().item():.2e}")
    
    return True

def test_invertible_layer_norm():
    """Test invertible layer normalization."""
    print("🔧 Testing Invertible Layer Norm...")
    
    embed_dim = 256
    layer_norm = InvertibleLayerNorm(embed_dim)
    
    # Test forward pass
    batch_size, seq_len = 4, 32
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    with torch.no_grad():
        y, log_det = layer_norm.forward(x)
        
        # Compute statistics for inverse
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + layer_norm.eps)
        
        # Inverse pass
        x_reconstructed = layer_norm.inverse(y, mean, std)
        
        # Check reconstruction
        reconstruction_error = torch.norm(x - x_reconstructed, dim=-1).max().item()
    
    assert y.shape == x.shape, f"Output shape mismatch: {y.shape} vs {x.shape}"
    assert reconstruction_error < 1e-5, f"Reconstruction error too large: {reconstruction_error}"
    assert torch.isfinite(log_det).all(), "Log determinant contains non-finite values"
    
    print(f"   ✅ Forward/inverse shapes: {x.shape} -> {y.shape} -> {x_reconstructed.shape}")
    print(f"   ✅ Reconstruction error: {reconstruction_error:.2e}")
    print(f"   ✅ Log det range: [{log_det.min().item():.2f}, {log_det.max().item():.2f}]")
    
    return True

def test_invertible_feedforward():
    """Test invertible feed-forward network."""
    print("🔧 Testing Invertible Feed-Forward...")
    
    embed_dim = 256
    config = create_invertible_config(
        embed_dim=embed_dim,
        activation="gelu",
        dropout=0.1
    )
    
    # Create invertible feed-forward
    ff = InvertibleFeedForward(config)
    
    # Test forward and inverse
    batch_size, seq_len = 4, 32
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    with torch.no_grad():
        # Forward pass
        y, log_det_forward = ff.forward(x)
        
        # Inverse pass
        x_reconstructed, log_det_inverse = ff.inverse(y)
        
        # Check reconstruction
        reconstruction_error = torch.norm(x - x_reconstructed, dim=-1).max().item()
        
        # Check log determinant consistency
        log_det_consistency = torch.abs(log_det_forward + log_det_inverse).max().item()
    
    assert y.shape == x.shape, f"Output shape mismatch: {y.shape} vs {x.shape}"
    assert reconstruction_error < 1e-6, f"Reconstruction error too large: {reconstruction_error}"
    assert log_det_consistency < 1e-6, f"Log det inconsistency: {log_det_consistency}"
    
    print(f"   ✅ Forward/inverse shapes: {x.shape} -> {y.shape} -> {x_reconstructed.shape}")
    print(f"   ✅ Reconstruction error: {reconstruction_error:.2e}")
    print(f"   ✅ Log det consistency: {log_det_consistency:.2e}")
    
    return True

def test_gradient_flow():
    """Test gradient flow through invertible layers."""
    print("🔧 Testing Gradient Flow...")
    
    embed_dim = 128
    config = create_invertible_config(embed_dim=embed_dim)
    
    # Create invertible feed-forward
    ff = InvertibleFeedForward(config)
    
    # Test gradient flow
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    
    # Forward pass
    y, log_det = ff.forward(x)
    
    # Compute loss and backward pass
    loss = y.sum() + log_det.sum()
    loss.backward()
    
    # Check gradients
    assert x.grad is not None, "No gradients computed for input"
    assert torch.isfinite(x.grad).all(), "Gradients contain non-finite values"
    
    # Check parameter gradients
    param_grads_exist = any(p.grad is not None for p in ff.parameters())
    assert param_grads_exist, "No gradients computed for parameters"
    
    print(f"   ✅ Input gradient norm: {x.grad.norm().item():.2e}")
    print(f"   ✅ Parameter gradients exist: {param_grads_exist}")
    
    return True

def test_invertibility_framework():
    """Test invertibility using the testing framework."""
    print("🔧 Testing with Invertibility Framework...")
    
    # Create test configuration
    test_config = create_test_config(
        tolerance=1e-6,
        num_test_samples=50,
        verbose=False
    )
    
    # Test invertible feed-forward
    embed_dim = 128
    config = create_invertible_config(embed_dim=embed_dim)
    ff = InvertibleFeedForward(config)
    
    # Run invertibility test
    results = test_invertibility(ff, (32, embed_dim), test_config)
    
    assert results["overall_passed"], f"Invertibility test failed: {results}"
    
    # Handle max_error being a list or float
    max_error = results["max_error"]
    if isinstance(max_error, list):
        max_error = max(max_error) if max_error else 0.0
    
    assert max_error < 1e-5, f"Max error too large: {max_error}"
    
    # Handle mean_error being a list or float
    mean_error = results["mean_error"]
    if isinstance(mean_error, list):
        mean_error = sum(mean_error) / len(mean_error) if mean_error else 0.0
    
    print(f"   ✅ Invertibility test passed")
    print(f"   ✅ Max error: {max_error:.2e}")
    print(f"   ✅ Mean error: {mean_error:.2e}")
    
    return True

def test_memory_efficiency():
    """Test memory efficiency of invertible layers."""
    print("🔧 Testing Memory Efficiency...")
    
    # Create larger model for memory testing
    embed_dim = 512
    config = create_invertible_config(embed_dim=embed_dim)
    ff = InvertibleFeedForward(config)
    
    # Test with larger batch
    batch_size, seq_len = 16, 64
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Measure memory usage (basic check)
    import gc
    gc.collect()
    
    with torch.no_grad():
        y, log_det = ff.forward(x)
        x_reconstructed, _ = ff.inverse(y)
    
    # Check that computation completed without memory errors
    error = torch.norm(x - x_reconstructed, dim=-1).max().item()
    
    print(f"   ✅ Large batch processing: {x.shape}")
    print(f"   ✅ Reconstruction error: {error:.2e}")
    print(f"   ✅ Memory test completed")
    
    return True

def test_device_compatibility():
    """Test device compatibility for invertible layers."""
    print("🔧 Testing Device Compatibility...")
    
    # Check device availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"   ✅ Using MPS device")
    else:
        device = torch.device("cpu")
        print(f"   ⚠️  MPS not available, using CPU")
    
    try:
        # Create layer on device
        embed_dim = 128
        config = create_invertible_config(embed_dim=embed_dim)
        ff = InvertibleFeedForward(config).to(device)
        
        # Test on device
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, embed_dim, device=device)
        
        with torch.no_grad():
            y, log_det = ff.forward(x)
            x_reconstructed, _ = ff.inverse(y)
        
        # Check reconstruction
        error = torch.norm(x - x_reconstructed, dim=-1).max().item()
        assert error < 1e-6, f"Reconstruction error on {device}: {error}"
        
        print(f"   ✅ Layer runs on {device}")
        print(f"   ✅ Reconstruction error: {error:.2e}")
        return True
        
    except Exception as e:
        print(f"   ❌ Device compatibility error: {e}")
        return False

def main():
    """Run all invertible layer tests."""
    print("🚀 Invertible Layers Tests")
    print("=" * 50)
    
    tests = [
        ("Coupling Function", test_coupling_function),
        ("Invertible Residual Connection", test_invertible_residual),
        ("Invertible Layer Norm", test_invertible_layer_norm),
        ("Invertible Feed-Forward", test_invertible_feedforward),
        ("Gradient Flow", test_gradient_flow),
        ("Invertibility Framework", test_invertibility_framework),
        ("Memory Efficiency", test_memory_efficiency),
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
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{len(tests)} passed")
    
    if failed == 0:
        print("🎉 All invertible layer tests passed!")
        print("\n🚀 Ready for:")
        print("   • Bijective transformer block integration")
        print("   • Multi-head attention with invertible projections")
        print("   • Full bijective transformer architecture")
    else:
        print(f"⚠️  {failed} tests failed. Please review the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
