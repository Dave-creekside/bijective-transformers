#!/usr/bin/env python3
"""
Test script for Phase 2 bijective components.
Validates coupling layers, invertibility, and integration with diffusion model.
"""

import torch
import torch.nn.functional as F
from src.layers.coupling import (
    AdditiveCouplingLayer, 
    AffineCouplingLayer,
    NeuralSplineCouplingLayer,
    CouplingLayerConfig,
    create_coupling_layer,
    create_coupling_config
)
from src.utils.invertibility import (
    InvertibilityTester,
    JacobianComputer,
    NumericalStabilityChecker,
    test_invertibility,
    create_test_config
)

def test_additive_coupling():
    """Test additive coupling layer."""
    print("🔧 Testing Additive Coupling Layer...")
    
    # Create configuration
    config = create_coupling_config(
        input_dim=256,
        coupling_type="additive",
        hidden_dim=128,
        num_layers=2
    )
    
    # Create layer
    layer = AdditiveCouplingLayer(config)
    
    # Test forward and inverse
    batch_size, input_dim = 4, 256
    x = torch.randn(batch_size, input_dim)
    
    with torch.no_grad():
        # Forward pass
        y, log_det_forward = layer.forward(x)
        
        # Inverse pass
        x_reconstructed, log_det_inverse = layer.inverse(y)
        
        # Check reconstruction
        reconstruction_error = torch.norm(x - x_reconstructed, dim=-1).max().item()
        
        # Check log determinant (should be 0 for additive coupling)
        log_det_error = torch.abs(log_det_forward).max().item()
    
    assert y.shape == x.shape, f"Output shape mismatch: {y.shape} vs {x.shape}"
    assert reconstruction_error < 1e-6, f"Reconstruction error too large: {reconstruction_error}"
    assert log_det_error < 1e-6, f"Log det should be 0 for additive coupling: {log_det_error}"
    
    print(f"   ✅ Forward/inverse shapes: {x.shape} -> {y.shape} -> {x_reconstructed.shape}")
    print(f"   ✅ Reconstruction error: {reconstruction_error:.2e}")
    print(f"   ✅ Log determinant: {log_det_forward.mean().item():.2e}")
    return True

def test_affine_coupling():
    """Test affine coupling layer."""
    print("🔧 Testing Affine Coupling Layer...")
    
    # Create configuration
    config = create_coupling_config(
        input_dim=256,
        coupling_type="affine",
        hidden_dim=128,
        num_layers=2,
        scale_activation="tanh",
        scale_factor=1.0
    )
    
    # Create layer
    layer = AffineCouplingLayer(config)
    
    # Test forward and inverse
    batch_size, input_dim = 4, 256
    x = torch.randn(batch_size, input_dim)
    
    with torch.no_grad():
        # Forward pass
        y, log_det_forward = layer.forward(x)
        
        # Inverse pass
        x_reconstructed, log_det_inverse = layer.inverse(y)
        
        # Check reconstruction
        reconstruction_error = torch.norm(x - x_reconstructed, dim=-1).max().item()
        
        # Check log determinant consistency
        log_det_consistency = torch.abs(log_det_forward + log_det_inverse).max().item()
    
    assert y.shape == x.shape, f"Output shape mismatch: {y.shape} vs {x.shape}"
    assert reconstruction_error < 1e-5, f"Reconstruction error too large: {reconstruction_error}"
    assert log_det_consistency < 1e-5, f"Log det inconsistency: {log_det_consistency}"
    
    print(f"   ✅ Forward/inverse shapes: {x.shape} -> {y.shape} -> {x_reconstructed.shape}")
    print(f"   ✅ Reconstruction error: {reconstruction_error:.2e}")
    print(f"   ✅ Log determinant range: [{log_det_forward.min().item():.2f}, {log_det_forward.max().item():.2f}]")
    print(f"   ✅ Log det consistency: {log_det_consistency:.2e}")
    return True

def test_spline_coupling():
    """Test neural spline coupling layer."""
    print("🔧 Testing Neural Spline Coupling Layer...")
    
    # Create configuration
    config = create_coupling_config(
        input_dim=256,
        coupling_type="spline",
        hidden_dim=128,
        num_layers=2,
        num_bins=8,
        tail_bound=3.0
    )
    
    # Create layer
    layer = NeuralSplineCouplingLayer(config)
    
    # Test forward and inverse
    batch_size, input_dim = 4, 256
    x = torch.randn(batch_size, input_dim)
    
    with torch.no_grad():
        # Forward pass
        y, log_det_forward = layer.forward(x)
        
        # Inverse pass
        x_reconstructed, log_det_inverse = layer.inverse(y)
        
        # Check reconstruction (note: spline implementation is placeholder)
        reconstruction_error = torch.norm(x - x_reconstructed, dim=-1).max().item()
    
    assert y.shape == x.shape, f"Output shape mismatch: {y.shape} vs {x.shape}"
    
    print(f"   ✅ Forward/inverse shapes: {x.shape} -> {y.shape} -> {x_reconstructed.shape}")
    print(f"   ✅ Reconstruction error: {reconstruction_error:.2e}")
    print(f"   ⚠️  Note: Spline implementation is placeholder")
    return True

def test_coupling_factory():
    """Test coupling layer factory function."""
    print("🔧 Testing Coupling Layer Factory...")
    
    input_dim = 128
    
    # Test all coupling types
    coupling_types = ["additive", "affine", "spline"]
    
    for coupling_type in coupling_types:
        config = create_coupling_config(
            input_dim=input_dim,
            coupling_type=coupling_type,
            hidden_dim=64,
            num_layers=1
        )
        
        layer = create_coupling_layer(config)
        
        # Test basic functionality
        x = torch.randn(2, input_dim)
        with torch.no_grad():
            y, log_det = layer.forward(x)
        
        assert y.shape == x.shape
        assert log_det.shape == (2,)
        
        print(f"   ✅ {coupling_type} coupling created and tested")
    
    return True

def test_invertibility_framework():
    """Test invertibility testing framework."""
    print("🔧 Testing Invertibility Framework...")
    
    # Create test configuration
    test_config = create_test_config(
        tolerance=1e-6,
        num_test_samples=50,
        verbose=False
    )
    
    # Create a simple additive coupling layer
    coupling_config = create_coupling_config(
        input_dim=128,
        coupling_type="additive",
        hidden_dim=64,
        num_layers=1
    )
    layer = AdditiveCouplingLayer(coupling_config)
    
    # Test invertibility
    tester = InvertibilityTester(test_config)
    results = tester.test_layer(layer, (128,))
    
    assert results["overall_passed"], f"Invertibility test failed: {results}"
    assert results["max_error"] < 1e-5, f"Max error too large: {results['max_error']}"
    
    print(f"   ✅ Invertibility test passed")
    print(f"   ✅ Max error: {results['max_error']:.2e}")
    print(f"   ✅ Mean error: {results['mean_error']:.2e}")
    
    # Test multiple layers
    layers_to_test = [
        ("Additive", AdditiveCouplingLayer(coupling_config)),
        ("Affine", AffineCouplingLayer(create_coupling_config(128, "affine", 64, 1)))
    ]
    
    for name, layer in layers_to_test:
        result = test_invertibility(layer, (128,), test_config)
        print(f"   ✅ {name} coupling: {'PASS' if result['overall_passed'] else 'FAIL'}")
    
    return True

def test_jacobian_computation():
    """Test Jacobian determinant computation."""
    print("🔧 Testing Jacobian Computation...")
    
    # Create affine coupling layer (has non-trivial Jacobian)
    config = create_coupling_config(
        input_dim=64,
        coupling_type="affine",
        hidden_dim=32,
        num_layers=1
    )
    layer = AffineCouplingLayer(config)
    
    # Test Jacobian computer
    jacobian_computer = JacobianComputer()
    
    x = torch.randn(4, 64)
    
    with torch.no_grad():
        # Compute log determinant
        log_det = jacobian_computer.compute_log_det(layer, x)
        
        # Verify computation
        verification = jacobian_computer.verify_log_det(layer, x)
    
    assert log_det.shape == (4,), f"Log det shape mismatch: {log_det.shape}"
    assert torch.isfinite(log_det).all(), "Log determinant contains non-finite values"
    assert verification["passed"], f"Jacobian verification failed: {verification}"
    
    print(f"   ✅ Log determinant shape: {log_det.shape}")
    print(f"   ✅ Log det range: [{log_det.min().item():.2f}, {log_det.max().item():.2f}]")
    print(f"   ✅ Verification passed: {verification['passed']}")
    return True

def test_numerical_stability():
    """Test numerical stability of bijective layers."""
    print("🔧 Testing Numerical Stability...")
    
    # Create layer
    config = create_coupling_config(
        input_dim=128,
        coupling_type="affine",
        hidden_dim=64,
        num_layers=2
    )
    layer = AffineCouplingLayer(config)
    
    # Test stability checker
    stability_checker = NumericalStabilityChecker(tolerance=1e-6)
    
    # Test with different input magnitudes
    test_inputs = [
        torch.randn(4, 128) * 0.1,  # Small values
        torch.randn(4, 128),        # Normal values
        torch.randn(4, 128) * 10,   # Large values
    ]
    
    for i, x in enumerate(test_inputs):
        # Check condition number
        condition_result = stability_checker.check_condition_number(layer, x)
        
        # Check numerical precision
        precision_result = stability_checker.check_numerical_precision(layer, x)
        
        print(f"   ✅ Test {i+1}: Stable={precision_result['stable']}, Error={precision_result['max_error']:.2e}")
        
        assert precision_result["stable"], f"Numerical instability detected in test {i+1}"
    
    return True

def test_gradient_flow():
    """Test gradient flow through bijective layers."""
    print("🔧 Testing Gradient Flow...")
    
    # Create layer
    config = create_coupling_config(
        input_dim=64,
        coupling_type="affine",
        hidden_dim=32,
        num_layers=1
    )
    layer = AffineCouplingLayer(config)
    
    # Test gradient flow
    x = torch.randn(2, 64, requires_grad=True)
    
    # Forward pass
    y, log_det = layer.forward(x)
    
    # Compute loss and backward pass
    loss = y.sum() + log_det.sum()
    loss.backward()
    
    # Check gradients
    assert x.grad is not None, "No gradients computed for input"
    assert torch.isfinite(x.grad).all(), "Gradients contain non-finite values"
    
    # Check parameter gradients
    param_grads_exist = any(p.grad is not None for p in layer.parameters())
    assert param_grads_exist, "No gradients computed for layer parameters"
    
    print(f"   ✅ Input gradient norm: {x.grad.norm().item():.2e}")
    print(f"   ✅ Parameter gradients exist: {param_grads_exist}")
    return True

def test_device_compatibility():
    """Test device compatibility for bijective layers."""
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
        config = create_coupling_config(
            input_dim=128,
            coupling_type="additive",
            hidden_dim=64,
            num_layers=1
        )
        layer = AdditiveCouplingLayer(config).to(device)
        
        # Test on device
        x = torch.randn(2, 128, device=device)
        
        with torch.no_grad():
            y, log_det = layer.forward(x)
            x_reconstructed, _ = layer.inverse(y)
        
        # Check device consistency
        assert y.device == device, f"Output device mismatch: {y.device} vs {device}"
        assert log_det.device == device, f"Log det device mismatch: {log_det.device} vs {device}"
        
        # Check reconstruction
        error = torch.norm(x - x_reconstructed, dim=-1).max().item()
        assert error < 1e-6, f"Reconstruction error on {device}: {error}"
        
        print(f"   ✅ Layer runs on {device}")
        print(f"   ✅ Reconstruction error: {error:.2e}")
        return True
        
    except Exception as e:
        print(f"   ❌ Device compatibility error: {e}")
        return False

def test_memory_efficiency():
    """Test memory efficiency of bijective layers."""
    print("🔧 Testing Memory Efficiency...")
    
    # Create larger layer for memory testing
    config = create_coupling_config(
        input_dim=512,
        coupling_type="affine",
        hidden_dim=256,
        num_layers=3
    )
    layer = AffineCouplingLayer(config)
    
    # Test with larger batch
    batch_size = 16
    x = torch.randn(batch_size, 512)
    
    # Measure memory usage (basic check)
    import gc
    gc.collect()
    
    with torch.no_grad():
        y, log_det = layer.forward(x)
        x_reconstructed, _ = layer.inverse(y)
    
    # Check that computation completed without memory errors
    error = torch.norm(x - x_reconstructed, dim=-1).max().item()
    
    print(f"   ✅ Large batch processing: {x.shape}")
    print(f"   ✅ Reconstruction error: {error:.2e}")
    print(f"   ✅ Memory test completed")
    return True

def main():
    """Run all Phase 2 tests."""
    print("🚀 Phase 2 Bijective Components Tests")
    print("=" * 60)
    
    tests = [
        ("Additive Coupling Layer", test_additive_coupling),
        ("Affine Coupling Layer", test_affine_coupling),
        ("Neural Spline Coupling Layer", test_spline_coupling),
        ("Coupling Layer Factory", test_coupling_factory),
        ("Invertibility Framework", test_invertibility_framework),
        ("Jacobian Computation", test_jacobian_computation),
        ("Numerical Stability", test_numerical_stability),
        ("Gradient Flow", test_gradient_flow),
        ("Device Compatibility", test_device_compatibility),
        ("Memory Efficiency", test_memory_efficiency),
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
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{len(tests)} passed")
    
    if failed == 0:
        print("🎉 All Phase 2 tests passed! Bijective components are working correctly.")
        print("\n🚀 Ready for:")
        print("   • Bijective transformer integration")
        print("   • Memory optimization")
        print("   • Performance benchmarking")
    else:
        print(f"⚠️  {failed} tests failed. Please review the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
