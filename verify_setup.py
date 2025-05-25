#!/usr/bin/env python3
"""
Verification script for Bijective Transformers environment setup.
Run this after setting up the environment to ensure everything is working correctly.
"""

import sys
import importlib
from typing import List, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires Python 3.8+)")
        return False

def check_imports() -> List[Tuple[str, bool, str]]:
    """Check if all required packages can be imported."""
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("torchaudio", "TorchAudio"),
        ("transformers", "HuggingFace Transformers"),
        ("tokenizers", "HuggingFace Tokenizers"),
        ("datasets", "HuggingFace Datasets"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("einops", "Einops"),
        ("wandb", "Weights & Biases"),
        ("nflows", "Normalizing Flows"),
        ("hydra", "Hydra"),
        ("omegaconf", "OmegaConf"),
        ("rich", "Rich"),
        ("pytest", "PyTest"),
        ("jupyter", "Jupyter"),
        ("black", "Black"),
    ]
    
    results = []
    for package, name in packages:
        try:
            importlib.import_module(package)
            results.append((name, True, ""))
            print(f"✅ {name}")
        except ImportError as e:
            results.append((name, False, str(e)))
            print(f"❌ {name}: {e}")
    
    return results

def check_pytorch_setup() -> bool:
    """Check PyTorch configuration and MPS availability."""
    try:
        import torch
        print(f"\n🔥 PyTorch Configuration:")
        print(f"   Version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        # Check MPS (Metal Performance Shaders) for M1/M2/M3 Macs
        if hasattr(torch.backends, 'mps'):
            mps_available = torch.backends.mps.is_available()
            mps_built = torch.backends.mps.is_built()
            print(f"   MPS available: {mps_available}")
            print(f"   MPS built: {mps_built}")
            
            if mps_available:
                try:
                    # Test MPS device
                    device = torch.device("mps")
                    x = torch.randn(10, 10).to(device)
                    y = torch.randn(10, 10).to(device)
                    z = torch.mm(x, y)
                    print(f"   ✅ MPS device test passed")
                    return True
                except Exception as e:
                    print(f"   ❌ MPS device test failed: {e}")
                    return False
            else:
                print(f"   ⚠️  MPS not available (this is normal on non-Apple Silicon)")
                return True
        else:
            print(f"   ⚠️  MPS backend not found (older PyTorch version)")
            return True
            
    except Exception as e:
        print(f"❌ PyTorch setup check failed: {e}")
        return False

def check_transformers_setup() -> bool:
    """Check HuggingFace transformers setup."""
    try:
        from transformers import AutoTokenizer, AutoModel
        print(f"\n🤗 HuggingFace Transformers:")
        
        # Test loading a small model
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        
        # Test tokenization and forward pass
        text = "Hello, world!"
        inputs = tokenizer(text, return_tensors="pt")
        
        # Use no_grad to avoid potential device issues
        import torch
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"   ✅ Model loading and inference test passed")
        print(f"   ✅ Output shape: {outputs.last_hidden_state.shape}")
        return True
        
    except Exception as e:
        print(f"   ❌ Transformers test failed: {e}")
        print(f"   ℹ️  This may be due to version compatibility - core functionality should still work")
        return False

def check_bijective_libraries() -> bool:
    """Check bijective/flow-specific libraries."""
    try:
        import nflows
        print(f"\n🔄 Bijective Libraries:")
        
        # Try to get version, but don't fail if it doesn't exist
        try:
            version = nflows.__version__
            print(f"   nflows version: {version}")
        except AttributeError:
            print(f"   nflows imported successfully (version info not available)")
        
        # Test basic flow creation
        from nflows.flows.base import Flow
        from nflows.distributions.normal import StandardNormal
        from nflows.transforms.coupling import AdditiveCouplingTransform
        from nflows.transforms.base import CompositeTransform
        from nflows.nn.nets import ResidualNet
        
        # Create a simple flow
        base_dist = StandardNormal(shape=[2])
        transform = AdditiveCouplingTransform(
            mask=[1, 0],
            transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                in_features, out_features, hidden_features=16, num_blocks=2
            )
        )
        flow = Flow(transform, base_dist)
        
        print(f"   ✅ Flow creation test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Bijective libraries test failed: {e}")
        print(f"   ℹ️  This may be due to version compatibility - try updating nflows")
        return False

def check_environment_variables() -> bool:
    """Check if environment variables are set correctly."""
    import os
    print(f"\n🌍 Environment Variables:")
    
    env_vars = [
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    
    all_set = True
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"   ✅ {var}={value}")
        else:
            print(f"   ⚠️  {var} not set")
            all_set = False
    
    return all_set

def main():
    """Run all verification checks."""
    print("🔍 Bijective Transformers Environment Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Package Imports", lambda: all(result[1] for result in check_imports())),
        ("PyTorch Setup", check_pytorch_setup),
        ("Transformers Setup", check_transformers_setup),
        ("Bijective Libraries", check_bijective_libraries),
        ("Environment Variables", check_environment_variables),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        print("\n🎉 Environment setup is complete and working correctly!")
        print("You're ready to start Phase 1 implementation.")
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        print("You may need to reinstall the environment or check your setup.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
