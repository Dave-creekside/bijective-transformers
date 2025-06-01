#!/usr/bin/env python3
"""
Debug script to systematically test WikiText-2 loading approaches
"""

import sys
import traceback

def test_approach(name, test_func):
    """Test an approach and report results"""
    print(f"\n{'='*50}")
    print(f"TESTING: {name}")
    print(f"{'='*50}")
    
    try:
        result = test_func()
        print(f"✅ SUCCESS: {name}")
        return result
    except Exception as e:
        print(f"❌ FAILED: {name}")
        print(f"Error: {e}")
        traceback.print_exc()
        return None

def approach_1_basic_import():
    """Test basic datasets import"""
    import datasets as hf_datasets
    print("✅ Basic datasets import successful")
    return True

def approach_2_simple_load():
    """Test simple dataset loading"""
    import datasets as hf_datasets
    dataset = hf_datasets.load_dataset("wikitext", "wikitext-2-v1", split="train[:10]")
    print(f"✅ Loaded {len(dataset)} samples")
    return dataset

def approach_3_with_fsspec_downgrade():
    """Test with fsspec downgrade"""
    import subprocess
    import os
    
    # Try downgrading fsspec
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "fsspec==2023.1.0"
    ], capture_output=True, text=True)
    
    print(f"fsspec downgrade result: {result.returncode}")
    if result.stdout:
        print(f"stdout: {result.stdout}")
    if result.stderr:
        print(f"stderr: {result.stderr}")
    
    # Now try loading
    import datasets as hf_datasets
    dataset = hf_datasets.load_dataset("wikitext", "wikitext-2-v1", split="train[:10]")
    print(f"✅ Loaded {len(dataset)} samples with fsspec downgrade")
    return dataset

def approach_4_alternative_loading():
    """Test alternative loading methods"""
    # Try using different cache directory
    import datasets as hf_datasets
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = hf_datasets.load_dataset(
            "wikitext", 
            "wikitext-2-v1", 
            split="train[:10]",
            cache_dir=cache_dir
        )
        print(f"✅ Loaded {len(dataset)} samples with temp cache")
        return dataset

def approach_5_manual_download():
    """Test manual dataset download approach"""
    import requests
    import json
    
    # Try the HuggingFace datasets API directly
    url = "https://datasets-server.huggingface.co/rows?dataset=wikitext&config=wikitext-2-v1&split=train&offset=0&length=10"
    
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    texts = [row['row']['text'] for row in data['rows']]
    
    print(f"✅ Manual download got {len(texts)} samples")
    return texts

def approach_6_environment_debug():
    """Debug environment information"""
    import datasets
    import fsspec
    import pkg_resources
    
    print(f"datasets version: {datasets.__version__}")
    print(f"fsspec version: {fsspec.__version__}")
    
    # Check for conflicting packages
    installed_packages = [d.project_name for d in pkg_resources.working_set]
    potential_conflicts = ['s3fs', 'gcsfs', 'adlfs', 'fsspec']
    
    print("\nInstalled filesystem packages:")
    for pkg in potential_conflicts:
        if pkg in installed_packages:
            version = pkg_resources.get_distribution(pkg).version
            print(f"  {pkg}: {version}")
    
    return True

def main():
    """Run all tests systematically"""
    print("🔍 SYSTEMATIC DATASETS DEBUGGING")
    print("="*60)
    
    # Test each approach
    approaches = [
        ("Basic Import", approach_1_basic_import),
        ("Simple Load", approach_2_simple_load),
        ("Environment Debug", approach_6_environment_debug),
        ("Manual Download", approach_5_manual_download),
        ("Alternative Loading", approach_4_alternative_loading),
        ("FSSpec Downgrade", approach_3_with_fsspec_downgrade),
    ]
    
    results = {}
    for name, func in approaches:
        results[name] = test_approach(name, func)
    
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*60}")
    
    for name, result in results.items():
        status = "✅ SUCCESS" if result is not None else "❌ FAILED"
        print(f"{name:20} : {status}")
    
    # Find working approaches
    working = [name for name, result in results.items() if result is not None]
    
    if working:
        print(f"\n🎉 WORKING APPROACHES:")
        for approach in working:
            print(f"  ✅ {approach}")
        
        print(f"\n💡 RECOMMENDATION:")
        if "Manual Download" in working:
            print("   Use manual download approach - most reliable")
        elif "Alternative Loading" in working:
            print("   Use alternative loading with custom cache")
        elif "FSSpec Downgrade" in working:
            print("   Use fsspec downgrade approach")
        else:
            print(f"   Use the first working approach: {working[0]}")
    else:
        print(f"\n❌ NO WORKING APPROACHES FOUND")
        print("   This indicates a fundamental environment issue")

if __name__ == "__main__":
    main()
