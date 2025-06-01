# colab_gnn_debug_wikitext.py

print("===== Colab GNN Debug WikiText Script Started =====")
import os
import sys
import subprocess

# --- Helper function to print library versions ---
def print_library_versions():
    try:
        import datasets
        print(f"Found datasets version: {datasets.__version__}")
    except ImportError:
        print("datasets library not found.")
    
    try:
        import transformers
        print(f"Found transformers version: {transformers.__version__}")
    except ImportError:
        print("transformers library not found.")

    try:
        import huggingface_hub
        print(f"Found huggingface_hub version: {huggingface_hub.__version__}")
    except ImportError:
        print("huggingface_hub library not found.")

    try:
        import fsspec
        print(f"Found fsspec version: {fsspec.__version__}")
    except ImportError:
        print("fsspec library not found.")
    
    try:
        import tokenizers
        print(f"Found tokenizers version: {tokenizers.__version__}")
    except ImportError:
        print("tokenizers library not found.")


# --- 1. Environment Info ---
print("\n--- Step 1: Current Library Versions ---")
print_library_versions()

# --- 2. Cache Clearing (Pythonic way) ---
print("\n--- Step 2: Attempting to clear Hugging Face datasets cache ---")
try:
    hf_cache_home = os.path.expanduser("~/.cache/huggingface")
    datasets_cache_dir = os.path.join(hf_cache_home, "datasets")
    
    if os.path.exists(datasets_cache_dir):
        print(f"Cache directory found at {datasets_cache_dir}. Attempting to remove...")
        import shutil
        shutil.rmtree(datasets_cache_dir)
        print(f"✅ Successfully removed {datasets_cache_dir}")
    else:
        print(f"Cache directory {datasets_cache_dir} not found. No need to clear.")
except Exception as e:
    print(f"⚠️ Error clearing cache: {e}. This might be okay if permissions are an issue but dataset loads.")

# --- 3. Attempt to Load WikiText-2-v1 ---
print("\n--- Step 3: Attempting to load wikitext-2-v1 ---")
data_loaded_successfully = False
try:
    # Re-import after potential cache clearing and to ensure fresh state
    import datasets as hf_datasets 
    # from transformers import AutoTokenizer # Not strictly needed for just loading dataset structure

    print(f"Attempting load with: datasets.load_dataset('wikitext', 'wikitext-2-v1')")
    print(f"Using datasets version (re-check): {hf_datasets.__version__}") # Good to confirm

    # Minimal load just to check if the dataset itself can be accessed and parsed
    dataset_train = hf_datasets.load_dataset(
        "wikitext", 
        "wikitext-2-v1", 
        split="train"
        # trust_remote_code=True # Removed as datasets==2.14.7 doesn't support it here
    )
    dataset_val = hf_datasets.load_dataset(
        "wikitext", 
        "wikitext-2-v1", 
        split="validation"
        # trust_remote_code=True # Removed
    )
    
    print(f"✅ SUCCESS: WikiText-2-v1 loaded!")
    print(f"Train dataset features: {dataset_train.features}")
    print(f"Number of train examples: {len(dataset_train)}")
    print(f"Number of validation examples: {len(dataset_val)}")
    
    # Simple check of data
    if len(dataset_train) > 0:
        print("Sample train entry (first 200 chars):", dataset_train[0]['text'][:200])
    else:
        print("Train dataset is empty after loading.")
    data_loaded_successfully = True

except Exception as e:
    print(f"❌ FAILED to load WikiText-2-v1.")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Script Summary ---")
if data_loaded_successfully:
    print("🎉 WikiText-2-v1 data loading was SUCCESSFUL in this environment.")
else:
    print("🔴 WikiText-2-v1 data loading FAILED. Review error messages and library versions.")
    print("\nRecommendations for Colab (run in SEPARATE Colab cells BEFORE this script):")
    print("1. Clear previous installs and cache:")
    print("   !pip uninstall -y datasets fsspec huggingface_hub transformers tokenizers")
    print("   !rm -rf ~/.cache/huggingface/datasets")
    print("2. Option A (Try specific older, often stable, set):")
    print("   !pip install datasets==2.14.7 fsspec==2023.10.0 huggingface_hub==0.17.3 transformers==4.35.2 tokenizers==0.15.0")
    print("   (Restart runtime after this cell)")
    print("3. Option B (Try latest stable versions):")
    print("   !pip install --upgrade datasets transformers huggingface_hub fsspec tokenizers")
    print("   (Restart runtime after this cell)")
    print("4. After restarting runtime, re-run this debug script: !python colab_gnn_debug_wikitext.py")

print("\n===== Colab GNN Debug WikiText Script Finished =====")
