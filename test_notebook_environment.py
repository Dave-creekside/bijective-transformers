#!/usr/bin/env python3
"""
Test script that mimics the exact notebook environment and calls
"""

def test_notebook_style_loading():
    """Test the exact same approach used in the notebook"""
    print("🧪 Testing exact notebook-style loading...")
    
    # Mimic the notebook imports and setup
    import torch
    from torch.utils.data import Dataset, DataLoader
    import datasets as hf_datasets
    from transformers import AutoTokenizer
    
    print("✅ All imports successful")
    
    # Test the exact dataset loading call from notebook
    try:
        print("📦 Loading WikiText-2 (exact notebook call)...")
        dataset = hf_datasets.load_dataset("wikitext", "wikitext-2-v1")
        print("✅ Dataset loading successful!")
        
        # Test text extraction 
        train_texts = [item['text'].strip() for item in dataset['train'] 
                      if len(item['text'].strip()) > 30][:2000]
        print(f"✅ Extracted {len(train_texts)} training texts")
        
        eval_texts = [item['text'].strip() for item in dataset['validation'] 
                     if len(item['text'].strip()) > 30][:500]
        print(f"✅ Extracted {len(eval_texts)} eval texts")
        
        # Test tokenizer setup
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Tokenizer ready, vocab_size: {len(tokenizer):,}")
        
        # Test dataset creation (minimal)
        print("📝 Testing dataset creation...")
        sample_text = train_texts[0]
        encoding = tokenizer(sample_text, max_length=128, 
                           padding='max_length', truncation=True, return_tensors='pt')
        print("✅ Tokenization works!")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_create_working_data_loader():
    """Create a working data loader using the successful approach"""
    print("\n🏗️ Creating working data loader...")
    
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        import datasets as hf_datasets
        from transformers import AutoTokenizer
        
        # Load dataset
        dataset = hf_datasets.load_dataset("wikitext", "wikitext-2-v1")
        
        # Extract texts
        train_texts = [item['text'].strip() for item in dataset['train'] 
                      if len(item['text'].strip()) > 30][:100]  # Smaller for test
        
        # Setup tokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        # Simple dataset class
        class TestDataset(Dataset):
            def __init__(self, tokenizer, texts, max_length=128):
                self.tokenizer = tokenizer
                self.texts = texts
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = self.texts[idx]
                encoding = self.tokenizer(text, max_length=self.max_length,
                                        padding='max_length', truncation=True, return_tensors='pt')
                return {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze()
                }
        
        # Create dataset and loader
        test_dataset = TestDataset(tokenizer, train_texts)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
        
        # Test a batch
        print("🧪 Testing data loader...")
        for batch in test_loader:
            print(f"✅ Batch loaded: input_ids shape {batch['input_ids'].shape}")
            break
        
        print("🎉 COMPLETE WORKING DATA LOADER CREATED!")
        return test_loader, tokenizer
        
    except Exception as e:
        print(f"❌ Data loader creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    print("🔍 TESTING NOTEBOOK ENVIRONMENT REPLICATION")
    print("="*60)
    
    # Test 1: Notebook-style loading
    success = test_notebook_style_loading()
    
    if success:
        print("\n✅ NOTEBOOK APPROACH WORKS IN SCRIPT!")
        print("The issue must be notebook-specific")
        
        # Test 2: Create working components
        loader, tokenizer = test_create_working_data_loader()
        
        if loader is not None:
            print("\n🎯 SOLUTION FOUND!")
            print("The exact notebook approach works fine in isolation")
            print("Issue is likely import order or environment state in notebook")
        
    else:
        print("\n❌ SAME ERROR IN SCRIPT")
        print("The issue is system-wide")
