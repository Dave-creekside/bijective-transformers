# Cell 5: WORKING DATA LOADING - Based on tested successful approach

# 📚 REAL DATA LOADING - Tested and verified working approach

import torch
from torch.utils.data import Dataset, DataLoader
import random

class UltraSimpleDataset(Dataset):
    """Synthetic data for testing"""
    def __init__(self, vocab_size=5000, max_length=128, num_samples=2000):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        print(f"🎲 Generating {num_samples} synthetic samples...")
        self.sequences = []
        for _ in range(num_samples):
            seq_len = random.randint(10, max_length - 2)
            sequence = [1] + [random.randint(2, vocab_size-3) for _ in range(seq_len-2)] + [2]
            while len(sequence) < max_length:
                sequence.append(0)
            self.sequences.append(sequence)
        print(f"✅ Generated {len(self.sequences)} synthetic sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_ids = torch.tensor(sequence, dtype=torch.long)
        attention_mask = (input_ids != 0).long()
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

class RealTextDataset(Dataset):
    """Real text dataset with tokenizer"""
    def __init__(self, tokenizer, texts, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        print(f"📝 Processing {len(texts)} real text samples...")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, max_length=self.max_length,
                                padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()}

def load_real_wikitext_data():
    """Load real WikiText-2 data using the tested working approach"""
    
    print("🚀 LOADING REAL WIKITEXT-2 DATA...")
    print("📋 Using tested working approach from script validation")
    
    try:
        # Fresh imports to avoid notebook state issues
        import datasets as hf_datasets
        from transformers import AutoTokenizer
        
        print("✅ Libraries imported successfully")
        
        # Load WikiText-2 dataset
        print("📦 Loading WikiText-2 dataset...")
        dataset = hf_datasets.load_dataset("wikitext", "wikitext-2-v1")
        print("✅ Dataset loaded successfully!")
        
        # Extract training texts
        print("📝 Extracting training texts...")
        train_texts = [item['text'].strip() for item in dataset['train'] 
                      if len(item['text'].strip()) > 30][:2000]
        print(f"✅ Extracted {len(train_texts)} training texts")
        
        # Extract eval texts  
        print("📝 Extracting eval texts...")
        eval_texts = [item['text'].strip() for item in dataset['validation'] 
                     if len(item['text'].strip()) > 30][:500]
        print(f"✅ Extracted {len(eval_texts)} eval texts")
        
        # Setup tokenizer
        print("🔧 Setting up GPT-2 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Tokenizer ready, vocab_size: {len(tokenizer):,}")
        
        # Create datasets
        print("🏗️ Creating PyTorch datasets...")
        train_dataset = RealTextDataset(tokenizer, train_texts, config.max_seq_length)
        eval_dataset = RealTextDataset(tokenizer, eval_texts, config.max_seq_length)
        
        # Create data loaders
        print("📊 Creating data loaders...")
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)
        
        print("🎉 SUCCESS! REAL WIKITEXT-2 DATA READY!")
        print(f"📊 Training batches: {len(train_loader)}")
        print(f"🧪 Eval batches: {len(eval_loader)}")
        
        return train_loader, eval_loader, tokenizer, "REAL_WIKITEXT2"
        
    except Exception as e:
        print(f"❌ Real data loading failed: {e}")
        print("🔄 Falling back to synthetic data...")
        return None, None, None, "FAILED"

def create_synthetic_fallback():
    """Create synthetic data as fallback"""
    print("🎲 Creating synthetic fallback data...")
    
    train_dataset = UltraSimpleDataset(config.vocab_size, config.max_seq_length, 2000)
    eval_dataset = UltraSimpleDataset(config.vocab_size, config.max_seq_length, 500)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)
    
    print("✅ Synthetic fallback ready!")
    return train_loader, eval_loader, None, "SYNTHETIC_FALLBACK"

# ATTEMPT REAL DATA LOADING
print("🎯 ATTEMPTING REAL DATA LOADING...")
print("="*50)

# Try to load real data first
train_loader, eval_loader, tokenizer, data_mode = load_real_wikitext_data()

# If real data failed, use synthetic fallback
if data_mode == "FAILED":
    train_loader, eval_loader, tokenizer, data_mode = create_synthetic_fallback()

# Update config if we got a real tokenizer
if tokenizer is not None and data_mode == "REAL_WIKITEXT2":
    print("🔧 Updating config for real data...")
    old_vocab = config.vocab_size
    config.vocab_size = len(tokenizer)
    print(f"   Vocab size: {old_vocab:,} → {config.vocab_size:,}")

print(f"\n✅ DATA LOADING COMPLETE!")
print(f"🎯 Mode: {data_mode}")
print(f"📊 Train batches: {len(train_loader)}")
print(f"🧪 Eval batches: {len(eval_loader)}")

if data_mode == "REAL_WIKITEXT2":
    print("🎉 SUCCESS: Using real WikiText-2 data!")
    print("🔬 Ready for serious language modeling validation")
else:
    print("⚠️ Using synthetic fallback data")
    print("🔬 Good for architecture testing, limited for language validation")

print("\n🚀 READY FOR TRAINING!")
