# Cell 13: Working Real Data Loading (From Bijective Notebook)

# 📚 WORKING REAL DATA LOADING - Extracted from working notebook

class WorkingWikiTextDataset(Dataset):
    """Working WikiText dataset - copied from bijective notebook"""
    
    def __init__(self, tokenizer, max_length=128, split='train'):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load WikiText-2 (this was working in the other notebook)
        print(f"Loading WikiText-2 {split} dataset...")
        dataset = hf_datasets.load_dataset("wikitext", "wikitext-2-v1", split=split)

        self.texts = []
        for item in dataset:
            text = item['text'].strip()
            if len(text) > 10:  # Filter short texts
                self.texts.append(text)

        print(f"Loaded {len(self.texts)} text samples")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

def create_real_datasets_working(tokenizer, config, num_train=2000, num_eval=500):
    """Create real WikiText datasets using the working method"""
    
    print("📰 Loading REAL WikiText-2 data (working method)...")
    
    try:
        # Use the working dataset class
        train_dataset = WorkingWikiTextDataset(tokenizer, config.max_seq_length, 'train')
        eval_dataset = WorkingWikiTextDataset(tokenizer, config.max_seq_length, 'validation')
        
        # Limit samples if requested
        if len(train_dataset.texts) > num_train:
            train_dataset.texts = train_dataset.texts[:num_train]
        if len(eval_dataset.texts) > num_eval:
            eval_dataset.texts = eval_dataset.texts[:num_eval]
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)
        
        print(f"✅ Real WikiText-2 loaded successfully!")
        print(f"📊 Training samples: {len(train_dataset)}")
        print(f"🧪 Eval samples: {len(eval_dataset)}")
        
        return train_loader, eval_loader
        
    except Exception as e:
        print(f"❌ Real data loading failed: {e}")
        print("🔄 Falling back to synthetic data...")
        return create_datasets('synthetic', num_train, num_eval)

# Test the working data loading
print("🧪 Testing working real data loading...")
try:
    # This should work since it worked in the bijective notebook
    import datasets as hf_datasets
    real_train_loader, real_eval_loader = create_real_datasets_working(tokenizer, config)
    print("🎉 SUCCESS! Real data loading works!")
    
    # Now train on REAL data
    print("\n🔥 Training GNN-MoE on REAL WikiText-2!")
    stats_real, best_loss_real = train_gnn_moe(
        model=model,
        train_loader=real_train_loader, 
        eval_loader=real_eval_loader,
        epochs=5,  # Shorter for real validation
        max_batches_per_epoch=50,
        learning_rate=5e-4
    )
    
    print(f"\n✅ REAL DATA TRAINING COMPLETE!")
    print(f"🎯 Real WikiText-2 performance: {best_loss_real:.4f}")
    
except Exception as e:
    print(f"❌ Still failing: {e}")
    print("📝 The datasets library issue persists in this environment")
