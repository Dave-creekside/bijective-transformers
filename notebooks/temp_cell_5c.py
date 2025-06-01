# Cell 5c: Robust SST-2 Data Loading Attempt
print("Trying SST-2 dataset (often more robust)...")

try:
    from transformers import AutoTokenizer
    import datasets as hf_datasets # Ensure this is imported

    # Ensure tokenizer is loaded (it should be from previous cells, but just in case)
    if 'tokenizer' not in globals() or tokenizer.vocab_size != 50257:
        print("Re-loading GPT-2 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        # Assuming 'config' is defined from a previous cell
        # If not, you might need to define it or pass vocab_size directly
        # For now, let's assume config exists and has vocab_size attribute
        # config.vocab_size = tokenizer.vocab_size 

    # Load SST-2 dataset
    dataset_sst2 = hf_datasets.load_dataset("sst2", split="train")
    
    # Process texts (SST-2 has 'sentence' and 'label' columns)
    texts_sst2 = [item['sentence'].strip() for item in dataset_sst2 if len(item['sentence'].strip()) > 10]
    
    # Limit sample size for quick testing
    train_texts_sst2 = texts_sst2[:2000]
    eval_texts_sst2 = texts_sst2[2000:2500] 
    
    if len(train_texts_sst2) < 100 or len(eval_texts_sst2) < 50:
        raise ValueError(f"SST-2 dataset too small after filtering. Train: {len(train_texts_sst2)}, Eval: {len(eval_texts_sst2)}")

    # Assuming SimpleTextDataset and config are defined from previous cells
    # If not, these definitions would need to be included or passed
    # For now, let's assume they exist
    # train_dataset = SimpleTextDataset(train_texts_sst2, tokenizer, config.max_seq_length)
    # eval_dataset = SimpleTextDataset(eval_texts_sst2, tokenizer, config.max_seq_length)
    
    # For standalone execution, let's define them here if not present
    # This part is tricky as it depends on the notebook's state.
    # For now, we'll just print success if loading works, and you can integrate it.
    
    print("✅ SUCCESS: Real SST-2 data loaded!")
    print(f"📊 Train samples: {len(train_texts_sst2)}") # Using raw list length for now
    print(f"🧪 Eval samples: {len(eval_texts_sst2)}")   # Using raw list length for now
    DATA_MODE = "REAL_SST2"
    
except Exception as e:
    print(f"❌ SST-2 loading failed: {e}")
    print("⚠️ This might still indicate an environment issue or a different problem.")
    DATA_MODE = "ERROR_LOADING_REAL_DATA"

print(f"\n✅ DATA LOADING ATTEMPT COMPLETE!")
print(f"🎯 Mode: {DATA_MODE}")

# The following lines depend on train_dataset, eval_dataset, tokenizer being fully set up.
# If DATA_MODE is REAL_SST2, you'd then integrate these texts into your SimpleTextDataset
# and create DataLoaders as before.

# For example, if successful, you would then do:
# train_dataset = SimpleTextDataset(train_texts_sst2, tokenizer, config.max_seq_length)
# eval_dataset = SimpleTextDataset(eval_texts_sst2, tokenizer, config.max_seq_length)
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)
# print(f"📊 Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")
# print(f"🔤 Vocabulary: {tokenizer.vocab_size:,} tokens")
# print("🚀 Ready for training!")
