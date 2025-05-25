"""
Real WikiText-2 dataset implementation for bijective discrete diffusion.
Replaces placeholder with actual HuggingFace dataset loading.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Optional, Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)


class WikiTextDataset(Dataset):
    """Real WikiText-2 dataset with proper tokenization and processing."""
    
    def __init__(
        self,
        split: str = "train",
        tokenizer_name: str = "gpt2",
        max_length: int = 512,
        min_length: int = 10,
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Initialize WikiText-2 dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            tokenizer_name: HuggingFace tokenizer to use
            max_length: Maximum sequence length
            min_length: Minimum sequence length (filter shorter)
            cache_dir: Directory to cache processed data
            use_cache: Whether to use cached data
        """
        self.split = split
        self.max_length = max_length
        self.min_length = min_length
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset
        logger.info(f"Loading WikiText-2 dataset, split: {split}")
        try:
            dataset = load_dataset(
                "wikitext", 
                "wikitext-2-raw-v1",
                split=split,
                cache_dir=cache_dir
            )
        except Exception as e:
            logger.error(f"Failed to load WikiText-2: {e}")
            raise
        
        # Process and tokenize
        logger.info("Processing and tokenizing dataset...")
        self.processed_data = self._process_dataset(dataset, use_cache)
        
        logger.info(f"Dataset ready: {len(self.processed_data)} samples")
    
    def _process_dataset(self, dataset, use_cache: bool) -> List[Dict[str, torch.Tensor]]:
        """Process raw dataset into tokenized sequences."""
        processed = []
        
        for example in dataset:
            text = example["text"].strip()
            
            # Skip empty or very short texts
            if len(text) < self.min_length:
                continue
            
            # Skip section headers (WikiText-2 specific)
            if text.startswith("=") and text.endswith("="):
                continue
            
            # Tokenize
            try:
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                input_ids = encoded["input_ids"].squeeze(0)
                attention_mask = encoded["attention_mask"].squeeze(0)
                
                # Check if sequence has enough non-padding tokens
                non_pad_tokens = attention_mask.sum().item()
                if non_pad_tokens >= self.min_length:
                    processed.append({
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "text": text  # Keep original for debugging
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to tokenize text: {e}")
                continue
        
        return processed
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.processed_data[idx]
    
    def get_vocab_size(self) -> int:
        """Get tokenizer vocabulary size."""
        return len(self.tokenizer)
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class WikiTextDataModule:
    """Data module for WikiText-2 with proper configuration loading."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data module with configuration.
        
        Args:
            config: Configuration dictionary from YAML
        """
        self.config = config
        self.tokenizer_name = config.get("tokenizer_name", "gpt2")
        self.max_length = config.get("max_length", 512)
        self.batch_size = config.get("batch_size", 8)
        self.eval_batch_size = config.get("eval_batch_size", 16)
        self.num_workers = config.get("num_workers", 4)
        self.pin_memory = config.get("pin_memory", True)
        
        # Preprocessing config
        preprocessing = config.get("preprocessing", {})
        self.min_length = preprocessing.get("min_length", 10)
        
        # Cache settings
        self.cache_dir = config.get("cache_dir", "data/cache")
        self.use_cache = config.get("use_cache", True)
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup all datasets."""
        logger.info("Setting up WikiText-2 datasets...")
        
        # Create datasets
        self.train_dataset = WikiTextDataset(
            split="train",
            tokenizer_name=self.tokenizer_name,
            max_length=self.max_length,
            min_length=self.min_length,
            cache_dir=self.cache_dir,
            use_cache=self.use_cache
        )
        
        self.val_dataset = WikiTextDataset(
            split="validation",
            tokenizer_name=self.tokenizer_name,
            max_length=self.max_length,
            min_length=self.min_length,
            cache_dir=self.cache_dir,
            use_cache=self.use_cache
        )
        
        self.test_dataset = WikiTextDataset(
            split="test",
            tokenizer_name=self.tokenizer_name,
            max_length=self.max_length,
            min_length=self.min_length,
            cache_dir=self.cache_dir,
            use_cache=self.use_cache
        )
        
        logger.info("Dataset setup complete!")
        logger.info(f"Train: {len(self.train_dataset)} samples")
        logger.info(f"Validation: {len(self.val_dataset)} samples")
        logger.info(f"Test: {len(self.test_dataset)} samples")
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Must call setup() first")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.config.get("shuffle_train", True),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.config.get("drop_last", True)
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("Must call setup() first")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=self.config.get("shuffle_eval", False),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("Must call setup() first")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size from tokenizer."""
        if self.train_dataset is None:
            # Create temporary dataset to get vocab size
            temp_dataset = WikiTextDataset(
                split="train",
                tokenizer_name=self.tokenizer_name,
                max_length=10  # Small for speed
            )
            return temp_dataset.get_vocab_size()
        return self.train_dataset.get_vocab_size()


def create_wikitext_datamodule(config_path: str) -> WikiTextDataModule:
    """
    Create WikiText data module from configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configured WikiTextDataModule
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return WikiTextDataModule(config)


# Test function for validation
def test_wikitext_dataset():
    """Test WikiText dataset loading and processing."""
    print("🔧 Testing WikiText-2 Dataset Loading...")
    
    try:
        # Test dataset creation
        dataset = WikiTextDataset(
            split="validation",  # Use validation for faster testing
            max_length=128,      # Smaller for testing
            min_length=10
        )
        
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        print(f"✅ Vocab size: {dataset.get_vocab_size()}")
        
        # Test sample access
        sample = dataset[0]
        print(f"✅ Sample shape: {sample['input_ids'].shape}")
        print(f"✅ Attention mask shape: {sample['attention_mask'].shape}")
        
        # Test decoding
        decoded = dataset.decode(sample['input_ids'])
        print(f"✅ Sample text: {decoded[:100]}...")
        
        # Test dataloader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))
        print(f"✅ Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"✅ Batch attention_mask shape: {batch['attention_mask'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test
    success = test_wikitext_dataset()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: WikiText-2 dataset test")
