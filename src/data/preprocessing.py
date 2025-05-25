"""
Text preprocessing utilities for discrete diffusion.
Placeholder implementation for Phase 1 testing.
"""

import torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer


class TextPreprocessor:
    """Text preprocessing for discrete diffusion models."""
    
    def __init__(self, tokenizer_name: str = "gpt2", max_length: int = 512):
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.tokenizer = None
    
    def setup(self):
        """Setup tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Warning: Could not load tokenizer {self.tokenizer_name}: {e}")
            self.tokenizer = None
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess a single text string."""
        if self.tokenizer is None:
            # Fallback for testing
            return {
                "input_ids": torch.randint(0, 1000, (self.max_length,)),
                "attention_mask": torch.ones(self.max_length)
            }
        
        # Tokenize text
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }
    
    def preprocess_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Preprocess a batch of texts."""
        if self.tokenizer is None:
            # Fallback for testing
            batch_size = len(texts)
            return {
                "input_ids": torch.randint(0, 1000, (batch_size, self.max_length)),
                "attention_mask": torch.ones(batch_size, self.max_length)
            }
        
        # Tokenize batch
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }
    
    def decode_tokens(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs back to text."""
        if self.tokenizer is None:
            # Fallback for testing
            return [f"decoded_text_{i}" for i in range(token_ids.shape[0])]
        
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        
        texts = []
        for ids in token_ids:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(text)
        
        return texts
