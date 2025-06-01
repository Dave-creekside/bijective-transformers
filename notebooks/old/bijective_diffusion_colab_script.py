# 🚀 Bijective Discrete Diffusion for Text Generation - Colab Script
# This script contains all the code for the Bijective Discrete Diffusion model,
# designed to be easily runnable in a Colab environment or any Python environment
# with the necessary packages installed.

# ===========================================================================
# SETUP & IMPORTS
# ===========================================================================
print("📦 Installing required packages...")
# In a Colab notebook, you would run these with !pip
# For a local script, ensure these are installed in your environment.
# !pip install torch transformers datasets tqdm matplotlib --quiet
# !pip install --upgrade datasets==2.15.0 transformers==4.35.0 fsspec==2023.10.0 accelerate==0.25.0 --quiet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import datasets as hf_datasets
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import math
import time
from tqdm import tqdm
import json
import os
import shutil # For cache clearing

# Set HuggingFace cache directories to temporary paths for Colab
# This should be done before any HuggingFace library call
os.environ['HF_HOME'] = '/tmp/hf_cache_home'
os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_datasets_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
print(f"HF_DATASETS_CACHE set to: {os.environ['HF_DATASETS_CACHE']}")

# Optionally, clear cache if it exists (useful for repeated runs in same session)
# if os.path.exists(os.environ['HF_DATASETS_CACHE']):
#     print(f"Clearing existing cache at {os.environ['HF_DATASETS_CACHE']}")
#     shutil.rmtree(os.environ['HF_DATASETS_CACHE'])
# os.makedirs(os.environ['HF_DATASETS_CACHE'], exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")
print("✅ Setup complete!")

# ===========================================================================
# CONFIGURATION AND CORE MODEL IMPLEMENTATION
# ===========================================================================
print("🔧 Defining Configuration and Core Model Implementation...")

@dataclass
class Config:
    model_size: str = "small"  # Options: "small", "base", "large", "custom"
    vocab_size: int = 50257
    max_seq_length: int = 64
    embed_dim: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    dropout: float = 0.1

    def __post_init__(self):
        if self.model_size == "small":
            self.embed_dim = self.embed_dim if self.embed_dim is not None else 128
            self.num_layers = self.num_layers if self.num_layers is not None else 2
            self.num_heads = self.num_heads if self.num_heads is not None else 4
        elif self.model_size == "base":
            self.embed_dim = self.embed_dim if self.embed_dim is not None else 256
            self.num_layers = self.num_layers if self.num_layers is not None else 4
            self.num_heads = self.num_heads if self.num_heads is not None else 8
        elif self.model_size == "large":
            self.embed_dim = self.embed_dim if self.embed_dim is not None else 512
            self.num_layers = self.num_layers if self.num_layers is not None else 6
            self.num_heads = self.num_heads if self.num_heads is not None else 8
        elif self.model_size == "custom":
            if None in [self.embed_dim, self.num_layers, self.num_heads]:
                raise ValueError("For 'custom' model_size, embed_dim, num_layers, and num_heads must be specified.")
        else: 
            print(f"Warning: Unknown model_size '{self.model_size}'. Defaulting to 'small'.")
            self.model_size = "small"
            self.__post_init__()

class CouplingFunction(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, x): return self.net(x)

class InvertibleResidual(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.split = dim // 2
        self.F = CouplingFunction(dim - self.split, self.split)
        self.G = CouplingFunction(self.split, dim - self.split)
    def forward(self, x):
        x1, x2 = x[..., :self.split], x[..., self.split:]
        y1 = x1 + self.F(x2)
        y2 = x2 + self.G(y1)
        return torch.cat([y1, y2], dim=-1)
    def inverse(self, y):
        y1, y2 = y[..., :self.split], y[..., self.split:]
        x2 = y2 - self.G(y1)
        x1 = y1 - self.F(x2)
        return torch.cat([x1, x2], dim=-1)

class BijectiveAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.q_proj = InvertibleResidual(config.embed_dim)
        self.k_proj = InvertibleResidual(config.embed_dim)
        self.v_proj = InvertibleResidual(config.embed_dim)
        self.out_proj = InvertibleResidual(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.bool()
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

class BijectiveBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attn = BijectiveAttention(config)
        self.ffn = InvertibleResidual(config.embed_dim)
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x, mask=None):
        attn_out = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class BijectiveDiffusionModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.max_seq_length, config.embed_dim)
        self.time_emb = TimeEmbedding(config.embed_dim)
        self.blocks = nn.ModuleList([BijectiveBlock(config) for _ in range(config.num_layers)])
        self.head = nn.Linear(config.embed_dim, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, input_ids, timesteps, attention_mask=None):
        B, L = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        time_emb = self.time_emb(timesteps).unsqueeze(1).expand(-1, L, -1)
        x = x + time_emb
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, attention_mask)
        logits = self.head(x)
        return logits
    def training_step(self, clean_ids, attention_mask=None):
        B = clean_ids.shape[0]
        t = torch.randint(0, 1000, (B,), device=clean_ids.device)
        noise_level = torch.linspace(0.01, 0.99, 1000, device=clean_ids.device)[t]
        mask = torch.rand_like(clean_ids.float()) < noise_level.unsqueeze(1)
        if attention_mask is not None:
            mask = mask & attention_mask.bool()
        noisy_ids = clean_ids.clone()
        noisy_ids[mask] = torch.randint(0, self.config.vocab_size, (mask.sum().item(),), device=clean_ids.device)
        logits = self.forward(noisy_ids, t, attention_mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), clean_ids.view(-1), reduction='mean')
        return {'loss': loss, 'logits': logits}

print("✅ Model implementation complete!")

# ===========================================================================
# DATA LOADING
# ===========================================================================
print("📚 Setting up Data Loading...")

class WikiTextDataset(Dataset):
    def __init__(self, tokenizer, max_length=64, split='train'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        dataset_path = "wikitext"
        dataset_name_config = "wikitext-2-raw-v1" # Standard name for WikiText-2
        print(f"Loading {dataset_path} ({dataset_name_config}) {split} dataset...")
        
        try:
            dataset = hf_datasets.load_dataset(
                dataset_path, 
                name=dataset_name_config, 
                split=split, 
                trust_remote_code=True, 
                download_mode="force_redownload" # Force fresh download
            )
        except Exception as e:
            print(f"Error loading dataset {dataset_path}/{dataset_name_config}: {e}")
            # Fallback to wikitext-103-raw-v1 if wikitext-2-raw-v1 fails
            dataset_name_config_fallback = "wikitext-103-raw-v1"
            print(f"Trying fallback: {dataset_path} ({dataset_name_config_fallback})")
            try:
                 dataset = hf_datasets.load_dataset(
                    dataset_path, 
                    name=dataset_name_config_fallback, 
                    split=split, 
                    trust_remote_code=True,
                    download_mode="force_redownload"
                )
            except Exception as e_fallback:
                print(f"Fallback to {dataset_name_config_fallback} also failed: {e_fallback}")
                raise # Re-raise the last error if all attempts fail
        
        self.texts = [item['text'].strip() for item in dataset if len(item['text'].strip()) > 10]
        print(f"Loaded {len(self.texts)} text samples for {split} split.")
        
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze()}

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

global_config = Config(vocab_size=len(tokenizer), model_size='small') 

train_dataset = WikiTextDataset(tokenizer, global_config.max_seq_length, 'train')
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
print("✅ Data loading complete!")

# ===========================================================================
# LOAD MODEL FOR CONTINUED TRAINING (OPTIONAL)
# ===========================================================================
print("🔧 Configuring Model Loading (Optional)...")

resume_model_path = None
resume_config_path = None
model_to_train = None

if resume_model_path and resume_config_path and os.path.exists(resume_model_path) and os.path.exists(resume_config_path):
    print(f"Attempting to load model from: {resume_model_path}")
    try:
        with open(resume_config_path, 'r') as f: loaded_config_dict = json.load(f)
        if 'model_size' not in loaded_config_dict: loaded_config_dict['model_size'] = 'custom'
        
        global_config = Config(**loaded_config_dict)
        global_config.vocab_size = len(tokenizer)
        print("✅ Config loaded and updated successfully.")

        model_to_train = BijectiveDiffusionModel(global_config).to(device)
        model_to_train.load_state_dict(torch.load(resume_model_path, map_location=device))
        model_to_train.train()
        print(f"✅ Model state_dict loaded successfully.")
    except Exception as e:
        print(f"⚠️ Error loading model/config: {e}. Proceeding with new model.")
        model_to_train = None
else:
    print("No valid model/config path for resumption. Training from scratch.")
    model_to_train = None

# ===========================================================================
# TRAINING
# ===========================================================================
print("🏋️ Starting Training Process...")

if model_to_train is None: 
    print("Initializing new model for training...")
    model_to_train = BijectiveDiffusionModel(global_config).to(device)

optimizer = optim.AdamW(model_to_train.parameters(), lr=1e-4)
print(f"Model parameters: {sum(p.numel() for p in model_to_train.parameters()):,}")
print(f"Training with config: vocab_size={global_config.vocab_size}, embed_dim={global_config.embed_dim}, layers={global_config.num_layers}, heads={global_config.num_heads}")
model_to_train.train()
training_losses = []
num_epochs = 1 
max_batches_per_epoch = 20

for epoch in range(num_epochs):
    epoch_losses = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for i, batch in enumerate(pbar):
        if max_batches_per_epoch is not None and i >= max_batches_per_epoch: break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        optimizer.zero_grad()
        outputs = model_to_train.training_step(input_ids, attention_mask)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
    training_losses.extend(epoch_losses)
    print(f"Epoch {epoch+1}/{num_epochs} average loss: {avg_loss:.4f}")
print("✅ Training complete!")

if training_losses: 
    plt.figure(figsize=(10,6))
    plt.plot(training_losses)
    plt.title('Training Loss Over Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show() 
else: 
    print("No training steps completed.")

# ===========================================================================
# SAVE TRAINED MODEL
# ===========================================================================
print("💾 Saving Trained Model...")
model_save_path = "bijective_diffusion_model.pt"
config_save_path = "bijective_diffusion_config.json"
print(f"Saving model to: {model_save_path}")
torch.save(model_to_train.state_dict(), model_save_path)
model_config_dict = {
    "vocab_size": global_config.vocab_size, "max_seq_length": global_config.max_seq_length,
    "embed_dim": global_config.embed_dim, "num_layers": global_config.num_layers,
    "num_heads": global_config.num_heads, "dropout": global_config.dropout,
    "model_size": global_config.model_size
}
with open(config_save_path, 'w') as f: json.dump(model_config_dict, f, indent=2)
print("✅ Model and config saved!")

# ===========================================================================
# ENHANCED GENERATION DEMO
# ===========================================================================
print("🎯 Running Enhanced Generation Demo...")
def generate_text(model, tokenizer, prompt="The", max_length=32, num_steps=10):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        for _ in range(num_steps):
            current_ids_for_step = input_ids.clone()
            for i in range(input_ids.shape[1] -1, max_length -1): 
                if current_ids_for_step.shape[1] >= max_length: break
                padded_input = current_ids_for_step 
                if current_ids_for_step.shape[1] < max_length:
                     pad_len = max_length - current_ids_for_step.shape[1]
                     padded_input = torch.cat([current_ids_for_step, torch.full((1, pad_len), tokenizer.eos_token_id, device=device, dtype=torch.long)], dim=1)
                
                timesteps = torch.tensor([num_steps - 1 - _], device=device) 
                logits = model(padded_input[:,:max_length], timesteps) 
                
                next_token_logits = logits[0, current_ids_for_step.shape[1]-1] 
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
                current_ids_for_step = torch.cat([current_ids_for_step, next_token.unsqueeze(0)], dim=1)

            input_ids = current_ids_for_step 
            if input_ids.shape[1] >= max_length: break
            
    return tokenizer.decode(input_ids[0, :max_length], skip_special_tokens=True)

test_cases = [
    {"prompt": "The field of natural language processing", "max_length": 40, "num_steps": 10},
    {"prompt": "Once upon a time, in a land far away,", "max_length": 50, "num_steps": 15},
]
for i, case in enumerate(test_cases):
    print(f"--- Test Case {i+1} ---")
    print(f"Prompt: '{case['prompt']}'")
    generated_text = generate_text(model_to_train, tokenizer, **case)
    print(f"Generated: {generated_text}")
print("✅ Enhanced generation demo complete!")

# ===========================================================================
# MODEL ANALYSIS & INVERTIBILITY TEST
# ===========================================================================
print("📊 Running Model Analysis & Invertibility Test...")
def analyze_model(model_instance):
    total_params = sum(p.numel() for p in model_instance.parameters())
    print(f"🔍 Model Analysis: Total parameters: {total_params:,}")
    print(f"   Architecture: embed_dim={model_instance.config.embed_dim}, layers={model_instance.config.num_layers}, heads={model_instance.config.num_heads}")

def test_invertibility():
    print("🔄 Testing Numerical Invertibility of InvertibleResidual:")
    test_dim = global_config.embed_dim 
    invertible_layer = InvertibleResidual(test_dim).to(device)
    
    print("--- Test 1: At Initialization ---")
    x_init = torch.randn(2, 10, test_dim).to(device)
    y_init = invertible_layer.forward(x_init) 
    diff_at_init = torch.norm(y_init - x_init).item()
    print(f"L2 difference (y_init - x_init): {diff_at_init:.6f}")
    if diff_at_init < 1e-5: print("✅ Near-identity at init (expected).")
    else: print(f"⚠️ Non-identity at init (L2 diff: {diff_at_init:.6f}).")

    print("--- Test 2: Numerical Inversion (Simulated Trained Block) ---")
    test_block = InvertibleResidual(test_dim).to(device)
    with torch.no_grad():
        for param in test_block.F.net[-1].parameters(): param.data.uniform_(-0.1, 0.1) 
        for param in test_block.G.net[-1].parameters(): param.data.uniform_(-0.1, 0.1)
    x_test = torch.randn(2, 10, test_dim).to(device)
    y_test = test_block.forward(x_test) 
    diff_forward_modified = torch.norm(y_test - x_test).item()
    print(f"L2 diff (y_test - x_test) with modified block: {diff_forward_modified:.4f}")
    if diff_forward_modified > 1e-5: print("✅ Non-trivial transformation.")
    else: print("⚠️ Still near-identity after modifying weights.")

    if hasattr(test_block, 'inverse'):
        x_reconstructed = test_block.inverse(y_test) 
        reconstruction_error = torch.norm(x_test - x_reconstructed).item()
        print(f"Reconstruction error: {reconstruction_error:.6f}")
        if reconstruction_error < 1e-5: print("✅✅✅ InvertibleResidual is numerically invertible!")
        else: print(f"❌❌❌ High reconstruction error: {reconstruction_error:.6f}")
    else: print("⚠️ InvertibleResidual.inverse() method not found.")

analyze_model(model_to_train)
test_invertibility()
print("🎉 Script complete! Your bijective diffusion model is ready!")
