#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gnn_moe_wikitext_script.py

GNN-Coupled MoE model training on WikiText-2 dataset.
Based on gnn_moe_script.py, modified for WikiText-2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Optional, List, Tuple
from tqdm import tqdm
from collections import defaultdict
import time
import random
import os
import argparse # For command-line arguments
import shutil # For copying best model checkpoint

# --- 1. Initial Setup ---
def setup_environment(seed=42):
    plt.style.use('default')
    sns.set_palette("husl")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Device: Apple MPS")
    else:
        device = torch.device("cpu")
        print("🚀 Device: CPU")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if not os.path.exists("plots"):
        os.makedirs("plots")
        print("📁 Created 'plots' directory for output visualizations.")
    print("✅ Environment ready for GNN-MoE research on WikiText-2!")
    # Create checkpoints directory if it doesn't exist
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
        print("📁 Created 'checkpoints' directory for model saving.")
    return device

# --- Checkpoint Helper Functions ---
def save_checkpoint(state, is_best, checkpoint_dir="checkpoints", filename="checkpoint.pth.tar"):
    """Saves model and training parameters."""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, "best_model.pth.tar"))
        print(f"✅ Saved new best model to {os.path.join(checkpoint_dir, 'best_model.pth.tar')}")
    else:
        print(f"✅ Saved checkpoint to {filepath}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Loads model and training parameters."""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint file not found at '{checkpoint_path}'. Starting from scratch.")
        return None
    
    print(f"🔄 Loading checkpoint from '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False) # Load to CPU first, added weights_only=False
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
    step = checkpoint.get('step', 0)
    
    print(f"✅ Checkpoint loaded. Resuming from epoch {start_epoch}, step {step}, best_eval_loss {best_eval_loss:.4f}")
    return start_epoch, step, best_eval_loss

# --- 2. Configuration ---
@dataclass
class GNNMoEConfig:
    vocab_size: int = 50257
    max_seq_length: int = 128
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    num_experts: int = 4
    gnn_layers: int = 2
    batch_size: int = 32 
    learning_rate: float = 5e-4
    epochs: int = 8 
    max_batches_per_epoch: int = 563 # Approx. 18000 samples / batch_size 32
    eval_every: int = 200 # Evaluate a few times per epoch

# --- 3. Core GNN Architecture (Identical to gnn_moe_script.py) ---
class ExpertGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.neighbor_transform = nn.Linear(in_dim, out_dim)
        self.self_transform = nn.Linear(in_dim, out_dim)
        self.message_weight = nn.Linear(in_dim * 2, 1)
        self.adjacency_logits = nn.Parameter(torch.randn(num_experts, num_experts))

    def forward(self, expert_features):
        batch_size, seq_len, num_experts, embed_dim = expert_features.shape
        adjacency = torch.sigmoid(self.adjacency_logits)
        flat_features = expert_features.view(-1, num_experts, embed_dim)
        updated_features_list = []
        for expert_idx in range(num_experts):
            current_expert = flat_features[:, expert_idx, :]
            messages = []
            for other_idx in range(num_experts):
                if other_idx != expert_idx:
                    other_expert = flat_features[:, other_idx, :]
                    concat_features = torch.cat([current_expert, other_expert], dim=1)
                    content_weight = torch.sigmoid(self.message_weight(concat_features).squeeze(-1))
                    message_strength = adjacency[expert_idx, other_idx] * content_weight
                    weighted_message = other_expert * message_strength.unsqueeze(1)
                    messages.append(weighted_message)
            if messages:
                neighbor_msg = torch.stack(messages, dim=0).sum(dim=0)
                neighbor_out = self.neighbor_transform(neighbor_msg)
            else:
                neighbor_out = torch.zeros_like(self.neighbor_transform(current_expert))
            self_out = self.self_transform(current_expert)
            updated_expert = F.gelu(neighbor_out + self_out)
            updated_features_list.append(updated_expert)
        updated_stack = torch.stack(updated_features_list, dim=1)
        return updated_stack.view(batch_size, seq_len, num_experts, embed_dim)

    def get_adjacency_matrix(self):
        return torch.sigmoid(self.adjacency_logits).detach()

class ExpertBlock(nn.Module):
    def __init__(self, config: GNNMoEConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.embed_dim, config.num_heads,
            dropout=config.dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim * 4, config.embed_dim),
            nn.Dropout(config.dropout)
        )
    def forward(self, x, causal_mask=None, key_padding_mask=None):
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(
            x_norm, x_norm, x_norm,
            attn_mask=causal_mask, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class GNNExpertCoupler(nn.Module):
    def __init__(self, config: GNNMoEConfig):
        super().__init__()
        self.config = config
        self.gnn_layers_module = nn.ModuleList([
            ExpertGraphConv(config.embed_dim, config.embed_dim, config.num_experts)
            for _ in range(config.gnn_layers)
        ])
        self.combiner = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim)
        )
    def forward(self, expert_outputs):
        stacked_experts = torch.stack(expert_outputs, dim=2)
        expert_features = stacked_experts
        for gnn_layer_instance in self.gnn_layers_module:
            new_features = gnn_layer_instance(expert_features)
            expert_features = new_features + expert_features
        coordinated_output = expert_features.mean(dim=2)
        output = self.combiner(coordinated_output)
        return output
    def get_expert_communication_matrices(self):
        matrices = []
        for gnn_layer_instance in self.gnn_layers_module:
            matrices.append(gnn_layer_instance.get_adjacency_matrix())
        return matrices

# --- 4. Complete Model Architecture (Identical) ---
class GNNMoELayer(nn.Module):
    def __init__(self, config: GNNMoEConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([ExpertBlock(config) for _ in range(config.num_experts)])
        self.gnn_coupler = GNNExpertCoupler(config)
    def forward(self, x, causal_mask=None, key_padding_mask=None):
        expert_outputs = [expert(x, causal_mask, key_padding_mask) for expert in self.experts]
        coordinated = self.gnn_coupler(expert_outputs)
        return x + coordinated

class GNNMoEModel(nn.Module):
    def __init__(self, config: GNNMoEConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.max_seq_length, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.model_layers = nn.ModuleList([GNNMoELayer(config) for _ in range(config.num_layers)])
        self.output_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)
        self._init_weights()
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None: torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def create_causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    def forward(self, input_ids, attention_mask=None, return_loss=True, labels=None):
        B, L = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        x = self.dropout(x)
        causal_mask = self.create_causal_mask(L, input_ids.device)
        key_padding_mask = (~attention_mask.bool()) if attention_mask is not None else None
        for layer_instance in self.model_layers:
            x = layer_instance(x, causal_mask, key_padding_mask)
        x = self.output_norm(x)
        logits = self.lm_head(x)
        if return_loss and labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=0)
            return {'loss': loss, 'logits': logits}
        return {'logits': logits}
    def analyze_expert_communication(self):
        comm_data = {}
        for i, layer_instance in enumerate(self.model_layers):
            if hasattr(layer_instance, 'gnn_coupler'):
                 matrices = layer_instance.gnn_coupler.get_expert_communication_matrices()
                 comm_data[f'layer_{i}'] = matrices
        return comm_data

# --- 5. Data Handling (Modified for WikiText-2) ---
class SimpleTextDataset(Dataset): # Identical
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {'input_ids': encoding['input_ids'].squeeze(0), 'attention_mask': encoding['attention_mask'].squeeze(0)}

def load_data(config: GNNMoEConfig, num_train_samples_target=2000, num_eval_samples_target=500):
    print("🚀 Setting up data loading for WikiText-2...")
    try:
        from transformers import AutoTokenizer
        import datasets as hf_datasets
        
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        if config.vocab_size != tokenizer.vocab_size:
            print(f"⚠️ Updating config.vocab_size from {config.vocab_size} to {tokenizer.vocab_size} based on tokenizer.")
            config.vocab_size = tokenizer.vocab_size

        print("📦 Attempting wikitext-2-v1 dataset loading...") # Corrected name
        train_dataset_raw = hf_datasets.load_dataset("wikitext", "wikitext-2-v1", split="train") # trust_remote_code removed
        eval_dataset_raw = hf_datasets.load_dataset("wikitext", "wikitext-2-v1", split="validation") # trust_remote_code removed
        
        train_texts_all = [line.strip() for item in train_dataset_raw for line in item['text'].splitlines() if len(line.strip()) > 30]
        eval_texts_all = [line.strip() for item in eval_dataset_raw for line in item['text'].splitlines() if len(line.strip()) > 30]
        
        print(f"Raw WikiText-2 lines >30 chars: Train {len(train_texts_all)}, Eval {len(eval_texts_all)}")
        random.shuffle(train_texts_all) # Shuffle all available texts
        random.shuffle(eval_texts_all)

        num_train_samples = min(len(train_texts_all), num_train_samples_target)
        num_eval_samples = min(len(eval_texts_all), num_eval_samples_target)

        if num_train_samples < 100 or num_eval_samples < 50: # Basic check
            raise ValueError(f"WikiText-2 dataset too small after filtering. Effective train: {num_train_samples}, eval: {num_eval_samples}")

        train_texts = train_texts_all[:num_train_samples]
        eval_texts = eval_texts_all[:num_eval_samples]

        train_dataset = SimpleTextDataset(train_texts, tokenizer, config.max_seq_length)
        eval_dataset = SimpleTextDataset(eval_texts, tokenizer, config.max_seq_length)
        
        print("✅ SUCCESS: Real wikitext-2-v1 data loaded!") # Corrected name
        data_mode = "REAL_WIKITEXT2"
        
    except Exception as e:
        print(f"⚠️ Real data (WikiText-2) loading failed: {e}")
        print("🔄 Using high-quality synthetic fallback...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        if config.vocab_size != tokenizer.vocab_size:
            print(f"⚠️ Updating config.vocab_size from {config.vocab_size} to {tokenizer.vocab_size} for synthetic data.")
            config.vocab_size = tokenizer.vocab_size
        synthetic_texts = ["The transformer architecture revolutionized natural language processing."] * (num_train_samples_target + num_eval_samples_target)
        train_texts = synthetic_texts[:num_train_samples_target]
        eval_texts = synthetic_texts[num_train_samples_target:]
        train_dataset = SimpleTextDataset(train_texts, tokenizer, config.max_seq_length)
        eval_dataset = SimpleTextDataset(eval_texts, tokenizer, config.max_seq_length)
        print("✅ Synthetic data ready!")
        data_mode = "SYNTHETIC_REALISTIC"

    # Use num_workers=2 for A100, can be tuned.
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"\n✅ DATA LOADING COMPLETE!")
    print(f"🎯 Mode: {data_mode}")
    print(f"📊 Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    print(f"📦 Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")
    print(f"🔤 Vocabulary: {tokenizer.vocab_size:,} tokens (using {tokenizer.name_or_path})")
    return train_loader, eval_loader, tokenizer, data_mode

# --- 6. Training Loop (Identical) ---
def prepare_batch(batch, device):
    input_ids = batch['input_ids'].to(device, non_blocking=True)
    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
    labels = input_ids.clone()
    labels[~attention_mask.bool()] = 0 
    return input_ids, attention_mask, labels

def evaluate_model(model, eval_loader, device, config: GNNMoEConfig, max_batches=20): # max_batches for eval speed
    model.eval()
    total_loss = 0; total_tokens = 0
    pbar_eval = tqdm(eval_loader, desc="Evaluating", leave=False, total=min(len(eval_loader), max_batches))
    with torch.no_grad():
        for i, batch in enumerate(pbar_eval):
            if i >= max_batches: break
            input_ids, attention_mask, labels = prepare_batch(batch, device)
            outputs = model(input_ids, attention_mask, labels=labels)
            mask = labels != 0
            if mask.sum() > 0:
                total_loss += outputs['loss'].item() * mask.sum().item()
                total_tokens += mask.sum().item()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(min(avg_loss, 20)) # Cap perplexity for stability if loss is too high
    return avg_loss, perplexity

def train_gnn_moe(model, train_loader, eval_loader, device, config: GNNMoEConfig,
                  checkpoint_dir="checkpoints", resume_from_epoch=0, resume_step=0, initial_best_loss=float('inf')):
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    actual_batches_per_epoch = min(len(train_loader), config.max_batches_per_epoch)
    total_steps = config.epochs * actual_batches_per_epoch
    
    if total_steps == 0:
        print("⚠️ Total training steps is 0. Skipping training.")
        return defaultdict(list), float('inf')
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0) # Ensure eta_min for full decay
    
    # If resuming, fast-forward scheduler and optimizer if they were loaded
    # Note: If optimizer/scheduler are loaded from checkpoint, they are already at correct state.
    # This manual step advance is more for if only model weights were loaded and we are "resuming" conceptually.
    # However, load_checkpoint handles optimizer/scheduler state if available.
    # For simplicity, we assume load_checkpoint sets them up if they were saved.
    # If resuming step > 0, scheduler might need to be advanced if not loaded from checkpoint.
    # For robust resumption, ensure optimizer and scheduler states are saved and loaded.

    stats = defaultdict(list)
    best_eval_loss = initial_best_loss
    step = resume_step 
    start_time = time.time()

    print(f"\n🚀 Starting GNN-MoE Training on {device} with WikiText-2 Data")
    if resume_from_epoch > 0 or resume_step > 0:
        print(f"🔄 Resuming from epoch {resume_from_epoch}, step {resume_step}. Initial best_eval_loss: {initial_best_loss:.4f}")
    print(f"📊 Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"🎯 Training: {config.epochs} epochs × {actual_batches_per_epoch} batches/epoch = {total_steps} steps")

    for epoch in range(resume_from_epoch, config.epochs):
        model.train(); epoch_loss = 0; epoch_steps = 0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", total=actual_batches_per_epoch)
        
        # If resuming, skip batches already processed in the current epoch
        start_batch_idx = step % actual_batches_per_epoch if epoch == resume_from_epoch and resume_step > 0 else 0
        if start_batch_idx > 0:
            print(f"Epoch {epoch+1}: Skipping to batch {start_batch_idx} due to resumption.")
        
        for batch_idx, batch in enumerate(pbar_train):
            if batch_idx < start_batch_idx:
                # Need to advance scheduler for skipped steps if not loaded from checkpoint
                # However, if scheduler state IS loaded, this is not needed.
                # Assuming scheduler is correctly at `step` if loaded.
                # If creating a new scheduler for a resumed run where only model weights were loaded,
                # then one might need: if step < resume_step: scheduler.step()
                continue 
            if batch_idx >= actual_batches_per_epoch: break

            input_ids, attention_mask, labels = prepare_batch(batch, device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs['loss']
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # Important: scheduler.step() should be called based on global step if not batch-wise
            # For CosineAnnealingLR, it's typically called after each optimizer.step()
            scheduler.step() # Call scheduler after optimizer step

            # Global step count is incremented *after* this batch is processed
            current_global_step = step + 1 

            epoch_loss += loss.item(); epoch_steps += 1
            stats['train_loss'].append(loss.item()); stats['grad_norm'].append(grad_norm.item()); stats['learning_rate'].append(scheduler.get_last_lr()[0])
            tokens_processed = current_global_step * config.batch_size * config.max_seq_length
            elapsed_train = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed_train if elapsed_train > 0 else 0
            pbar_train.set_postfix({'loss': f'{loss.item():.4f}', 'grad': f'{grad_norm.item():.2f}', 'tok/s': f'{tokens_per_sec:.0f}', 'lr': f"{scheduler.get_last_lr()[0]:.1e}"})

            if current_global_step % config.eval_every == 0 or current_global_step == total_steps:
                eval_loss, perplexity = evaluate_model(model, eval_loader, device, config)
                stats['eval_loss'].append(eval_loss); stats['eval_perplexity'].append(perplexity); stats['eval_step'].append(current_global_step)
                print(f"\n📈 Step {current_global_step}: Eval Loss: {eval_loss:.4f}, PPL: {perplexity:.2f}")
                
                is_best = eval_loss < best_eval_loss
                if is_best:
                    best_eval_loss = eval_loss
                    print(f"🎯 New best eval loss: {best_eval_loss:.4f}")
                
                save_checkpoint({
                    'epoch': epoch + 1, # Next epoch to start from if resumed
                    'step': current_global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_eval_loss': best_eval_loss,
                    'config': config
                }, is_best, checkpoint_dir=checkpoint_dir)
                model.train()
            
            step = current_global_step # Update global step count

        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        print(f"Epoch {epoch+1} Summary - Avg Train Loss: {avg_epoch_loss:.4f}, Time: {(time.time() - start_time)/60:.1f}m")
        # Save checkpoint at end of each epoch as well
        save_checkpoint({
            'epoch': epoch + 1, 
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_eval_loss': best_eval_loss,
            'config': config
        }, False, checkpoint_dir=checkpoint_dir, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")
    total_time_min = (time.time() - start_time) / 60
    print("\n✅ GNN-MoE Training Complete!"); print(f"🎯 Best eval loss: {best_eval_loss:.4f}"); print(f"⏱️ Total time: {total_time_min:.1f} minutes")
    return stats, best_eval_loss

# --- 7. Analysis and Visualization Utilities (Plot filenames updated) ---
def analyze_expert_communication(model, config: GNNMoEConfig, detailed=True): # Identical
    print("\n🧠 Expert Communication Analysis")
    comm_data = model.analyze_expert_communication()
    for layer_name, matrices in comm_data.items():
        print(f"\n{layer_name.upper()}:")
        for gnn_idx, matrix in enumerate(matrices):
            print(f"  GNN Layer {gnn_idx+1} - Expert connectivity (Adjacency Strength):")
            connectivity = matrix.cpu().numpy()
            if detailed:
                for i in range(connectivity.shape[0]):
                    connections = [f"E{j}:{connectivity[i,j]:.3f}" for j in range(connectivity.shape[1]) if i != j]
                    print(f"    Expert {i} → [{', '.join(connections)}]")
            else: print(f"    Avg connectivity: {connectivity.mean():.3f}, Max: {connectivity.max():.3f}")
    return comm_data

def plot_expert_connectivity(comm_data, config: GNNMoEConfig, save_path="plots/wikitext_expert_connectivity.png"): # Path updated
    num_model_layers = len(comm_data);
    if num_model_layers == 0: print("⚠️ No communication data to plot."); return
    total_gnn_matrices = num_model_layers * config.gnn_layers
    cols = min(config.gnn_layers, 3); rows = (total_gnn_matrices + cols -1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False); axes = axes.flatten(); plot_idx = 0
    for layer_name, matrices_in_layer in comm_data.items():
        for gnn_idx, matrix in enumerate(matrices_in_layer):
            if plot_idx >= len(axes): break
            ax = axes[plot_idx]; connectivity = matrix.cpu().numpy()
            im = ax.imshow(connectivity, cmap='Blues', vmin=0, vmax=1)
            ax.set_title(f'{layer_name} GNN-{gnn_idx+1}'); ax.set_xlabel('To Expert'); ax.set_ylabel('From Expert')
            ax.set_xticks(np.arange(config.num_experts)); ax.set_yticks(np.arange(config.num_experts))
            ax.set_xticklabels([f'E{i}' for i in range(config.num_experts)]); ax.set_yticklabels([f'E{i}' for i in range(config.num_experts)])
            for i in range(config.num_experts):
                for j in range(config.num_experts):
                    ax.text(j, i, f'{connectivity[i,j]:.2f}', ha='center', va='center', color='white' if connectivity[i,j] > 0.5 else 'black', fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8); plot_idx += 1
    for idx_unused in range(plot_idx, len(axes)): axes[idx_unused].axis('off')
    plt.tight_layout(); plt.savefig(save_path); print(f"🎨 Expert connectivity plot saved to {save_path}"); plt.close(fig)

def plot_training_results(stats, config: GNNMoEConfig, save_path="plots/wikitext_training_results.png"): # Path updated
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0,0].plot(stats['train_loss'], alpha=0.7, label="Train Loss")
    if len(stats['train_loss']) > 50:
        smoothed_loss = np.convolve(stats['train_loss'], np.ones(50)/50, mode='valid')
        axes[0,0].plot(np.arange(49, len(stats['train_loss'])), smoothed_loss, label="Smoothed Train Loss (50 steps)")
    axes[0,0].set_title('Training Loss'); axes[0,0].set_xlabel('Step'); axes[0,0].set_ylabel('Loss'); axes[0,0].grid(True, alpha=0.3); axes[0,0].legend()
    if stats['eval_step']:
        ax_eval_loss = axes[0,1]
        ax_eval_loss.plot(stats['eval_step'], stats['eval_loss'], 'o-', color='orange', label='Eval Loss')
        ax_eval_loss.set_title('Evaluation Metrics'); ax_eval_loss.set_xlabel('Step'); ax_eval_loss.set_ylabel('Loss', color='orange'); ax_eval_loss.tick_params(axis='y', labelcolor='orange'); ax_eval_loss.grid(True, alpha=0.3)
        ax_ppl = ax_eval_loss.twinx()
        ax_ppl.plot(stats['eval_step'], stats['eval_perplexity'], 's-', color='red', label='Eval Perplexity'); ax_ppl.set_ylabel('Perplexity', color='red'); ax_ppl.tick_params(axis='y', labelcolor='red')
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2)
    axes[1,0].plot(stats['learning_rate']); axes[1,0].set_title('Learning Rate Schedule'); axes[1,0].set_xlabel('Step'); axes[1,0].set_ylabel('LR'); axes[1,0].grid(True, alpha=0.3)
    axes[1,1].plot(stats['grad_norm'], alpha=0.8, label="Grad Norm"); axes[1,1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Clip Threshold (1.0)'); axes[1,1].set_title('Gradient Norms'); axes[1,1].set_xlabel('Step'); axes[1,1].set_ylabel('Norm'); axes[1,1].grid(True, alpha=0.3); axes[1,1].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96]); fig.suptitle(f"GNN-MoE Training Results (WikiText-2, {config.num_experts} Experts, {config.gnn_layers} GNN Layers)", fontsize=16); plt.savefig(save_path); print(f"📊 Training results plot saved to {save_path}"); plt.close(fig)

def analyze_model_efficiency(model, config: GNNMoEConfig): # Identical
    print("\n⚡ GNN-MoE Efficiency Analysis")
    total_params = sum(p.numel() for p in model.parameters()); expert_params = sum(p.numel() for layer in model.model_layers for expert in layer.experts for p in expert.parameters()); gnn_params = sum(p.numel() for layer in model.model_layers for p in layer.gnn_coupler.parameters()); other_params = total_params - expert_params - gnn_params
    print(f"  Total Parameters: {total_params:,}"); print(f"  Expert Parameters: {expert_params:,} ({expert_params/total_params*100:.1f}%)"); print(f"  GNN Coord Params: {gnn_params:,} ({gnn_params/total_params*100:.1f}%)"); print(f"  Other (Embeds, etc.): {other_params:,} ({other_params/total_params*100:.1f}%)")
    print(f"  Expert Utilization: ALL {config.num_experts} experts active per layer."); print(f"  GNN Coordination: {config.gnn_layers} GNN layers per expert group.")

# --- 8. Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN-MoE WikiText-2 Training Script")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument('--resume_checkpoint', type=str, default=None, help="Path to checkpoint to resume training from.")
    # Add other arguments for GNNMoEConfig parameters here if desired
    # e.g., parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    
    args = parser.parse_args()

    # Ensure the specified checkpoint directory exists
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        print(f"📁 Created checkpoint directory: {args.checkpoint_dir}")

    print("===== GNN-MoE WikiText-2 Script Execution Started =====")
    selected_device = setup_environment(seed=42) # also creates ./plots and default ./checkpoints if not exists
    cfg = GNNMoEConfig() # Base config
    
    # Example of overriding config with args if they were added to parser:
    # if args.batch_size: cfg.batch_size = args.batch_size 
    
    train_loader, eval_loader, tokenizer, data_mode = load_data(cfg, num_train_samples_target=18000, num_eval_samples_target=1500)
    
    print(f"\n🏗️ Creating GNN-MoE Model with vocab_size: {cfg.vocab_size}")
    model = GNNMoEModel(cfg).to(selected_device)
    
    # Prepare for potential resumption
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01) # Re-create optimizer for load_checkpoint
    actual_batches_per_epoch = min(len(train_loader), cfg.max_batches_per_epoch)
    total_steps = cfg.epochs * actual_batches_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0) if total_steps > 0 else None

    start_epoch = 0
    current_step = 0
    best_eval_loss_resumed = float('inf')

    if args.resume_checkpoint:
        if os.path.isfile(args.resume_checkpoint):
            resume_data = load_checkpoint(args.resume_checkpoint, model, optimizer, scheduler)
            if resume_data:
                start_epoch, current_step, best_eval_loss_resumed = resume_data
                # Ensure model is on the correct device after loading
                model.to(selected_device) 
        else:
            print(f"⚠️ Checkpoint file not found at '{args.resume_checkpoint}'. Starting from scratch.")

    training_stats, final_best_loss = train_gnn_moe(
        model, train_loader, eval_loader, selected_device, cfg,
        checkpoint_dir=args.checkpoint_dir,
        resume_from_epoch=start_epoch,
        resume_step=current_step,
        initial_best_loss=best_eval_loss_resumed
    )
    
    if training_stats: # Check if training_stats is not empty (e.g. if total_steps was 0)
        plot_training_results(training_stats, cfg) 
        # Load the best model for final analysis if needed
        best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth.tar")
        if os.path.exists(best_model_path):
            print(f"🔄 Loading best model from {best_model_path} for final analysis...")
            # Create a new model instance or ensure current model is clean before loading best weights
            # For simplicity, we'll re-use the current model instance.
            # If loading into a fresh model: best_model_instance = GNNMoEModel(cfg).to(selected_device)
            load_checkpoint(best_model_path, model) # Only loads model_state_dict
            model.to(selected_device) # Ensure it's on the right device
        
        communication_data = analyze_expert_communication(model, cfg, detailed=False)
        if communication_data: # Check if communication_data is not empty
             plot_expert_connectivity(communication_data, cfg)
    
    analyze_model_efficiency(model, cfg)
    
    print("\n🎉 GNN-MoE WikiText-2 Script Execution Finished Successfully!")
    print(f"   Data Mode: {data_mode}"); print(f"   Best Eval Loss from run: {final_best_loss:.4f}")
    # To print the absolute best perplexity, it should be saved/loaded with the checkpoint state
    # For now, we print perplexity if stats are available from the current run
    if training_stats and 'eval_perplexity' in training_stats and training_stats['eval_perplexity']: # Corrected: stats -> training_stats
        # Find perplexity corresponding to the best_eval_loss if possible
        # This is a bit simplified; proper way is to save PPL with checkpoint.
        try:
            best_loss_idx = training_stats['eval_loss'].index(final_best_loss) # Corrected: stats -> training_stats
            best_ppl_from_run = training_stats['eval_perplexity'][best_loss_idx] # Corrected: stats -> training_stats
            print(f"   Best Eval Perplexity from run: {best_ppl_from_run:.2f}")
        except (ValueError, IndexError):
             if training_stats['eval_perplexity']: # Fallback to min PPL if exact match not found; Corrected: stats -> training_stats
                 print(f"   Min Eval Perplexity from run (fallback): {min(training_stats['eval_perplexity']):.2f}") # Corrected: stats -> training_stats
    print("==============================================")
else:
    print("GNN-MoE WikiText-2 script imported as a module.")
