#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gnn_moe_hyperparam_script.py

Hyperparameter-configurable script for GNN-Coupled MoE model training.
Allows setting model architecture, training params, and dataset via CLI.
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
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from tqdm import tqdm
from collections import defaultdict
import time
import random
import os
import argparse
import shutil

# --- 1. Configuration Dataclass ---
# Values here are defaults if not overridden by CLI args
@dataclass
class GNNMoEConfig:
    # Model Architecture
    vocab_size: int = 50257
    max_seq_length: int = 128
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8 # Often embed_dim // 64
    dropout_rate: float = 0.1 # Renamed from 'dropout' to avoid conflict with nn.Dropout
    num_experts: int = 4
    gnn_layers: int = 2 # GNN layers in the coupler

    # Training Hyperparameters
    batch_size: int = 32 
    learning_rate: float = 5e-4
    epochs: int = 8 
    max_batches_per_epoch: int = -1 # -1 means full epoch
    eval_every: int = 200 # Steps
    
    # Dataset
    dataset_name: str = "wikitext"
    dataset_config_name: str = "wikitext-2-v1"
    num_train_samples: int = 18000 # -1 for all available
    num_eval_samples: int = 1500   # -1 for all available

    # Checkpointing & Output
    checkpoint_dir: str = "checkpoints"
    resume_checkpoint: Optional[str] = None
    run_name: Optional[str] = None # For naming output files

    # Technical
    seed: int = 42
    num_workers_dataloader: int = 2

    def __post_init__(self):
        # Auto-calculate num_heads if not explicitly set or if embed_dim changes
        # and num_heads is still at a default that might not make sense.
        # This is a simple heuristic; more complex logic might be needed.
        if self.embed_dim % 64 == 0: # A common case
             expected_heads = self.embed_dim // 64
             if self.num_heads != expected_heads and self.num_heads == GNNMoEConfig.__dataclass_fields__['num_heads'].default : # only if num_heads is default
                print(f"Adjusting num_heads from {self.num_heads} to {expected_heads} based on embed_dim {self.embed_dim}")
                self.num_heads = expected_heads
        if self.embed_dim % self.num_heads != 0:
            print(f"Warning: embed_dim ({self.embed_dim}) is not divisible by num_heads ({self.num_heads}). This can cause issues.")


# --- 2. Initial Setup ---
def setup_environment(config: GNNMoEConfig):
    plt.style.use('default')
    sns.set_palette("husl")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Device: CUDA (Available: {torch.cuda.device_count()})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Device: Apple MPS")
    else:
        device = torch.device("cpu")
        print("🚀 Device: CPU")
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Create base plots and checkpoint directories if they don't exist
    # Specific run directories will be handled by args.run_name
    if not os.path.exists("plots"): os.makedirs("plots")
    if not os.path.exists(config.checkpoint_dir): # Use config.checkpoint_dir as base
         os.makedirs(config.checkpoint_dir)
         print(f"📁 Created base checkpoint directory: {config.checkpoint_dir}")

    print(f"✅ Environment ready. Seed: {config.seed}, Device: {device}")
    return device

# --- 3. Checkpoint Helper Functions (Identical to previous script) ---
def save_checkpoint(state, is_best, checkpoint_dir="checkpoints", filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, "best_model.pth.tar"))
        print(f"✅ Saved new best model to {os.path.join(checkpoint_dir, 'best_model.pth.tar')}")
    # else: # Reduce verbosity for non-best checkpoints during long runs
        # print(f"✅ Saved checkpoint to {filepath}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint file not found at '{checkpoint_path}'. Starting from scratch.")
        return 0, 0, float('inf') # epoch, step, best_loss
    
    print(f"🔄 Loading checkpoint from '{checkpoint_path}'")
    try:
        # Try loading with weights_only=True first if config object is not essential for resumption path
        # However, since we save config, we need weights_only=False or safe_globals
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except RuntimeError as e: # Catch specific error if GNNMoEConfig is not found
        if "GLOBAL __main__.GNNMoEConfig" in str(e):
            print("Info: GNNMoEConfig class not found during torch.load. This is okay if only loading model weights.")
            # Attempt to load only model state dict if config causes issues and isn't strictly needed for this load path
            # This is a fallback, ideally config should be loadable or not saved if problematic.
            # For now, we stick to weights_only=False which should work if class def matches.
            raise e # Re-raise if not the specific error we expect or if weights_only=False still fails
        else:
            raise e


    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
    step = checkpoint.get('step', 0)
    # loaded_config = checkpoint.get('config') # Could compare if needed
    
    print(f"✅ Checkpoint loaded. Resuming from epoch {start_epoch}, step {step}, best_eval_loss {best_eval_loss:.4f}")
    return start_epoch, step, best_eval_loss


# --- 4. Core GNN Architecture (Identical to previous script) ---
# Includes: ExpertGraphConv, ExpertBlock, GNNExpertCoupler
class ExpertGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts):
        super().__init__()
        self.in_dim = in_dim; self.out_dim = out_dim; self.num_experts = num_experts
        self.neighbor_transform = nn.Linear(in_dim, out_dim)
        self.self_transform = nn.Linear(in_dim, out_dim)
        self.message_weight = nn.Linear(in_dim * 2, 1)
        self.adjacency_logits = nn.Parameter(torch.randn(num_experts, num_experts))
    def forward(self, expert_features):
        B, S, N, E = expert_features.shape; adj = torch.sigmoid(self.adjacency_logits)
        flat_feat = expert_features.view(-1, N, E); updated_list = []
        for i in range(N):
            curr_exp = flat_feat[:, i, :]; msgs = []
            for j in range(N):
                if i == j: continue
                other_exp = flat_feat[:, j, :]
                concat_feat = torch.cat([curr_exp, other_exp], dim=1)
                content_w = torch.sigmoid(self.message_weight(concat_feat).squeeze(-1))
                msg_str = adj[i, j] * content_w
                weighted_msg = other_exp * msg_str.unsqueeze(1)
                msgs.append(weighted_msg)
            neigh_out = self.neighbor_transform(torch.stack(msgs).sum(0)) if msgs else torch.zeros_like(self.neighbor_transform(curr_exp))
            self_out = self.self_transform(curr_exp)
            updated_list.append(F.gelu(neigh_out + self_out))
        return torch.stack(updated_list, dim=1).view(B, S, N, E)
    def get_adjacency_matrix(self): return torch.sigmoid(self.adjacency_logits).detach()

class ExpertBlock(nn.Module):
    def __init__(self, config: GNNMoEConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(config.embed_dim, config.num_heads, dropout=config.dropout_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(config.embed_dim); self.norm2 = nn.LayerNorm(config.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4), nn.GELU(), nn.Dropout(config.dropout_rate),
            nn.Linear(config.embed_dim * 4, config.embed_dim), nn.Dropout(config.dropout_rate))
    def forward(self, x, causal_mask=None, key_padding_mask=None):
        x_norm = self.norm1(x); attn_out, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=causal_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out; x = x + self.ffn(self.norm2(x)); return x

class GNNExpertCoupler(nn.Module):
    def __init__(self, config: GNNMoEConfig):
        super().__init__(); self.config = config
        self.gnn_layers_module = nn.ModuleList([ExpertGraphConv(config.embed_dim, config.embed_dim, config.num_experts) for _ in range(config.gnn_layers)])
        self.combiner = nn.Sequential(nn.Linear(config.embed_dim, config.embed_dim), nn.GELU(), nn.LayerNorm(config.embed_dim))
    def forward(self, expert_outputs):
        stacked = torch.stack(expert_outputs, dim=2); features = stacked
        for gnn_layer in self.gnn_layers_module:
            new_features = gnn_layer(features); features = new_features + features # Residual
        return self.combiner(features.mean(dim=2))
    def get_expert_communication_matrices(self): return [gl.get_adjacency_matrix() for gl in self.gnn_layers_module]

# --- 5. Complete Model Architecture (Identical) ---
class GNNMoELayer(nn.Module):
    def __init__(self, config: GNNMoEConfig):
        super().__init__(); self.config = config
        self.experts = nn.ModuleList([ExpertBlock(config) for _ in range(config.num_experts)])
        self.gnn_coupler = GNNExpertCoupler(config)
    def forward(self, x, causal_mask=None, key_padding_mask=None):
        expert_outs = [exp(x, causal_mask, key_padding_mask) for exp in self.experts]
        return x + self.gnn_coupler(expert_outs)

class GNNMoEModel(nn.Module):
    def __init__(self, config: GNNMoEConfig):
        super().__init__(); self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.max_seq_length, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout_rate) # Use config.dropout_rate
        self.model_layers = nn.ModuleList([GNNMoELayer(config) for _ in range(config.num_layers)])
        self.output_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)
        self._init_weights()
    def _init_weights(self): # Identical
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, mean=0.0, std=0.02); 
            if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def create_causal_mask(self, L, dev): return torch.triu(torch.ones(L,L,device=dev),diagonal=1).bool()
    def forward(self, input_ids, attention_mask=None, return_loss=True, labels=None): # Identical
        B, L = input_ids.shape; pos_ids = torch.arange(L,device=input_ids.device).unsqueeze(0).expand(B,-1)
        x = self.dropout(self.token_emb(input_ids) + self.pos_emb(pos_ids))
        causal_mask = self.create_causal_mask(L, input_ids.device)
        key_pad_mask = (~attention_mask.bool()) if attention_mask is not None else None
        for layer in self.model_layers: x = layer(x, causal_mask, key_pad_mask)
        logits = self.lm_head(self.output_norm(x))
        if return_loss and labels is not None:
            loss = F.cross_entropy(logits[...,:-1,:].contiguous().view(-1,logits.size(-1)), labels[...,1:].contiguous().view(-1), ignore_index=0)
            return {'loss':loss, 'logits':logits}
        return {'logits':logits}
    def analyze_expert_communication(self): # Identical
        return {f'layer_{i}': l.gnn_coupler.get_expert_communication_matrices() for i,l in enumerate(self.model_layers) if hasattr(l,'gnn_coupler')}

# --- 6. Data Handling (Modified for CLI args) ---
class SimpleTextDataset(Dataset): # Identical
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts=texts; self.tokenizer=tokenizer; self.max_length=max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids':enc['input_ids'].squeeze(0), 'attention_mask':enc['attention_mask'].squeeze(0)}

def load_data(config: GNNMoEConfig):
    print(f"🚀 Setting up data loading for {config.dataset_name} / {config.dataset_config_name}...")
    try:
        from transformers import AutoTokenizer
        import datasets as hf_datasets
        
        tokenizer = AutoTokenizer.from_pretrained('gpt2') # Keep gpt2 tokenizer for now
        tokenizer.pad_token = tokenizer.eos_token
        if config.vocab_size != tokenizer.vocab_size:
            print(f"⚠️ Config vocab_size {config.vocab_size} != tokenizer vocab_size {tokenizer.vocab_size}. Using tokenizer's.")
            config.vocab_size = tokenizer.vocab_size

        print(f"📦 Attempting {config.dataset_name} ({config.dataset_config_name}) dataset loading...")
        train_raw = hf_datasets.load_dataset(config.dataset_name, config.dataset_config_name, split="train")
        eval_raw = hf_datasets.load_dataset(config.dataset_name, config.dataset_config_name, split="validation")
        
        train_texts_all = [ln.strip() for item in train_raw for ln in item['text'].splitlines() if len(ln.strip()) > 30]
        eval_texts_all = [ln.strip() for item in eval_raw for ln in item['text'].splitlines() if len(ln.strip()) > 30]
        
        print(f"Raw lines >30 chars: Train {len(train_texts_all)}, Eval {len(eval_texts_all)}")
        random.shuffle(train_texts_all); random.shuffle(eval_texts_all)

        train_s = len(train_texts_all) if config.num_train_samples == -1 else min(len(train_texts_all), config.num_train_samples)
        eval_s = len(eval_texts_all) if config.num_eval_samples == -1 else min(len(eval_texts_all), config.num_eval_samples)

        if train_s < 100 or eval_s < 50: raise ValueError(f"Dataset too small. Train: {train_s}, Eval: {eval_s}")

        train_dataset = SimpleTextDataset(train_texts_all[:train_s], tokenizer, config.max_seq_length)
        eval_dataset = SimpleTextDataset(eval_texts_all[:eval_s], tokenizer, config.max_seq_length)
        
        print(f"✅ SUCCESS: Real {config.dataset_config_name} data loaded!")
        data_mode = f"REAL_{config.dataset_config_name.upper()}"
        
    except Exception as e:
        print(f"⚠️ Real data ({config.dataset_config_name}) loading failed: {e}")
        print("🔄 Using synthetic fallback...")
        # Fallback logic remains similar, uses config.num_train_samples etc.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2'); tokenizer.pad_token = tokenizer.eos_token
        if config.vocab_size != tokenizer.vocab_size: config.vocab_size = tokenizer.vocab_size
        num_total_synth = (config.num_train_samples if config.num_train_samples != -1 else 2000) + \
                          (config.num_eval_samples if config.num_eval_samples != -1 else 500)
        synthetic_texts = ["The transformer architecture revolutionized NLP."] * num_total_synth
        train_s_synth = config.num_train_samples if config.num_train_samples != -1 else 2000
        
        train_dataset = SimpleTextDataset(synthetic_texts[:train_s_synth], tokenizer, config.max_seq_length)
        eval_dataset = SimpleTextDataset(synthetic_texts[train_s_synth:], tokenizer, config.max_seq_length)
        print("✅ Synthetic data ready!"); data_mode = "SYNTHETIC"

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers_dataloader, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers_dataloader, pin_memory=True)
    
    print(f"\n✅ DATA LOADING COMPLETE! Mode: {data_mode}"); print(f"📊 Samples: Train {len(train_dataset)}, Eval {len(eval_dataset)}")
    print(f"📦 Batches: Train {len(train_loader)}, Eval {len(eval_loader)}"); print(f"🔤 Vocab: {tokenizer.vocab_size:,} ({tokenizer.name_or_path})")
    return train_loader, eval_loader, tokenizer, data_mode


# --- 7. Training Loop (Checkpointing integrated) ---
def prepare_batch(batch, device): # Identical
    ids=batch['input_ids'].to(device,non_blocking=True); mask=batch['attention_mask'].to(device,non_blocking=True)
    labels=ids.clone(); labels[~mask.bool()]=0; return ids,mask,labels

def evaluate_model(model, eval_loader, device, config: GNNMoEConfig, max_batches=20): # Identical
    model.eval(); total_loss=0; total_tokens=0
    eff_max_batches = min(len(eval_loader), max_batches) if max_batches > 0 else len(eval_loader)
    pbar = tqdm(eval_loader, desc="Evaluating", leave=False, total=eff_max_batches)
    with torch.no_grad():
        for i,batch in enumerate(pbar):
            if max_batches > 0 and i >= max_batches: break
            ids,mask,labels = prepare_batch(batch,device); outputs=model(ids,mask,labels=labels)
            val_mask = labels!=0; if val_mask.sum()>0: total_loss+=outputs['loss'].item()*val_mask.sum().item(); total_tokens+=val_mask.sum().item()
    avg_loss = total_loss/total_tokens if total_tokens>0 else float('inf'); ppl = math.exp(min(avg_loss,20))
    return avg_loss, ppl

def train_gnn_moe(model, train_loader, eval_loader, device, config: GNNMoEConfig,
                  resume_from_epoch=0, resume_step=0, initial_best_loss=float('inf')): # Uses config.checkpoint_dir
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    actual_batches_per_epoch = len(train_loader) if config.max_batches_per_epoch == -1 else min(len(train_loader), config.max_batches_per_epoch)
    total_steps = config.epochs * actual_batches_per_epoch
    if total_steps == 0: print("⚠️ Total steps is 0."); return defaultdict(list), float('inf')
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6) # Added eta_min
    
    stats=defaultdict(list); best_eval_loss=initial_best_loss; step=resume_step; start_time=time.time()
    print(f"\n🚀 Training on {device}. Model: {sum(p.numel() for p in model.parameters()):,} params.")
    if resume_from_epoch > 0 or resume_step > 0: print(f"🔄 Resuming: epoch {resume_from_epoch}, step {resume_step}, best_loss {initial_best_loss:.4f}")
    print(f"🎯 Target: {config.epochs} epochs × {actual_batches_per_epoch} batches/epoch = {total_steps} steps")

    for epoch in range(resume_from_epoch, config.epochs):
        model.train(); epoch_loss=0; epoch_steps=0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", total=actual_batches_per_epoch)
        start_batch_idx = step % actual_batches_per_epoch if epoch == resume_from_epoch and resume_step > 0 else 0
        if start_batch_idx > 0: print(f"Epoch {epoch+1}: Skipping to batch {start_batch_idx}.")
        
        for batch_idx, batch_data in enumerate(pbar):
            if batch_idx < start_batch_idx: continue
            if batch_idx >= actual_batches_per_epoch: break
            ids,mask,labels = prepare_batch(batch_data,device); optimizer.zero_grad()
            outputs = model(ids,mask,labels=labels); loss=outputs['loss']; loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); scheduler.step()
            
            current_global_step = step + 1; epoch_loss+=loss.item(); epoch_steps+=1
            stats['train_loss'].append(loss.item()); stats['grad_norm'].append(grad_norm.item()); stats['learning_rate'].append(scheduler.get_last_lr()[0])
            tok_proc = current_global_step*config.batch_size*config.max_seq_length; elap_train=time.time()-start_time
            pbar.set_postfix({'loss':f'{loss.item():.4f}','grad':f'{grad_norm.item():.2f}','tok/s':f'{tok_proc/elap_train if elap_train>0 else 0:.0f}','lr':f"{scheduler.get_last_lr()[0]:.1e}"})

            if current_global_step % config.eval_every == 0 or current_global_step == total_steps:
                eval_loss,ppl = evaluate_model(model,eval_loader,device,config,max_batches=-1) # Eval on full set
                stats['eval_loss'].append(eval_loss);stats['eval_perplexity'].append(ppl);stats['eval_step'].append(current_global_step)
                print(f"\n📈 Step {current_global_step}: Eval Loss {eval_loss:.4f}, PPL {ppl:.2f}")
                is_best = eval_loss < best_eval_loss
                if is_best: best_eval_loss=eval_loss; print(f"🎯 New best: {best_eval_loss:.4f}")
                save_checkpoint({'epoch':epoch+1,'step':current_global_step,'model_state_dict':model.state_dict(),
                                 'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),
                                 'best_eval_loss':best_eval_loss,'config':vars(config)}, # Save config as dict
                                is_best, checkpoint_dir=config.checkpoint_dir)
                model.train()
            step = current_global_step
        
        print(f"Epoch {epoch+1} Summary: Avg Train Loss {epoch_loss/epoch_steps if epoch_steps>0 else 0:.4f}, Time {(time.time()-start_time)/60:.1f}m")
        save_checkpoint({'epoch':epoch+1,'step':step,'model_state_dict':model.state_dict(),
                         'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),
                         'best_eval_loss':best_eval_loss,'config':vars(config)}, 
                        False, checkpoint_dir=config.checkpoint_dir, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")
    print(f"\n✅ Training Complete! Best Eval Loss: {best_eval_loss:.4f}, Total Time: {(time.time()-start_time)/60:.1f}m")
    return stats, best_eval_loss

# --- 8. Analysis & Visualization (Plot filenames use run_name) ---
# analyze_expert_communication, plot_expert_connectivity, plot_training_results, analyze_model_efficiency
# are largely identical but save_path for plots will use config.run_name if provided.
def get_save_path(config: GNNMoEConfig, base_filename: str):
    prefix = f"{config.run_name}_" if config.run_name else ""
    return os.path.join("plots", f"{prefix}{base_filename}")

def analyze_expert_communication(model, config: GNNMoEConfig, detailed=True): # Identical
    # ... (content from previous script, no changes needed here for functionality) ...
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


def plot_expert_connectivity(comm_data, config: GNNMoEConfig):
    save_path = get_save_path(config, "expert_connectivity.png")
    # ... (plotting logic from previous script, ensure save_path is used) ...
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


def plot_training_results(stats, config: GNNMoEConfig):
    save_path = get_save_path(config, "training_results.png")
    # ... (plotting logic from previous script, ensure save_path is used) ...
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
    plt.tight_layout(rect=[0, 0, 1, 0.96]); fig.suptitle(f"GNN-MoE Results ({config.run_name if config.run_name else 'DefaultRun'}, {config.num_experts}E, {config.gnn_layers}GNN)", fontsize=16); plt.savefig(save_path); print(f"📊 Training results plot saved to {save_path}"); plt.close(fig)

def analyze_model_efficiency(model, config: GNNMoEConfig): # Identical
    # ... (content from previous script, no changes needed here for functionality) ...
    print("\n⚡ GNN-MoE Efficiency Analysis")
    total_params = sum(p.numel() for p in model.parameters()); expert_params = sum(p.numel() for layer in model.model_layers for expert in layer.experts for p in expert.parameters()); gnn_params = sum(p.numel() for layer in model.model_layers for p in layer.gnn_coupler.parameters()); other_params = total_params - expert_params - gnn_params
    print(f"  Total Parameters: {total_params:,}"); print(f"  Expert Parameters: {expert_params:,} ({expert_params/total_params*100:.1f}%)"); print(f"  GNN Coord Params: {gnn_params:,} ({gnn_params/total_params*100:.1f}%)"); print(f"  Other (Embeds, etc.): {other_params:,} ({other_params/total_params*100:.1f}%)")
    print(f"  Expert Utilization: ALL {config.num_experts} experts active per layer."); print(f"  GNN Coordination: {config.gnn_layers} GNN layers per expert group.")


# --- 9. Main Execution Block (Modified for argparse) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN-MoE Hyperparameter Training Script")
    
    # Model Architecture Args
    parser.add_argument('--embed_dim', type=int, default=GNNMoEConfig.embed_dim, help="Embedding dimension")
    parser.add_argument('--num_layers', type=int, default=GNNMoEConfig.num_layers, help="Number of GNNMoELayers")
    parser.add_argument('--num_heads', type=int, default=GNNMoEConfig.num_heads, help="Number of attention heads")
    parser.add_argument('--dropout_rate', type=float, default=GNNMoEConfig.dropout_rate, help="Dropout rate")
    parser.add_argument('--num_experts', type=int, default=GNNMoEConfig.num_experts, help="Number of experts per layer")
    parser.add_argument('--gnn_layers', type=int, default=GNNMoEConfig.gnn_layers, help="Number of GNN layers in coupler")

    # Training Hyperparameters Args
    parser.add_argument('--batch_size', type=int, default=GNNMoEConfig.batch_size, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=GNNMoEConfig.learning_rate, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=GNNMoEConfig.epochs, help="Number of epochs")
    parser.add_argument('--max_batches_per_epoch', type=int, default=GNNMoEConfig.max_batches_per_epoch, help="Max batches per epoch (-1 for full epoch)")
    parser.add_argument('--eval_every', type=int, default=GNNMoEConfig.eval_every, help="Evaluate every N steps")

    # Dataset Args
    parser.add_argument('--dataset_name', type=str, default=GNNMoEConfig.dataset_name, help="Hugging Face dataset name")
    parser.add_argument('--dataset_config_name', type=str, default=GNNMoEConfig.dataset_config_name, help="Hugging Face dataset config name")
    parser.add_argument('--num_train_samples', type=int, default=GNNMoEConfig.num_train_samples, help="Number of training samples (-1 for all)")
    parser.add_argument('--num_eval_samples', type=int, default=GNNMoEConfig.num_eval_samples, help="Number of evaluation samples (-1 for all)")

    # Checkpointing & Output Args
    parser.add_argument('--checkpoint_dir', type=str, default=GNNMoEConfig.checkpoint_dir, help="Base directory for checkpoints.") # This will be the PARENT of the run-specific dir
    parser.add_argument('--resume_checkpoint', type=str, default=GNNMoEConfig.resume_checkpoint, help="Path to checkpoint to resume training from.")
    parser.add_argument('--run_name', type=str, default=None, help="Optional run name for outputs subdir (e.g., plots/run_name_plot.png, checkpoints/run_name/ckpt.pth)")
    
    # Technical Args
    parser.add_argument('--seed', type=int, default=GNNMoEConfig.seed, help="Random seed")
    parser.add_argument('--num_workers_dataloader', type=int, default=GNNMoEConfig.num_workers_dataloader, help="Num workers for DataLoader")

    args = parser.parse_args()

    # Create a config instance using CLI args or defaults from GNNMoEConfig
    cfg = GNNMoEConfig() # Start with all defaults
    cli_args_dict = vars(args)

    for attr_name in GNNMoEConfig.__dataclass_fields__:
        if cli_args_dict.get(attr_name) is not None: # Check if CLI arg was provided (it will override default from parser)
            setattr(cfg, attr_name, cli_args_dict[attr_name])
            print(f"Overriding config.{attr_name} with CLI arg: {cli_args_dict[attr_name]}")
    
    # Specific handling for run_name to create a subdirectory for this run's checkpoints
    # The cfg.checkpoint_dir will now point to the specific run's directory.
    # The base_checkpoint_dir is what was passed to --checkpoint_dir or its default.
    base_checkpoint_dir = args.checkpoint_dir # Use the one from args (which has a default)
    if cfg.run_name:
        cfg.checkpoint_dir = os.path.join(base_checkpoint_dir, cfg.run_name)
    else:
        cfg.checkpoint_dir = base_checkpoint_dir # Use base if no run_name
    
    if not os.path.exists(cfg.checkpoint_dir): # This is now the run-specific or base dir
        os.makedirs(cfg.checkpoint_dir)
        print(f"📁 Ensured checkpoint directory exists: {cfg.checkpoint_dir}")
    
    # The setup_environment function creates the *default* "checkpoints" and "plots" if they don't exist.
    # We've already handled the specific run's checkpoint directory above.
    # Plots will be saved into "plots/" possibly prefixed by run_name.
    selected_device = setup_environment(cfg) # Pass full config for seed

    print("===== GNN-MoE Hyperparameter Script Execution Started =====")
    print(f"Run Name: {cfg.run_name if cfg.run_name else 'default_run'}")
    print(f"Effective Config: {cfg}")
    
    train_loader, eval_loader, tokenizer, data_mode = load_data(cfg)
    
    print(f"\n🏗️ Creating GNN-MoE Model with effective vocab_size: {cfg.vocab_size}")
    model = GNNMoEModel(cfg).to(selected_device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    actual_batches_per_epoch_main = len(train_loader) if cfg.max_batches_per_epoch == -1 else min(len(train_loader), cfg.max_batches_per_epoch)
    total_steps_main = cfg.epochs * actual_batches_per_epoch_main
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps_main, eta_min=1e-6) if total_steps_main > 0 else None

    start_epoch = 0
    current_step = 0
    best_eval_loss_resumed = float('inf')

    if cfg.resume_checkpoint:
        if os.path.isfile(cfg.resume_checkpoint):
            print(f"Attempting to resume from: {cfg.resume_checkpoint}")
            # Pass the cfg.checkpoint_dir (which is run-specific) to load_checkpoint
            # if checkpoints are expected to be within their run-specific dirs.
            # However, resume_checkpoint is a full path, so it's fine.
            resume_data = load_checkpoint(cfg.resume_checkpoint, model, optimizer, scheduler)
            if resume_data:
                start_epoch, current_step, best_eval_loss_resumed = resume_data
            model.to(selected_device) 
        else:
            print(f"⚠️ Resume checkpoint not found: '{cfg.resume_checkpoint}'. Starting fresh.")

    training_stats, final_best_loss = train_gnn_moe(
        model, train_loader, eval_loader, selected_device, cfg, # cfg.checkpoint_dir is now run-specific
        resume_from_epoch=start_epoch,
        resume_step=current_step,
        initial_best_loss=best_eval_loss_resumed
    )
    
    if training_stats: 
        plot_training_results(training_stats, cfg) # Uses cfg.run_name for plot filename
        
        best_model_path = os.path.join(cfg.checkpoint_dir, "best_model.pth.tar")
        if os.path.exists(best_model_path):
            print(f"🔄 Loading best model from {best_model_path} for final analysis...")
            final_analysis_model = GNNMoEModel(cfg).to(selected_device)
            load_checkpoint(best_model_path, final_analysis_model) 
            
            communication_data = analyze_expert_communication(final_analysis_model, cfg, detailed=False)
            if communication_data:
                plot_expert_connectivity(communication_data, cfg) # Uses cfg.run_name for plot filename
            analyze_model_efficiency(final_analysis_model, cfg)
        else: 
            print(f"⚠️ best_model.pth.tar not found in {cfg.checkpoint_dir}, analyzing current model state.")
            communication_data = analyze_expert_communication(model, cfg, detailed=False)
            if communication_data:
                plot_expert_connectivity(communication_data, cfg)
            analyze_model_efficiency(model, cfg)

    print("\n🎉 GNN-MoE Hyperparameter Script Execution Finished Successfully!")
    print(f"   Run Name: {cfg.run_name if cfg.run_name else 'default_run'}")
    print(f"   Data Mode: {data_mode}")
    print(f"   Best Eval Loss from run: {final_best_loss:.4f}")
    
    if training_stats and 'eval_perplexity' in training_stats and stats['eval_perplexity']:
        try:
            if final_best_loss in stats['eval_loss']:
                 best_loss_idx = stats['eval_loss'].index(final_best_loss)
                 best_ppl_from_run = stats['eval_perplexity'][best_loss_idx]
                 print(f"   Best Eval Perplexity from run: {best_ppl_from_run:.2f}")
            elif stats['eval_perplexity']: 
                 print(f"   Min Eval Perplexity from run (fallback): {min(stats['eval_perplexity']):.2f}")
        except (ValueError, IndexError):
             if stats['eval_perplexity']: 
                 print(f"   Min Eval Perplexity from run (exception fallback): {min(stats['eval_perplexity']):.2f}")
    print("==============================================")
else:
    print("GNN-MoE Hyperparameter script imported as a module.")

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.

<environment_details>
# VSCode Visible Files
gnn_moe_hyperparam_script.py

# VSCode Open Tabs
gnn_moe_script.py
colab_gnn_debug_wikitext.py
gnn_moe_wikitext_script.py
gnn_moe_hyperparam_script.py

# Current Time
5/31/2025, 11:23:03 PM (America/Detroit, UTC-4:00)

# Context Window Usage
564,157 / 1,048.576K tokens used (54%)

# Current Mode
ACT MODE
</environment_details>
