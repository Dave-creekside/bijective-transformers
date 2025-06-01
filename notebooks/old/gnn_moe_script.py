#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gnn_moe_script.py

Consolidated Python script for GNN-Coupled Mixture of Experts (MoE) research.
This script combines all components from the research notebook into a runnable
Python file, focusing on modularity, clear logging, and file-based outputs
for analysis (e.g., saving plots).
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
import os # For creating directories

# --- 1. Initial Setup ---
def setup_environment(seed=42):
    """Sets up the environment, device, and seeds."""
    plt.style.use('default')
    sns.set_palette("husl")

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Device: Apple MPS")
    else:
        device = torch.device("cpu")
        print("🚀 Device: CPU")

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create plots directory if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")
        print("📁 Created 'plots' directory for output visualizations.")

    print("✅ Environment ready for GNN-MoE research!")
    print("🧠 Innovation: Graph Neural Networks coordinate ALL experts")
    print("⚡ No sparse routing - learned graph communication patterns")
    return device

# --- 2. Configuration ---
@dataclass
class GNNMoEConfig:
    vocab_size: int = 50257        # Default to GPT-2 vocab size
    max_seq_length: int = 128
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    num_experts: int = 4
    gnn_layers: int = 2
    # Training specific
    batch_size: int = 16
    learning_rate: float = 5e-4
    epochs: int = 8
    max_batches_per_epoch: int = 50 # For faster training cycles
    eval_every: int = 25            # Evaluate every N steps

# --- 3. Core GNN Architecture ---
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
                    content_weight = torch.sigmoid(self.message_weight(concat_features).squeeze(-1)) # Ensure squeeze is correct
                    message_strength = adjacency[expert_idx, other_idx] * content_weight
                    weighted_message = other_expert * message_strength.unsqueeze(1)
                    messages.append(weighted_message)
            
            if messages:
                neighbor_msg = torch.stack(messages, dim=0).sum(dim=0) # sum over num_other_experts
                neighbor_out = self.neighbor_transform(neighbor_msg)
            else:
                neighbor_out = torch.zeros_like(self.neighbor_transform(current_expert)) # Ensure correct shape if no messages

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
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False # Typically not needed for forward pass unless analyzing attention
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class GNNExpertCoupler(nn.Module):
    def __init__(self, config: GNNMoEConfig):
        super().__init__()
        self.config = config
        self.gnn_layers_module = nn.ModuleList([ # Renamed to avoid conflict
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
        for gnn_layer_instance in self.gnn_layers_module: # Use renamed attribute
            new_features = gnn_layer_instance(expert_features)
            expert_features = new_features + expert_features # Residual connection for GNN layers
        
        coordinated_output = expert_features.mean(dim=2)
        output = self.combiner(coordinated_output)
        return output

    def get_expert_communication_matrices(self):
        matrices = []
        for gnn_layer_instance in self.gnn_layers_module: # Use renamed attribute
            matrices.append(gnn_layer_instance.get_adjacency_matrix())
        return matrices

# --- 4. Complete Model Architecture ---
class GNNMoELayer(nn.Module):
    def __init__(self, config: GNNMoEConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            ExpertBlock(config) for _ in range(config.num_experts)
        ])
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
        self.model_layers = nn.ModuleList([ # Renamed to avoid conflict
            GNNMoELayer(config) for _ in range(config.num_layers)
        ])
        self.output_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, input_ids, attention_mask=None, return_loss=True, labels=None):
        B, L = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        x = self.dropout(x)
        
        causal_mask = self.create_causal_mask(L, input_ids.device)
        key_padding_mask = (~attention_mask.bool()) if attention_mask is not None else None
        
        for layer_instance in self.model_layers: # Use renamed attribute
            x = layer_instance(x, causal_mask, key_padding_mask)
            
        x = self.output_norm(x)
        logits = self.lm_head(x)
        
        if return_loss and labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0 
            )
            return {'loss': loss, 'logits': logits}
        return {'logits': logits}

    def analyze_expert_communication(self):
        comm_data = {}
        for i, layer_instance in enumerate(self.model_layers): # Use renamed attribute
            if hasattr(layer_instance, 'gnn_coupler'): # Ensure it's a GNNMoELayer
                 matrices = layer_instance.gnn_coupler.get_expert_communication_matrices()
                 comm_data[f'layer_{i}'] = matrices
        return comm_data

# --- 5. Data Handling ---
class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0), # Squeeze batch dim
            'attention_mask': encoding['attention_mask'].squeeze(0) # Squeeze batch dim
        }

def load_data(config: GNNMoEConfig, num_train_samples=2000, num_eval_samples=500):
    print("🚀 Setting up data loading...")
    try:
        from transformers import AutoTokenizer
        import datasets as hf_datasets
        
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        # Update config vocab size if tokenizer changes it
        if config.vocab_size != tokenizer.vocab_size:
            print(f"⚠️ Updating config.vocab_size from {config.vocab_size} to {tokenizer.vocab_size} based on tokenizer.")
            config.vocab_size = tokenizer.vocab_size

        print("📦 Attempting SST-2 dataset loading...")
        # Try with trust_remote_code=True as it sometimes helps with new/community datasets
        dataset_sst2 = hf_datasets.load_dataset("sst2", split="train", trust_remote_code=True)
        
        texts_sst2 = [item['sentence'].strip() for item in dataset_sst2 if len(item['sentence'].strip()) > 10]
        
        # Ensure enough samples
        if len(texts_sst2) < (num_train_samples + num_eval_samples):
            print(f"⚠️ SST-2 has only {len(texts_sst2)} samples after filtering. Adjusting sample counts or falling back.")
            # Fallback to using all available if not enough, or could raise error
            if len(texts_sst2) < 150: # Arbitrary minimum
                 raise ValueError("SST-2 dataset too small.")
            actual_train_samples = int(len(texts_sst2) * 0.8)
            actual_eval_samples = len(texts_sst2) - actual_train_samples
        else:
            actual_train_samples = num_train_samples
            actual_eval_samples = num_eval_samples

        train_texts = texts_sst2[:actual_train_samples]
        eval_texts = texts_sst2[actual_train_samples : actual_train_samples + actual_eval_samples]

        train_dataset = SimpleTextDataset(train_texts, tokenizer, config.max_seq_length)
        eval_dataset = SimpleTextDataset(eval_texts, tokenizer, config.max_seq_length)
        
        print("✅ SUCCESS: Real SST-2 data loaded!")
        data_mode = "REAL_SST2"
        
    except Exception as e:
        print(f"⚠️ Real data (SST-2) loading failed: {e}")
        print("🔄 Using high-quality synthetic fallback...")
        
        from transformers import AutoTokenizer # Ensure tokenizer is available for fallback
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        if config.vocab_size != tokenizer.vocab_size: # Ensure config matches
            print(f"⚠️ Updating config.vocab_size from {config.vocab_size} to {tokenizer.vocab_size} for synthetic data.")
            config.vocab_size = tokenizer.vocab_size

        synthetic_texts = [
            "The transformer architecture revolutionized natural language processing.",
            "Machine learning models learn from data to make predictions.",
            "Neural networks consist of layers of interconnected nodes.",
            "Deep learning uses multiple layers to extract hierarchical features.",
            "Language models predict the next word in a sequence.",
        ] * ( (num_train_samples + num_eval_samples) // 5 + 1) # Ensure enough synthetic samples
        
        train_dataset = SimpleTextDataset(synthetic_texts[:num_train_samples], tokenizer, config.max_seq_length)
        eval_dataset = SimpleTextDataset(synthetic_texts[num_train_samples : num_train_samples + num_eval_samples], tokenizer, config.max_seq_length)
        
        print("✅ Synthetic data ready!")
        data_mode = "SYNTHETIC_REALISTIC"

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"\n✅ DATA LOADING COMPLETE!")
    print(f"🎯 Mode: {data_mode}")
    print(f"📊 Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    print(f"📦 Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")
    print(f"🔤 Vocabulary: {tokenizer.vocab_size:,} tokens (using {tokenizer.name_or_path})")
    return train_loader, eval_loader, tokenizer, data_mode


# --- 6. Training Loop ---
def prepare_batch(batch, device):
    input_ids = batch['input_ids'].to(device, non_blocking=True)
    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
    labels = input_ids.clone()
    labels[~attention_mask.bool()] = 0 
    return input_ids, attention_mask, labels

def evaluate_model(model, eval_loader, device, config: GNNMoEConfig, max_batches=20):
    model.eval()
    total_loss = 0
    total_tokens = 0
    pbar_eval = tqdm(eval_loader, desc="Evaluating", leave=False, total=min(len(eval_loader), max_batches))
    with torch.no_grad():
        for i, batch in enumerate(pbar_eval):
            if i >= max_batches:
                break
            input_ids, attention_mask, labels = prepare_batch(batch, device)
            outputs = model(input_ids, attention_mask, labels=labels)
            mask = labels != 0
            if mask.sum() > 0:
                total_loss += outputs['loss'].item() * mask.sum().item()
                total_tokens += mask.sum().item()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(min(avg_loss, 10)) 
    return avg_loss, perplexity

def train_gnn_moe(model, train_loader, eval_loader, device, config: GNNMoEConfig):
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    total_steps = config.epochs * min(len(train_loader), config.max_batches_per_epoch)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    
    stats = defaultdict(list)
    best_eval_loss = float('inf')
    step = 0
    start_time = time.time()

    print(f"\n🚀 Starting GNN-MoE Training on {device}")
    print(f"📊 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"⚡ Config: {config.num_experts} experts/layer, {config.gnn_layers} GNN layers for coordination")
    print(f"🎯 Training: {config.epochs} epochs × {min(len(train_loader), config.max_batches_per_epoch)} batches/epoch = {total_steps} steps")

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        epoch_steps = 0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", total=min(len(train_loader), config.max_batches_per_epoch))

        for batch_idx, batch in enumerate(pbar_train):
            if batch_idx >= config.max_batches_per_epoch:
                break
            
            input_ids, attention_mask, labels = prepare_batch(batch, device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs['loss']
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            epoch_loss += loss.item()
            epoch_steps += 1
            stats['train_loss'].append(loss.item())
            stats['grad_norm'].append(grad_norm.item())
            stats['learning_rate'].append(scheduler.get_last_lr()[0])

            tokens_processed = step * config.batch_size * config.max_seq_length
            elapsed_train = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed_train if elapsed_train > 0 else 0
            
            pbar_train.set_postfix({
                'loss': f'{loss.item():.4f}', 'grad': f'{grad_norm.item():.2f}',
                'tok/s': f'{tokens_per_sec:.0f}', 'lr': f"{scheduler.get_last_lr()[0]:.1e}"
            })

            if step % config.eval_every == 0 or step == total_steps:
                eval_loss, perplexity = evaluate_model(model, eval_loader, device, config)
                stats['eval_loss'].append(eval_loss)
                stats['eval_perplexity'].append(perplexity)
                stats['eval_step'].append(step)
                print(f"\n📈 Step {step}: Eval Loss: {eval_loss:.4f}, PPL: {perplexity:.2f}")
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    print(f"🎯 New best eval loss: {best_eval_loss:.4f}")
                    # Could save checkpoint here
                model.train()
        
        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        print(f"Epoch {epoch+1} Summary - Avg Train Loss: {avg_epoch_loss:.4f}, Time: {(time.time() - start_time)/60:.1f}m")

    total_time_min = (time.time() - start_time) / 60
    print("\n✅ GNN-MoE Training Complete!")
    print(f"🎯 Best eval loss: {best_eval_loss:.4f}")
    print(f"⏱️ Total time: {total_time_min:.1f} minutes")
    return stats, best_eval_loss

# --- 7. Analysis and Visualization Utilities ---
def analyze_expert_communication(model, config: GNNMoEConfig, detailed=True):
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
            else:
                print(f"    Avg connectivity: {connectivity.mean():.3f}, Max: {connectivity.max():.3f}")
    return comm_data

def plot_expert_connectivity(comm_data, config: GNNMoEConfig, save_path="plots/expert_connectivity.png"):
    num_model_layers = len(comm_data)
    if num_model_layers == 0:
        print("⚠️ No communication data to plot.")
        return
        
    # Determine grid size for subplots
    # Each GNNMoELayer has `config.gnn_layers` GNNs in its coupler
    total_gnn_matrices = num_model_layers * config.gnn_layers
    cols = min(config.gnn_layers, 3) # Max 3 GNN layers per row in plot
    rows = (total_gnn_matrices + cols -1) // cols # Calculate rows needed for all GNN matrices
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()
    plot_idx = 0

    for layer_name, matrices_in_layer in comm_data.items():
        for gnn_idx, matrix in enumerate(matrices_in_layer):
            if plot_idx >= len(axes): break
            ax = axes[plot_idx]
            connectivity = matrix.cpu().numpy()
            im = ax.imshow(connectivity, cmap='Blues', vmin=0, vmax=1)
            ax.set_title(f'{layer_name} GNN-{gnn_idx+1}')
            ax.set_xlabel('To Expert'); ax.set_ylabel('From Expert')
            ax.set_xticks(np.arange(config.num_experts))
            ax.set_yticks(np.arange(config.num_experts))
            ax.set_xticklabels([f'E{i}' for i in range(config.num_experts)])
            ax.set_yticklabels([f'E{i}' for i in range(config.num_experts)])
            for i in range(config.num_experts):
                for j in range(config.num_experts):
                    ax.text(j, i, f'{connectivity[i,j]:.2f}', ha='center', va='center', 
                            color='white' if connectivity[i,j] > 0.5 else 'black', fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8)
            plot_idx += 1
    
    for idx_unused in range(plot_idx, len(axes)):
        axes[idx_unused].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"🎨 Expert connectivity plot saved to {save_path}")
    plt.close(fig)


def plot_training_results(stats, config: GNNMoEConfig, save_path="plots/training_results.png"):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10)) # Adjusted to 2x2 for key plots
    
    # Training Loss
    axes[0,0].plot(stats['train_loss'], alpha=0.7, label="Train Loss")
    if len(stats['train_loss']) > 50: # Add smoothed loss if enough data
        smoothed_loss = np.convolve(stats['train_loss'], np.ones(50)/50, mode='valid')
        axes[0,0].plot(np.arange(49, len(stats['train_loss'])), smoothed_loss, label="Smoothed Train Loss (50 steps)")
    axes[0,0].set_title('Training Loss'); axes[0,0].set_xlabel('Step'); axes[0,0].set_ylabel('Loss')
    axes[0,0].grid(True, alpha=0.3); axes[0,0].legend()

    # Eval Loss and Perplexity
    if stats['eval_step']:
        ax_eval_loss = axes[0,1]
        ax_eval_loss.plot(stats['eval_step'], stats['eval_loss'], 'o-', color='orange', label='Eval Loss')
        ax_eval_loss.set_title('Evaluation Metrics'); ax_eval_loss.set_xlabel('Step'); ax_eval_loss.set_ylabel('Loss', color='orange')
        ax_eval_loss.tick_params(axis='y', labelcolor='orange')
        ax_eval_loss.grid(True, alpha=0.3)

        ax_ppl = ax_eval_loss.twinx()
        ax_ppl.plot(stats['eval_step'], stats['eval_perplexity'], 's-', color='red', label='Eval Perplexity')
        ax_ppl.set_ylabel('Perplexity', color='red')
        ax_ppl.tick_params(axis='y', labelcolor='red')
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2) # Combined legend for eval plot

    # Learning Rate
    axes[1,0].plot(stats['learning_rate'])
    axes[1,0].set_title('Learning Rate Schedule'); axes[1,0].set_xlabel('Step'); axes[1,0].set_ylabel('LR')
    axes[1,0].grid(True, alpha=0.3)

    # Gradient Norms
    axes[1,1].plot(stats['grad_norm'], alpha=0.8, label="Grad Norm")
    axes[1,1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Clip Threshold (1.0)')
    axes[1,1].set_title('Gradient Norms'); axes[1,1].set_xlabel('Step'); axes[1,1].set_ylabel('Norm')
    axes[1,1].grid(True, alpha=0.3); axes[1,1].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    fig.suptitle(f"GNN-MoE Training Results ({config.num_experts} Experts, {config.gnn_layers} GNN Layers)", fontsize=16)
    plt.savefig(save_path)
    print(f"📊 Training results plot saved to {save_path}")
    plt.close(fig)

def analyze_model_efficiency(model, config: GNNMoEConfig):
    print("\n⚡ GNN-MoE Efficiency Analysis")
    total_params = sum(p.numel() for p in model.parameters())
    expert_params = sum(p.numel() for layer in model.model_layers for expert in layer.experts for p in expert.parameters())
    gnn_params = sum(p.numel() for layer in model.model_layers for p in layer.gnn_coupler.parameters())
    other_params = total_params - expert_params - gnn_params
    
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Expert Parameters: {expert_params:,} ({expert_params/total_params*100:.1f}%)")
    print(f"  GNN Coord Params: {gnn_params:,} ({gnn_params/total_params*100:.1f}%)")
    print(f"  Other (Embeds, etc.): {other_params:,} ({other_params/total_params*100:.1f}%)")
    print(f"  Expert Utilization: ALL {config.num_experts} experts active per layer.")
    print(f"  GNN Coordination: {config.gnn_layers} GNN layers per expert group.")

# --- 8. Main Execution Block ---
if __name__ == "__main__":
    print("===== GNN-MoE Script Execution Started =====")
    
    # Setup
    selected_device = setup_environment(seed=42)
    cfg = GNNMoEConfig() # Use default config or modify here
    
    # Load Data
    # Note: data loading might try to download. Ensure internet access.
    # It will also update cfg.vocab_size if the tokenizer's vocab_size is different.
    train_loader, eval_loader, tokenizer, data_mode = load_data(cfg)
    
    # Create Model (vocab_size might have been updated by load_data)
    print(f"\n🏗️ Creating GNN-MoE Model with vocab_size: {cfg.vocab_size}")
    model = GNNMoEModel(cfg).to(selected_device)
    
    # Train Model
    training_stats, best_loss = train_gnn_moe(model, train_loader, eval_loader, selected_device, cfg)
    
    # Analyze and Visualize
    if training_stats: # Only plot if training happened and stats were collected
        plot_training_results(training_stats, cfg)
        communication_data = analyze_expert_communication(model, cfg, detailed=False) # Set detailed=True for more verbose output
        if communication_data:
             plot_expert_connectivity(communication_data, cfg)
    
    analyze_model_efficiency(model, cfg)
    
    print("\n🎉 GNN-MoE Script Execution Finished Successfully!")
    print(f"   Data Mode: {data_mode}")
    print(f"   Best Eval Loss: {best_loss:.4f}")
    if training_stats and 'eval_perplexity' in training_stats and training_stats['eval_perplexity']:
        print(f"   Best Eval Perplexity: {min(training_stats['eval_perplexity']):.2f}")
    print("==============================================")

else:
    # This part allows importing components if needed, without running main script
    print("GNN-MoE script imported as a module. Main execution block not run.")
