#!/usr/bin/env python3
"""
🧠 GNN-Coupled MoE Test Script
Revolutionary expert coordination using Graph Neural Networks
Optimized for Apple M3 Pro with MPS acceleration

Key Innovation: Using GNN to coordinate ALL experts instead of sparse routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List, Tuple
from tqdm import tqdm
from collections import defaultdict
import time
import random

# Device setup with MPS support for Apple Silicon
def setup_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Will be set in main()
device = None

# ============================================================================
# SYNTHETIC DATA FOR FAST TESTING
# ============================================================================

class UltraSimpleDataset(Dataset):
    """Ultra-simple synthetic dataset for testing GNN-MoE"""
    
    def __init__(self, vocab_size=5000, max_length=64, num_samples=2000):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        print(f"Generating {num_samples} synthetic samples...")
        
        # Create simple structured sequences
        self.sequences = []
        for _ in range(num_samples):
            # Pattern: [START] + random tokens + [END] + padding
            seq_len = random.randint(10, max_length - 2)
            sequence = [1]  # START token
            sequence.extend([random.randint(2, vocab_size-3) for _ in range(seq_len-2)])
            sequence.append(2)  # END token
            
            # Pad to max_length
            while len(sequence) < max_length:
                sequence.append(0)  # PAD token
                
            self.sequences.append(sequence)
        
        print(f"✅ Generated {len(self.sequences)} synthetic sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_ids = torch.tensor(sequence, dtype=torch.long)
        attention_mask = (input_ids != 0).long()  # 1 for non-pad, 0 for pad
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

# ============================================================================
# CUSTOM GNN FOR EXPERT COORDINATION
# ============================================================================

class ExpertGraphConv(nn.Module):
    """
    Custom Graph Neural Network layer for expert coordination
    The key innovation: experts communicate through learnable graph structure
    """
    
    def __init__(self, in_dim, out_dim, num_experts):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        
        # Message passing layers
        self.neighbor_transform = nn.Linear(in_dim, out_dim)
        self.self_transform = nn.Linear(in_dim, out_dim)
        self.message_weight = nn.Linear(in_dim * 2, 1)
        
        # Learnable adjacency (which experts should communicate)
        self.adjacency_logits = nn.Parameter(torch.randn(num_experts, num_experts))
        
    def forward(self, expert_features):
        """
        expert_features: [batch_size, seq_len, num_experts, embed_dim]
        Returns: [batch_size, seq_len, num_experts, embed_dim] 
        """
        batch_size, seq_len, num_experts, embed_dim = expert_features.shape
        
        # Learnable adjacency matrix
        adjacency = torch.sigmoid(self.adjacency_logits)
        
        # Reshape for efficient processing
        # [batch_size * seq_len, num_experts, embed_dim]
        flat_features = expert_features.view(-1, num_experts, embed_dim)
        
        # Message passing
        updated_features = []
        for expert_idx in range(num_experts):
            # Current expert features across all batch*seq positions
            current_expert = flat_features[:, expert_idx, :]  # [batch*seq, embed_dim]
            
            # Aggregate messages from all other experts
            messages = []
            for other_idx in range(num_experts):
                if other_idx != expert_idx:
                    other_expert = flat_features[:, other_idx, :]
                    
                    # Message weight based on adjacency and content similarity
                    concat_features = torch.cat([current_expert, other_expert], dim=1)
                    weight = adjacency[expert_idx, other_idx] * torch.sigmoid(
                        self.message_weight(concat_features).squeeze()
                    )
                    
                    weighted_message = other_expert * weight.unsqueeze(1)
                    messages.append(weighted_message)
            
            # Aggregate neighbor messages
            if messages:
                neighbor_message = torch.stack(messages, dim=0).sum(dim=0)
                neighbor_out = self.neighbor_transform(neighbor_message)
            else:
                neighbor_out = torch.zeros_like(current_expert)
            
            # Self transformation
            self_out = self.self_transform(current_expert)
            
            # Combine with activation
            updated_feature = F.gelu(neighbor_out + self_out)
            updated_features.append(updated_feature)
        
        # Stack back to original shape
        updated_stack = torch.stack(updated_features, dim=1)
        return updated_stack.view(batch_size, seq_len, num_experts, embed_dim)
    
    def get_adjacency_matrix(self):
        """Return current learned adjacency matrix"""
        return torch.sigmoid(self.adjacency_logits).detach()

# ============================================================================
# GNN-COUPLED MOE ARCHITECTURE
# ============================================================================

@dataclass
class GNNMoEConfig:
    vocab_size: int = 5000        # Smaller vocab for fast testing
    max_seq_length: int = 64      # Shorter sequences
    embed_dim: int = 256          # Reasonable size
    num_layers: int = 4           # Moderate depth
    num_heads: int = 8
    dropout: float = 0.1
    
    # GNN-MoE specific
    num_experts: int = 4          # 4 experts per layer
    gnn_layers: int = 2           # Depth of expert communication

class ExpertBlock(nn.Module):
    """Individual expert - standard transformer block"""
    
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
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(
            x_norm, x_norm, x_norm, 
            attn_mask=causal_mask, 
            key_padding_mask=key_padding_mask
        )
        x = x + attn_out
        
        # Feed-forward
        x = x + self.ffn(self.norm2(x))
        return x

class GNNExpertCoupler(nn.Module):
    """
    🚀 THE KEY INNOVATION: GNN-based expert coordination!
    Uses graph neural networks to coordinate experts instead of sparse routing
    """
    
    def __init__(self, config: GNNMoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.embed_dim = config.embed_dim
        
        # Stack of GNN layers for multi-hop expert communication
        self.gnn_layers = nn.ModuleList([
            ExpertGraphConv(config.embed_dim, config.embed_dim, config.num_experts)
            for _ in range(config.gnn_layers)
        ])
        
        # Final combination
        self.combiner = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim)
        )
        
    def forward(self, expert_outputs):
        """
        🧠 GNN-based expert coordination
        expert_outputs: List of [batch_size, seq_len, embed_dim] tensors
        """
        # Stack expert outputs: [batch_size, seq_len, num_experts, embed_dim]
        stacked_experts = torch.stack(expert_outputs, dim=2)
        
        # Apply GNN layers for multi-hop expert communication
        expert_features = stacked_experts
        for gnn_layer in self.gnn_layers:
            new_features = gnn_layer(expert_features)
            # Residual connection
            expert_features = new_features + stacked_experts
        
        # Aggregate experts (could use attention, but mean is simple)
        coordinated_output = expert_features.mean(dim=2)
        
        # Final transformation
        output = self.combiner(coordinated_output)
        return output
    
    def get_expert_communication_matrices(self):
        """Analyze how experts communicate"""
        matrices = []
        for gnn_layer in self.gnn_layers:
            matrices.append(gnn_layer.get_adjacency_matrix())
        return matrices

class GNNMoELayer(nn.Module):
    """Complete GNN-MoE layer: Experts + GNN coordination"""
    
    def __init__(self, config: GNNMoEConfig):
        super().__init__()
        self.config = config
        
        # Create expert blocks
        self.experts = nn.ModuleList([
            ExpertBlock(config) for _ in range(config.num_experts)
        ])
        
        # GNN-based coordination
        self.gnn_coupler = GNNExpertCoupler(config)
        
    def forward(self, x, causal_mask=None, key_padding_mask=None):
        # ALL experts process input (no sparse routing!)
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x, causal_mask, key_padding_mask)
            expert_outputs.append(expert_out)
        
        # GNN coordinates expert outputs
        coordinated = self.gnn_coupler(expert_outputs)
        
        # Residual connection
        return x + coordinated

class GNNMoEModel(nn.Module):
    """Complete GNN-Coupled MoE Language Model"""
    
    def __init__(self, config: GNNMoEConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.max_seq_length, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # GNN-MoE layers
        self.layers = nn.ModuleList([
            GNNMoELayer(config) for _ in range(config.num_layers)
        ])
        
        # Output
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
        
        # Embeddings
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        x = self.dropout(x)
        
        # Causal mask
        causal_mask = self.create_causal_mask(L, input_ids.device)
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        
        # Apply GNN-MoE layers
        for layer in self.layers:
            x = layer(x, causal_mask, key_padding_mask)
        
        # Output
        x = self.output_norm(x)
        logits = self.lm_head(x)
        
        if return_loss and labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0  # Ignore pad tokens
            )
            return {'loss': loss, 'logits': logits}
        
        return {'logits': logits}
    
    def analyze_expert_communication(self):
        """Analyze how experts communicate via GNN"""
        comm_data = {}
        for i, layer in enumerate(self.layers):
            matrices = layer.gnn_coupler.get_expert_communication_matrices()
            comm_data[f'layer_{i}'] = matrices
        return comm_data

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def prepare_batch(batch, device):
    """Prepare autoregressive training batch"""
    input_ids = batch['input_ids'].to(device, non_blocking=True)
    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
    
    labels = input_ids.clone()
    labels[~attention_mask.bool()] = 0  # Use 0 (pad) as ignore index
    
    return input_ids, attention_mask, labels

def evaluate_model(model, eval_loader, device, max_batches=20):
    """Quick evaluation"""
    model.eval()
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= max_batches:
                break
                
            input_ids, attention_mask, labels = prepare_batch(batch, device)
            outputs = model(input_ids, attention_mask, labels=labels)
            total_loss += outputs['loss'].item()
            total_batches += 1
    
    return total_loss / total_batches if total_batches > 0 else float('inf')

def analyze_expert_communication(model):
    """Analyze how experts are learning to communicate"""
    comm_data = model.analyze_expert_communication()
    
    print("\n🧠 Expert Communication Analysis:")
    print("="*50)
    
    for layer_name, matrices in comm_data.items():
        print(f"\n{layer_name.upper()}:")
        for i, matrix in enumerate(matrices):
            print(f"  GNN Layer {i+1} - Expert connectivity:")
            connectivity = matrix.cpu().numpy()
            
            # Show connectivity strength between experts
            for expert_i in range(connectivity.shape[0]):
                connections = []
                for expert_j in range(connectivity.shape[1]):
                    if expert_i != expert_j:
                        strength = connectivity[expert_i][expert_j]
                        connections.append(f"E{expert_j}:{strength:.3f}")
                
                print(f"    Expert {expert_i} → [{', '.join(connections)}]")
    
    return comm_data

def plot_training_results(stats, comm_data):
    """Plot training results and expert communication"""
    fig = plt.figure(figsize=(16, 12))
    
    # Training curves
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(stats['train_loss'])
    plt.title('GNN-MoE Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    
    ax2 = plt.subplot(2, 3, 2)
    if stats['eval_loss']:
        plt.plot(stats['eval_step'], stats['eval_loss'], 'o-', color='orange')
        plt.title('Evaluation Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(True)
    
    ax3 = plt.subplot(2, 3, 3)
    plt.plot(stats['grad_norm'])
    plt.title('Gradient Norms')
    plt.xlabel('Step')
    plt.ylabel('Norm')
    plt.grid(True)
    
    # Expert communication heatmaps
    for layer_idx, (layer_name, matrices) in enumerate(comm_data.items()):
        if layer_idx >= 3:  # Only show first 3 layers
            break
            
        ax = plt.subplot(2, 3, 4 + layer_idx)
        # Show the first GNN layer's connectivity for this transformer layer
        connectivity = matrices[0].cpu().numpy()
        
        im = plt.imshow(connectivity, cmap='Blues', vmin=0, vmax=1)
        plt.title(f'{layer_name} Expert Connectivity')
        plt.xlabel('To Expert')
        plt.ylabel('From Expert')
        plt.colorbar(im)
        
        # Add text annotations
        for i in range(connectivity.shape[0]):
            for j in range(connectivity.shape[1]):
                plt.text(j, i, f'{connectivity[i,j]:.2f}', 
                        ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('gnn_moe_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main training and testing function"""
    global device
    device = setup_device()
    
    if device.type == "mps":
        print("🚀 Using Apple MPS (Metal Performance Shaders)")
    elif device.type == "cuda":
        print("🚀 Using CUDA")
    else:
        print("🚀 Using CPU")
    
    print("🧠 GNN-Coupled MoE Innovation Test")
    print("="*50)
    
    # Configuration
    config = GNNMoEConfig()
    
    # Create datasets
    print("📊 Creating synthetic datasets...")
    train_dataset = UltraSimpleDataset(
        vocab_size=config.vocab_size, 
        max_length=config.max_seq_length, 
        num_samples=2000
    )
    eval_dataset = UltraSimpleDataset(
        vocab_size=config.vocab_size, 
        max_length=config.max_seq_length, 
        num_samples=500
    )
    
    # Data loaders (removed num_workers to prevent repeated MPS messages)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)
    
    # Create model
    print(f"🚀 Creating GNN-MoE model...")
    model = GNNMoEModel(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"📊 Parameters: {param_count:,}")
    print(f"🧠 Experts per layer: {config.num_experts}")
    print(f"🔗 GNN layers for coordination: {config.gnn_layers}")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    
    # Training loop
    print(f"\n🏋️ Starting training...")
    stats = defaultdict(list)
    best_eval_loss = float('inf')
    step = 0
    start_time = time.time()
    
    num_epochs = 8
    max_batches_per_epoch = 50
    eval_every = 25
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= max_batches_per_epoch:
                break
            
            # Training step
            input_ids, attention_mask, labels = prepare_batch(batch, device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs['loss']
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Track stats
            step += 1
            epoch_loss += loss.item()
            epoch_steps += 1
            
            stats['train_loss'].append(loss.item())
            stats['grad_norm'].append(grad_norm.item())
            stats['learning_rate'].append(scheduler.get_last_lr()[0])
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'grad_norm': f'{grad_norm.item():.2f}'
            })
            
            # Periodic evaluation
            if step % eval_every == 0:
                eval_loss = evaluate_model(model, eval_loader, device)
                stats['eval_loss'].append(eval_loss)
                stats['eval_step'].append(step)
                
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    print(f"\n🎯 New best eval loss: {eval_loss:.4f} at step {step}")
                
                model.train()
        
        # Epoch summary
        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        elapsed = (time.time() - start_time) / 60
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Time: {elapsed:.1f}m")
    
    print("✅ Training complete!")
    print(f"🎯 Best eval loss: {best_eval_loss:.4f}")
    print(f"⏱️ Total time: {(time.time() - start_time)/60:.1f} minutes")
    
    # Analyze expert communication
    comm_data = analyze_expert_communication(model)
    
    # Plot results
    plot_training_results(stats, comm_data)
    
    print(f"\n🧠 GNN-MoE Innovation Summary:")
    print(f"✅ {config.num_experts} experts per layer, ALL active")
    print(f"✅ GNN coordination instead of sparse routing")
    print(f"✅ Learnable expert communication patterns")
    print(f"✅ {param_count:,} parameters trained successfully")
    print(f"✅ Results saved to gnn_moe_results.png")
    
    return model, stats, comm_data

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Run the test
    model, stats, comm_data = main()
