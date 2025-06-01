#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_gnn_moe.py

Main executable script for GNN-Coupled MoE model training and experimentation.
Uses modularized components for config, architecture, data, training, and analysis.
Allows setting model architecture, training params, and dataset via CLI.
"""

import torch
import torch.optim as optim
import os
import argparse
from dataclasses import fields # To iterate over dataclass fields

# Import from our new modules
from gnn_moe_config import GNNMoEConfig
from gnn_moe_architecture import GNNMoEModel
from gnn_moe_data import load_data
from gnn_moe_training import setup_environment, load_checkpoint, train_gnn_moe
from gnn_moe_analysis import plot_training_results, analyze_expert_communication, plot_expert_connectivity, analyze_model_efficiency

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN-MoE Hyperparameter Training Script")
    
    # Dynamically add arguments from GNNMoEConfig defaults
    # This makes it easy to update GNNMoEConfig and have CLI args reflect changes.
    config_fields = {f.name: f for f in fields(GNNMoEConfig)}

    # Model Architecture Args
    parser.add_argument('--embed_dim', type=int, default=config_fields['embed_dim'].default, help="Embedding dimension")
    parser.add_argument('--num_layers', type=int, default=config_fields['num_layers'].default, help="Number of GNNMoELayers")
    parser.add_argument('--num_heads', type=int, default=config_fields['num_heads'].default, help="Number of attention heads")
    parser.add_argument('--dropout_rate', type=float, default=config_fields['dropout_rate'].default, help="Dropout rate")
    parser.add_argument('--num_experts', type=int, default=config_fields['num_experts'].default, help="Number of experts per layer")
    parser.add_argument('--gnn_layers', type=int, default=config_fields['gnn_layers'].default, help="Number of GNN layers in coupler")

    # Training Hyperparameters Args
    parser.add_argument('--batch_size', type=int, default=config_fields['batch_size'].default, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=config_fields['learning_rate'].default, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=config_fields['epochs'].default, help="Number of epochs")
    parser.add_argument('--max_batches_per_epoch', type=int, default=config_fields['max_batches_per_epoch'].default, help="Max batches per epoch (-1 for full epoch)")
    parser.add_argument('--eval_every', type=int, default=config_fields['eval_every'].default, help="Evaluate every N steps")

    # Dataset Args
    parser.add_argument('--dataset_name', type=str, default=config_fields['dataset_name'].default, help="Hugging Face dataset name")
    parser.add_argument('--dataset_config_name', type=str, default=config_fields['dataset_config_name'].default, help="Hugging Face dataset config name")
    parser.add_argument('--num_train_samples', type=int, default=config_fields['num_train_samples'].default, help="Number of training samples (-1 for all)")
    parser.add_argument('--num_eval_samples', type=int, default=config_fields['num_eval_samples'].default, help="Number of evaluation samples (-1 for all)")

    # Checkpointing & Output Args
    parser.add_argument('--checkpoint_dir', type=str, default=config_fields['checkpoint_dir'].default, help="Base directory for checkpoints.")
    parser.add_argument('--resume_checkpoint', type=str, default=config_fields['resume_checkpoint'].default, help="Path to checkpoint to resume training from.")
    parser.add_argument('--run_name', type=str, default=config_fields['run_name'].default, help="Optional run name for outputs subdir")
    
    # Technical Args
    parser.add_argument('--seed', type=int, default=config_fields['seed'].default, help="Random seed")
    parser.add_argument('--num_workers_dataloader', type=int, default=config_fields['num_workers_dataloader'].default, help="Num workers for DataLoader")

    args = parser.parse_args()

    # Create a config instance and override with CLI arguments
    cfg = GNNMoEConfig()
    for arg_name, arg_val in vars(args).items():
        if hasattr(cfg, arg_name) and arg_val is not None: # Check if arg is a field in GNNMoEConfig
            # Special handling for checkpoint_dir which is base, run_name makes it specific
            if arg_name not in ['run_name', 'resume_checkpoint', 'checkpoint_dir']: # these are handled separately or are not direct cfg overrides in the same way
                 print(f"Overriding config.{arg_name} with CLI arg: {arg_val}")
            setattr(cfg, arg_name, arg_val)
    
    # Handle run_name and checkpoint_dir logic
    # cfg.checkpoint_dir is initialized with the default "checkpoints" or the CLI override for the base.
    # If run_name is given, create a subdirectory.
    base_checkpoint_dir_from_arg = args.checkpoint_dir # This is what user specified for base, or its default
    if cfg.run_name:
        cfg.checkpoint_dir = os.path.join(base_checkpoint_dir_from_arg, cfg.run_name)
    else: # No run_name, use the checkpoint_dir as is (could be default "checkpoints" or user-specified base)
        cfg.checkpoint_dir = base_checkpoint_dir_from_arg
    
    if not os.path.exists(cfg.checkpoint_dir):
        os.makedirs(cfg.checkpoint_dir)
        print(f"📁 Created checkpoint directory: {cfg.checkpoint_dir}")

    print("===== GNN-MoE Hyperparameter Script Execution Started =====")
    if cfg.run_name: print(f"Run Name: {cfg.run_name}")
    print(f"Effective Config: {cfg}")

    selected_device = setup_environment(cfg) # setup_environment uses cfg.seed and creates default ./plots
    
    train_loader, eval_loader, tokenizer, data_mode = load_data(cfg) # load_data uses cfg for dataset params
    
    # vocab_size in cfg might be updated by load_data if tokenizer's vocab is different
    print(f"\n🏗️ Creating GNN-MoE Model with effective vocab_size: {cfg.vocab_size}")
    model = GNNMoEModel(cfg).to(selected_device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    
    # Calculate total_steps for scheduler
    actual_batches_per_epoch_main = len(train_loader) if cfg.max_batches_per_epoch == -1 else min(len(train_loader), cfg.max_batches_per_epoch)
    if actual_batches_per_epoch_main == 0 and cfg.num_train_samples != 0 : # If loader is empty but samples were requested
        print(f"ERROR: Train loader has 0 batches. Check dataset path and num_train_samples ({cfg.num_train_samples}).")
        exit(1)
    total_steps_main = cfg.epochs * actual_batches_per_epoch_main
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps_main, eta_min=1e-6) if total_steps_main > 0 else None

    start_epoch = 0
    current_step = 0
    best_eval_loss_resumed = float('inf')

    if cfg.resume_checkpoint:
        if os.path.isfile(cfg.resume_checkpoint):
            print(f"Attempting to resume from: {cfg.resume_checkpoint}")
            # load_checkpoint expects model, optimizer, scheduler to be defined
            resume_data = load_checkpoint(cfg.resume_checkpoint, model, optimizer, scheduler)
            if resume_data:
                start_epoch, current_step, best_eval_loss_resumed = resume_data
            model.to(selected_device) # Ensure model is on correct device after loading
        else:
            print(f"⚠️ Resume checkpoint not found: '{cfg.resume_checkpoint}'. Starting fresh.")

    # cfg.checkpoint_dir is already set to the specific run's directory if run_name was provided
    training_stats, final_best_loss = train_gnn_moe(
        model, train_loader, eval_loader, selected_device, cfg, 
        resume_from_epoch=start_epoch,
        resume_step=current_step,
        initial_best_loss=best_eval_loss_resumed
    )
    
    if training_stats: 
        plot_training_results(training_stats, cfg) # Uses cfg.run_name for plot filename
        
        best_model_path = os.path.join(cfg.checkpoint_dir, "best_model.pth.tar")
        if os.path.exists(best_model_path):
            print(f"🔄 Loading best model from {best_model_path} for final analysis...")
            # Create a new model instance with the same config to load best weights
            final_analysis_model = GNNMoEModel(cfg).to(selected_device)
            load_checkpoint(best_model_path, final_analysis_model) # Loads model_state_dict
            
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
    if cfg.run_name: print(f"   Run Name: {cfg.run_name}")
    print(f"   Data Mode: {data_mode}")
    print(f"   Best Eval Loss from run: {final_best_loss:.4f}")
    
    if training_stats and 'eval_perplexity' in training_stats and training_stats['eval_perplexity']:
        try:
            # Find perplexity corresponding to the final_best_loss
            if final_best_loss in training_stats['eval_loss']:
                 best_loss_idx = training_stats['eval_loss'].index(final_best_loss)
                 best_ppl_from_run = training_stats['eval_perplexity'][best_loss_idx]
                 print(f"   Best Eval Perplexity from run: {best_ppl_from_run:.2f}")
            elif training_stats['eval_perplexity']: 
                 print(f"   Min Eval Perplexity from run (fallback): {min(training_stats['eval_perplexity']):.2f}")
        except (ValueError, IndexError): # Should not happen if final_best_loss is from stats
             if training_stats['eval_perplexity']: 
                 print(f"   Min Eval Perplexity from run (exception fallback): {min(training_stats['eval_perplexity']):.2f}")
    print("==============================================")
else:
    print("GNN-MoE Hyperparameter script (run_gnn_moe.py) imported as a module.")
