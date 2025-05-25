#!/usr/bin/env python3
"""
Generate the complete self-contained Bijective Discrete Diffusion notebook.
"""

import json

def create_notebook():
    """Create the complete notebook structure."""
    
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Title cell
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 Bijective Discrete Diffusion - Self-Contained Training Notebook\n",
            "\n",
            "**Complete Implementation**: All code inline, no external dependencies!\n",
            "\n",
            "## 🎯 Features\n",
            "- ✅ **100% Self-Contained**: All model code included inline\n",
            "- ✅ **Easy Scaling**: Simple configuration for model size & datasets\n",
            "- ✅ **Extensive Logging**: Real-time metrics, plots, and progress tracking\n",
            "- ✅ **Google Drive Integration**: Persistent checkpoints\n",
            "- ✅ **Interactive Generation**: Test your model during training\n",
            "- ✅ **No External Dependencies**: No git cloning or src imports\n",
            "\n",
            "---\n",
            "\n",
            "**⚡ Ready to train? Just run all cells!**"
        ]
    })
    
    # Configuration section
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 🎛️ Master Configuration\n\n**Change these settings to scale your model and customize training!**"]
    })
    
    # Configuration code
    config_code = '''# ==================== 🎛️ EASY CONFIGURATION ====================
# Change these parameters to scale up/modify your training!

# 📚 DATASET CONFIGURATION
DATASET_CONFIG = {
    "name": "wikitext-2",          # Options: "wikitext-2", "wikitext-103", "openwebtext", "custom"
    "subset": "wikitext-2-raw-v1", # Dataset subset (for wikitext)
    "custom_path": None,           # Path for custom datasets
    "max_length": 256,             # Sequence length (increase for longer context)
    "min_length": 10,              # Filter short sequences
    "validation_split": 0.1,       # For datasets without validation
    "cache_dir": "/content/data_cache"  # Cache downloaded data
}

# 🧠 MODEL SCALING (Easy Presets + Custom)
MODEL_SIZE = "small"               # Options: "tiny", "small", "medium", "large", "xl", "custom"

# Predefined model sizes (automatically configured)
MODEL_PRESETS = {
    "tiny":   {"embed_dim": 128,  "num_layers": 4,  "num_heads": 4,  "ff_mult": 4},   # ~5M params
    "small":  {"embed_dim": 256,  "num_layers": 6,  "num_heads": 8,  "ff_mult": 4},   # ~20M params  
    "medium": {"embed_dim": 512,  "num_layers": 8,  "num_heads": 8,  "ff_mult": 4},   # ~80M params
    "large":  {"embed_dim": 768,  "num_layers": 12, "num_heads": 12, "ff_mult": 4},   # ~200M params
    "xl":     {"embed_dim": 1024, "num_layers": 16, "num_heads": 16, "ff_mult": 4}    # ~400M params
}

# Custom model configuration (used if MODEL_SIZE == "custom")
CUSTOM_MODEL = {
    "embed_dim": 512,
    "num_layers": 8, 
    "num_heads": 8,
    "ff_mult": 4,
    "likelihood_weight": 0.1,
    "hybrid_layers": None  # e.g., [0, 2, 4] to make only specific layers bijective
}

# 🏃 TRAINING CONFIGURATION
TRAINING_CONFIG = {
    "epochs": 10,
    "batch_size": "auto",              # "auto" optimizes for your GPU, or set manually
    "learning_rate": 1e-4,
    "warmup_steps": 1000,
    "gradient_accumulation_steps": 1,   # Increase for larger effective batch
    "checkpoint_every": 2,              # Save every N epochs
    "eval_every": 1,                    # Evaluate every N epochs
    "log_every": 10,                    # Log metrics every N batches
    "max_grad_norm": 1.0,               # Gradient clipping
    "weight_decay": 0.01,               # L2 regularization
    "batches_per_epoch": None           # Limit batches per epoch (None = use all)
}

# 🖥️ HARDWARE OPTIMIZATION
HARDWARE_CONFIG = {
    "gpu_type": "auto",          # "auto", "T4", "A100", "V100", "RTX4090"
    "mixed_precision": True,     # Use automatic mixed precision (faster)
    "optimize_memory": True,     # Enable memory optimizations
    "compile_model": False,      # PyTorch 2.0 compilation (faster but more memory)
    "gradient_checkpointing": False  # Trade compute for memory (for large models)
}

# 📊 LOGGING & VISUALIZATION  
LOGGING_CONFIG = {
    "detailed_metrics": True,    # Extensive logging
    "plot_frequency": 50,        # Update plots every N batches
    "save_samples": True,        # Save generation samples
    "track_gradients": False,    # Monitor gradient norms (memory intensive)
    "use_tensorboard": False,    # Enable TensorBoard logging
    "wandb_project": None        # W&B project name (None to disable)
}

# 💾 STORAGE CONFIGURATION
STORAGE_CONFIG = {
    "use_google_drive": True,    # Save to Google Drive
    "checkpoint_dir": "/content/drive/MyDrive/bijective_checkpoints",
    "export_dir": "/content/drive/MyDrive/bijective_exports",
    "local_checkpoint_dir": "/content/checkpoints",  # Fallback if no Drive
    "keep_n_checkpoints": 3      # Keep only N most recent checkpoints
}

print("✅ Configuration loaded!")
print(f"📊 Model size: {MODEL_SIZE}")
print(f"📚 Dataset: {DATASET_CONFIG['name']}")
print(f"🏃 Training for {TRAINING_CONFIG['epochs']} epochs")'''
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": config_code.split('\n')
    })
    
    # Continue adding all cells...
    # (Due to length, I'll save this and continue in the next step)
    
    return notebook

def save_notebook(notebook, filename):
    """Save notebook to file."""
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f"✅ Notebook saved to {filename}")

if __name__ == "__main__":
    notebook = create_notebook()
    save_notebook(notebook, "Bijective_Discrete_Diffusion_Self_Contained.ipynb")
