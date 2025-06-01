# Project Report: GNN-Coupled Mixture of Experts (GNN-MoE) Language Model

**Date:** June 1, 2025

**Prepared by:** Cline (AI Software Engineer)

## 1. Project Goal and Concept

*   **Objective:** To develop and evaluate a novel Transformer-based language model that utilizes a Mixture of Experts (MoE) architecture. The key innovation is the use of Graph Neural Networks (GNNs) to facilitate dynamic, learned communication and coordination between the experts within each MoE layer.
*   **Hypothesis:** GNN-based coordination can lead to more effective expert specialization and utilization compared to traditional sparse MoE routing mechanisms, potentially offering better performance for a given parameter count or improved efficiency. A notable feature observed is that all experts are active in every layer, with the GNNs learning to weight their contributions.
*   **Core Architecture:**
    *   Standard Transformer blocks (self-attention, Feed-Forward Network) form the "experts."
    *   Multiple experts exist per MoE layer.
    *   A GNN-based "coupler" module takes the outputs of all experts in a layer.
    *   The GNN coupler consists of one or more `ExpertGraphConv` layers. Each `ExpertGraphConv` layer allows experts to exchange information (messages) based on learnable adjacency logits (representing inherent connectivity preferences) and content-aware message weighting.
    *   The GNN-processed expert features are then combined (e.g., by averaging and passing through a final linear layer) to produce the output of the `GNNMoELayer`.
    *   The model is a decoder-only Transformer, trained on a standard causal language modeling objective (predicting the next token).

## 2. Current State of the Build

*   **Codebase Structure (Successfully Refactored):**
    *   The project has been modularized from an initial Jupyter notebook / single script into a set of Python files:
        *   `gnn_moe_config.py`: Defines `GNNMoEConfig` dataclass for all hyperparameters.
        *   `gnn_moe_architecture.py`: Contains all PyTorch `nn.Module` classes for the model (e.g., `ExpertGraphConv`, `ExpertBlock`, `GNNExpertCoupler`, `GNNMoELayer`, `GNNMoEModel`).
        *   `gnn_moe_data.py`: Handles data loading (`SimpleTextDataset`, `load_data` function using Hugging Face `datasets`). Includes fallback to synthetic data if real dataset loading fails.
        *   `gnn_moe_training.py`: Contains the main training loop (`train_gnn_moe`), evaluation function (`evaluate_model`), checkpointing (`save_checkpoint`, `load_checkpoint`), and batch preparation.
        *   `gnn_moe_analysis.py`: Includes functions for analyzing expert communication, plotting training results and connectivity matrices, and calculating model efficiency.
        *   `run_gnn_moe.py`: The main executable script. It uses `argparse` for comprehensive command-line configuration of all aspects of a training run (model architecture, training params, dataset, checkpointing, run naming). It orchestrates the components from the other modules.
        *   `sweep_gnn_moe.py`: A script to automate hyperparameter sweeps. It calls `run_gnn_moe.py` as a subprocess with different configurations, selected via a `--sweep_name` argument. It logs results to a CSV file.
*   **Functionality Implemented and Verified:**
    *   **Model Training:** The model trains successfully on both MPS (Apple Silicon) and CUDA (A100 GPUs).
    *   **Data Loading:** Can load standard datasets (e.g., WikiText-2, SST-2) via Hugging Face `datasets` (though currently facing issues in the sweep context).
    *   **Checkpointing:** Robust saving and loading of model, optimizer, scheduler, and config state is implemented, allowing resumption of training.
    *   **Hyperparameter Configuration:** `run_gnn_moe.py` allows nearly all relevant parameters to be set via CLI.
    *   **Analysis & Plotting:** Scripts generate plots for training loss/PPL, learning rate, gradient norms, and GNN expert connectivity matrices. Model parameter efficiency is also reported.
    *   **Sweep Automation:** `sweep_gnn_moe.py` can run series of experiments.
*   **Performance Baseline (Current ~40M Parameter Model):**
    *   On a subset of WikiText-2 (18k training samples, 1.5k eval samples), the model (256 embed dim, 4 layers, 4 experts, 2 GNN layers) achieved a **Best Eval Perplexity of ~88.77** (Eval Loss ~4.4860) after 10 epochs.
    *   Training time for this on an A100 was ~15 minutes.
*   **Key Architectural Finding (So Far):**
    *   The GNN expert connectivity patterns (adjacency matrices) appear to be remarkably stable across different datasets, data sizes, hardware, and even very early in training.

## 3. Specific Error We Keep Running Into (The Current Blocker)

*   **The Issue:** The `load_data` function in `gnn_moe_data.py` is currently **failing to load the real WikiText-2 dataset when executed via `sweep_gnn_moe.py` on Colab, and is instead falling back to using synthetic data.**
*   **Symptom:** The sweep runs very fast, and the output indicates "SYNTHETIC_FALLBACK" mode instead of "REAL\_WIKITEXT\_2\_V1".
*   **Why this is a problem:** Experiments run with synthetic data are not meaningful for evaluating the model's performance on actual language tasks or for comparing different hyperparameter configurations.
*   **Immediate Impact:** The hyperparameter sweeps are not producing valid results.
*   **History of this type of error:** Previous data loading issues were related to incorrect dataset names or `load_dataset` parameters. The current issue seems specific to the Colab environment during sweep subprocess execution.
*   **What we don't know yet:** The *exact exception* from `hf_datasets.load_dataset` when it fails in this context.

## 4. Next Immediate Steps (To Resolve the Blocker)

1.  **Diagnose the Data Loading Failure:**
    *   Obtain the specific error message and traceback from `hf_datasets.load_dataset` when it fails during a `run_gnn_moe.py` execution on Colab (ideally one launched by `sweep_gnn_moe.py`, or a direct run replicating the environment).
    *   This may involve temporarily modifying `gnn_moe_data.py` to print `traceback.print_exc()` or re-raise the exception.
2.  **Fix the Root Cause:** Based on the error, address potential issues such as:
    *   Colab environment problems (disk space, network, Hugging Face Hub authentication).
    *   Dataset caching issues in Colab.
    *   Problems with the `datasets` library version or its dependencies in Colab.

## 5. Once Data Loading is Reliable Again

*   Resume the planned hyperparameter sweeps (e.g., `embed_dim_sweep`, `num_experts_sweep`) using `sweep_gnn_moe.py`.
*   Analyze the CSV results to understand scaling laws for the GNN-MoE architecture.
*   Proceed with larger model configurations, training on the full WikiText-2, and then potentially WikiText-103.
*   Implement and train baseline models (standard Transformer) for comparison.

## Summary of AI Assistant's Role and Errors

*   **Role:** To assist in implementing the research idea, refactoring code, adding features (checkpointing, hyperparameterization), and debugging issues.
*   **Errors by AI:**
    *   Generating syntactically incorrect code, requiring multiple iterations to fix.
    *   Misinterpreting requests (e.g., for separate sweep scripts).
    *   Introducing minor bugs (e.g., `NameError`s) during refactoring.
    *   Not being sufficiently proactive in suggesting robust error reporting for critical functions like data loading.

Efforts are ongoing to improve reliability and address the current data loading blocker to enable effective experimentation.
