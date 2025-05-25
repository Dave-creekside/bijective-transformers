# Bijective Transformers for Discrete Diffusion

A research project exploring the integration of bijective modular systems with discrete diffusion models for improved non-autoregressive text generation.

## 🎯 Project Overview

This project tests the hypothesis that bijective transformations can improve discrete diffusion models for text generation by providing:
- **Exact likelihood computation** instead of variational bounds
- **Perfect information preservation** through invertible transformations
- **Improved denoising quality** via mathematically guaranteed reversibility

### Core Innovation
Unlike autoregressive models where bijective constraints conflict with causal masking, discrete diffusion models process all tokens simultaneously, making them naturally compatible with bijective architectures.

## 🚀 Quick Start

### Prerequisites
- **macOS** (optimized for Apple M3, but adaptable)
- **Conda** or **Miniconda** ([Download here](https://docs.conda.io/en/latest/miniconda.html))
- **Git** for version control

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd bijective-transformers
   ```

2. **Run the setup script:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Activate the environment:**
   ```bash
   conda activate bijective-transformers
   ```

### Manual Installation (Alternative)

If you prefer manual setup or the script doesn't work:

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate bijective-transformers

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, MPS: {torch.backends.mps.is_available()}')"
```

### Pip Installation (Fallback)

If conda is not available:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📁 Project Structure

```
bijective-transformers/
├── src/                    # Core implementation
│   ├── models/            # Model architectures
│   ├── layers/            # Bijective layer implementations
│   ├── diffusion/         # Discrete diffusion components
│   └── utils/             # Utilities and helpers
├── tests/                 # Test suite
├── experiments/           # Experimental scripts and configs
├── data/                  # Datasets (gitignored)
├── models/                # Saved models (gitignored)
├── logs/                  # Training logs (gitignored)
├── configs/               # Configuration files
├── notebooks/             # Jupyter notebooks for analysis
├── project-knowledge/     # Research documentation
├── environment.yml        # Conda environment
├── requirements.txt       # Pip requirements
└── setup.sh              # Setup script
```

## 🧪 Research Phases

### Phase 1: Foundation (Weeks 1-2)
- [x] Environment setup
- [ ] Basic discrete diffusion implementation
- [ ] Bidirectional transformer baseline
- [ ] Evaluation metrics framework

### Phase 2: Bijective Components (Weeks 3-5)
- [ ] Coupling layer implementation
- [ ] Invertibility testing framework
- [ ] Integration with transformer architecture
- [ ] Memory optimization

### Phase 3: Full Integration (Weeks 6-8)
- [ ] Complete bijective diffusion model
- [ ] Training optimization
- [ ] Performance evaluation vs baseline
- [ ] Ablation studies

### Phase 4: Analysis (Weeks 9-12)
- [ ] Comprehensive evaluation
- [ ] Statistical analysis
- [ ] Documentation and writeup

## 🔬 Key Components

### Bijective Layers
- **Coupling Layers**: Invertible transformations for embeddings
- **Invertible Residual Networks**: Exact reversibility
- **Jacobian Computation**: For exact likelihood calculation

### Discrete Diffusion
- **Noise Processes**: Masking, substitution, deletion
- **Bidirectional Attention**: Full sequence context
- **Multi-step Denoising**: Parallel token generation

### Evaluation Framework
- **Text Quality**: BLEU, perplexity, exact match
- **Training Dynamics**: Convergence, stability, efficiency
- **Bijective Properties**: Invertibility error, information preservation

## 🛠 Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Formatting
```bash
black src/ tests/
isort src/ tests/
```

### Linting
```bash
flake8 src/ tests/
```

### Jupyter Notebooks
```bash
jupyter lab
```

## 📊 Monitoring and Logging

### Weights & Biases
```bash
wandb login
# Training runs will automatically log to W&B
```

### TensorBoard
```bash
tensorboard --logdir logs/
```

## 🍎 Apple M3 Optimizations

The project is optimized for Apple M3 Macs with:
- **MPS Backend**: GPU acceleration via Metal Performance Shaders
- **Memory Management**: Optimized for unified memory architecture
- **Environment Variables**: Automatic setup for Apple Silicon

### Verifying M3 Setup
```python
import torch
print(f"MPS Available: {torch.backends.mps.is_available()}")
print(f"MPS Built: {torch.backends.mps.is_built()}")

# Test MPS device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.randn(10, 10).to(device)
    print(f"✅ MPS device working: {x.device}")
```

## 🔧 Troubleshooting

### Common Issues

**Environment Creation Fails:**
```bash
# Clean conda cache
conda clean --all
# Try creating environment again
conda env create -f environment.yml
```

**MPS Not Available:**
```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"
# Ensure you have PyTorch >= 2.0 with MPS support
```

**Memory Issues:**
```bash
# Reduce batch size in configs
# Enable gradient checkpointing
# Monitor memory usage with Activity Monitor
```

### Getting Help

1. Check the [project documentation](project-knowledge/)
2. Review [computational considerations](project-knowledge/computational-considerations.md)
3. Examine [debugging framework](project-knowledge/computational-considerations.md#debugging-and-diagnostic-tools)

## 📚 References

### Key Papers
- **D3PM**: "Structured Denoising Diffusion Models in Discrete State-Spaces"
- **RealNVP**: "Density estimation using Real NVP"
- **Diffusion-LM**: "Diffusion-LM Improves Controllable Text Generation"

### Code References
- [nflows](https://github.com/bayesiains/nflows): Normalizing flows in PyTorch
- [FrEIA](https://github.com/VLL-HD/FrEIA): Framework for easily invertible architectures
- [transformers](https://github.com/huggingface/transformers): HuggingFace transformers

## 📄 License

[Add your license here]

## 🤝 Contributing

This is a research project. Contributions, suggestions, and discussions are welcome!

## 📧 Contact

[Add your contact information]

---

**Note**: This is experimental research. The hypothesis may be incorrect, and that's a valid scientific outcome. The focus is on rigorous testing and honest evaluation of results.
