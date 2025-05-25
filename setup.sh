#!/bin/bash

# Bijective Transformers Project Setup Script
# Optimized for Apple M3 Mac

set -e  # Exit on any error

echo "🚀 Setting up Bijective Transformers development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed. Please install Miniconda or Anaconda first."
    print_status "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

print_status "Conda found: $(conda --version)"

# Check if environment already exists
ENV_NAME="bijective-transformers"
if conda env list | grep -q "^${ENV_NAME} "; then
    print_warning "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        print_status "Updating existing environment..."
        conda env update -n ${ENV_NAME} -f environment.yml
        print_success "Environment updated successfully!"
        exit 0
    fi
fi

# Create conda environment
print_status "Creating conda environment from environment.yml..."
conda env create -f environment.yml

print_success "Environment '${ENV_NAME}' created successfully!"

# Activate environment and set up additional configurations
print_status "Activating environment and configuring for M3 Mac..."

# Source conda and activate environment
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Set environment variables for M3 Mac optimization
print_status "Setting up M3 Mac optimizations..."

# Create activation script for environment variables
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/bash
# M3 Mac optimizations for PyTorch
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Optimize for Apple Silicon
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Weights & Biases setup
export WANDB_SILENT=true

echo "🍎 M3 Mac optimizations activated"
echo "🔥 PyTorch MPS backend enabled"
EOF

cat > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh << 'EOF'
#!/bin/bash
unset PYTORCH_ENABLE_MPS_FALLBACK
unset PYTORCH_MPS_HIGH_WATERMARK_RATIO
unset OMP_NUM_THREADS
unset MKL_NUM_THREADS
unset WANDB_SILENT
EOF

# Make scripts executable
chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

# Test PyTorch MPS availability
print_status "Testing PyTorch MPS availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    print('✅ MPS backend is ready for M3 acceleration!')
else:
    print('⚠️  MPS backend not available')
"

# Test key dependencies
print_status "Testing key dependencies..."
python -c "
try:
    import transformers
    import torch
    import einops
    import nflows
    print('✅ All core dependencies imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

# Create project structure
print_status "Creating project directory structure..."
mkdir -p {src,tests,experiments,data,models,logs,configs,notebooks}

# Create basic project files
cat > src/__init__.py << 'EOF'
"""Bijective Transformers - Core Implementation"""
__version__ = "0.1.0"
EOF

cat > tests/__init__.py << 'EOF'
"""Test suite for Bijective Transformers"""
EOF

print_success "Project structure created!"

# Final instructions
echo
print_success "🎉 Setup complete!"
echo
print_status "To activate the environment in the future, run:"
echo "  conda activate ${ENV_NAME}"
echo
print_status "To deactivate the environment, run:"
echo "  conda deactivate"
echo
print_status "Project structure:"
echo "  src/          - Core implementation"
echo "  tests/        - Test suite"
echo "  experiments/  - Experimental scripts"
echo "  data/         - Datasets"
echo "  models/       - Saved models"
echo "  logs/         - Training logs"
echo "  configs/      - Configuration files"
echo "  notebooks/    - Jupyter notebooks"
echo
print_status "Next steps:"
echo "  1. conda activate ${ENV_NAME}"
echo "  2. Initialize git repository if needed"
echo "  3. Start with Phase 1 implementation"
echo
print_warning "Remember to commit your environment.yml to version control!"
