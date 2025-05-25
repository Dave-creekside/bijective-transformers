#!/bin/bash

# Quick fix script for environment compatibility issues
# Run this to update the environment with corrected dependencies

set -e

echo "🔧 Fixing Bijective Transformers environment compatibility issues..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed or not in PATH"
    exit 1
fi

# Check if environment exists
ENV_NAME="bijective-transformers"
if ! conda env list | grep -q "^${ENV_NAME} "; then
    print_error "Environment '${ENV_NAME}' does not exist. Please run ./setup.sh first."
    exit 1
fi

print_status "Updating environment with fixed dependencies..."

# Update the environment with the corrected environment.yml
conda env update -n ${ENV_NAME} -f environment.yml --prune

print_success "Environment updated successfully!"

# Activate environment and test
print_status "Activating environment and running verification..."

# Source conda and activate environment
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Run verification script
print_status "Running verification script..."
python verify_setup.py

print_success "Environment fix complete!"
echo
print_status "If verification still shows issues, you may need to:"
echo "  1. Remove and recreate the environment: conda env remove -n ${ENV_NAME} && ./setup.sh"
echo "  2. Check for any remaining compatibility warnings"
echo "  3. The core functionality should work even with minor warnings"
