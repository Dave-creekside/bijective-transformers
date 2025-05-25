#!/usr/bin/env python3
"""
Setup script for bijective-transformers package.
Fallback for environments that don't support pyproject.toml editable installs.
"""

from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file."""
    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_file):
        with open(req_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "tokenizers>=0.13.0",
        "accelerate>=0.20.0",
        "einops>=0.6.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
    ]

# Read README for long description
def read_readme():
    """Read README.md for long description."""
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return "First working implementation of bijective transformers for discrete diffusion"

setup(
    name="bijective-transformers",
    version="1.0.0",
    description="First working implementation of bijective transformers for discrete diffusion with exact likelihood computation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Bijective Transformers Team",
    url="https://github.com/your-username/bijective-transformers",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.4.0",
            "ipywidgets>=7.7.0",
        ],
        "monitoring": [
            "wandb>=0.13.0",
            "tensorboard>=2.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-bijective=train_bijective_with_checkpoints:main",
            "train-bijective-workstation=train_bijective_workstation:main",
            "test-checkpoint-system=test_checkpoint_system:test_checkpoint_system",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "bijective",
        "transformers", 
        "discrete diffusion",
        "exact likelihood",
        "invertible neural networks",
        "text generation"
    ],
    include_package_data=True,
    zip_safe=False,
)
