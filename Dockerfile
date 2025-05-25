# Dockerfile for Bijective Discrete Diffusion Training
# Optimized for 2x RTX 4070 (24GB VRAM) + 128GB RAM workstation

FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for CUDA and multi-GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 11.8 support (optimized for RTX 4070)
RUN pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install core ML dependencies
RUN pip3 install \
    transformers==4.35.0 \
    datasets==2.14.0 \
    tokenizers==0.14.1 \
    accelerate==0.24.0 \
    wandb==0.16.0 \
    tensorboard==2.15.0 \
    scikit-learn==1.3.0 \
    matplotlib==3.7.0 \
    seaborn==0.12.0 \
    pandas==2.1.0 \
    numpy==1.24.0 \
    scipy==1.11.0 \
    tqdm==4.66.0 \
    pyyaml==6.0 \
    omegaconf==2.3.0

# Install additional dependencies for bijective transformers
RUN pip3 install \
    einops==0.7.0 \
    flash-attn==2.3.0 \
    xformers==0.0.22 \
    deepspeed==0.11.0

# Set working directory
WORKDIR /workspace

# Create directories for data, models, and logs
RUN mkdir -p /workspace/data /workspace/models /workspace/logs /workspace/experiments

# Copy project files
COPY . /workspace/

# Install project in development mode
RUN pip3 install -e .

# Set environment variables for optimal performance
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=16
ENV MKL_NUM_THREADS=16

# Optimize for multi-GPU training
ENV NCCL_DEBUG=INFO
ENV NCCL_TREE_THRESHOLD=0

# Set default command
CMD ["/bin/bash"]

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" || exit 1
