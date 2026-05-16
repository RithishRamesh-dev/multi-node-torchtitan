FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel

RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Clone TorchTitan
RUN git clone https://github.com/pytorch/torchtitan.git /workspace/torchtitan

WORKDIR /workspace/torchtitan

# Step 1: Install PyTorch nightly (cu126 matches base image CUDA 12.6)
RUN pip install --pre torch \
    --index-url https://download.pytorch.org/whl/nightly/cu126 \
    --force-reinstall

# Step 2: Install smart_open first (torchdata dependency missing from index)
RUN pip install smart_open

# Step 3: Install torchdata nightly without dependency resolution
RUN pip install --pre torchdata \
    --index-url https://download.pytorch.org/whl/nightly/cpu \
    --no-deps

# Step 4: Install TorchTitan deps without touching torch
RUN pip install -r requirements.txt --no-deps || true
RUN pip install -e . --no-deps

# Step 5: Supporting packages
RUN pip install \
    numpy packaging typing_extensions pyyaml \
    protobuf sentencepiece accelerate \
    huggingface_hub safetensors fsspec datasets \
    docstring_parser tyro

# Verify
RUN python -c "from torch.distributed.checkpoint import HuggingFaceStorageWriter; print('HuggingFaceStorageWriter OK')"
RUN python -c "import torch; print('PyTorch:', torch.__version__)"

WORKDIR /workspace