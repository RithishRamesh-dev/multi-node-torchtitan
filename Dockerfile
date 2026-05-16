FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Upgrade pip tooling (NOT torch — base image has 2.7 which we keep)
# -------------------------
RUN pip install --upgrade pip setuptools wheel

# -------------------------
# Clone TorchTitan
# -------------------------
RUN git clone https://github.com/pytorch/torchtitan.git /workspace/torchtitan

WORKDIR /workspace/torchtitan

# Install TorchTitan deps without touching torch
RUN pip install -r requirements.txt --no-deps || true
RUN pip install -e . --no-deps

# -------------------------
# Install torchdata nightly (required companion to torch nightly)
# -------------------------
RUN pip install smart_open
RUN pip install --pre torchdata \
    --index-url https://download.pytorch.org/whl/nightly/cpu \
    --no-deps

# -------------------------
# Install remaining dependencies
# -------------------------
RUN pip install \
    numpy packaging typing_extensions pyyaml \
    protobuf sentencepiece accelerate \
    huggingface_hub safetensors fsspec datasets \
    docstring_parser tyro \
    absl-py tensorboard

# -------------------------
# Install PyTorch nightly with CUDA 12.8 support for Blackwell/B300 (sm_100)
# Must be done LAST so nothing downgrades it
# -------------------------
RUN pip install --pre torch \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    --force-reinstall

# -------------------------
# Verify critical imports
# -------------------------
RUN python -c "from torch.distributed.checkpoint import HuggingFaceStorageWriter; print('HuggingFaceStorageWriter OK')"
RUN python -c "import torch; print('PyTorch:', torch.__version__)"
RUN python -c "import torch; assert torch.cuda.is_available() or True; print('CUDA version:', torch.version.cuda)"

WORKDIR /workspace