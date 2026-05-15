FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

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
# Upgrade pip tooling
# -------------------------
RUN pip install --upgrade pip setuptools wheel

# -------------------------
# Install PyTorch nightly (IMPORTANT FIX)
# This is what provides missing distributed checkpoint APIs
# -------------------------
RUN pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu121

# -------------------------
# Install distributed + checkpoint dependencies
# -------------------------
RUN pip install \
    numpy \
    packaging \
    typing_extensions \
    pyyaml \
    protobuf \
    sentencepiece \
    accelerate

# -------------------------
# Install TorchTitan (your training framework)
# -------------------------
RUN git clone https://github.com/pytorch/torchtitan.git /workspace/torchtitan

WORKDIR /workspace/torchtitan

# Install TorchTitan dependencies
RUN pip install -r requirements.txt || true

# Install package in editable mode
RUN pip install -e .

# -------------------------
# Verify critical import exists at build time
# (fails fast if mismatch happens)
# -------------------------
RUN python -c "from torch.distributed.checkpoint import HuggingFaceStorageWriter; print('Checkpoint API OK')"

# -------------------------
# Default workdir
# -------------------------
WORKDIR /workspace