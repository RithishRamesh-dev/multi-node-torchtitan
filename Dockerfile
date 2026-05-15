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
# Upgrade pip tooling (but NOT torch itself)
# -------------------------
RUN pip install --upgrade pip setuptools wheel

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
# Install TorchTitan
# -------------------------
RUN git clone https://github.com/pytorch/torchtitan.git /workspace/torchtitan

WORKDIR /workspace/torchtitan

# Install TorchTitan dependencies (without upgrading torch)
RUN pip install -r requirements.txt --no-deps || true

# Install package in editable mode
RUN pip install -e . --no-deps

# -------------------------
# Verify critical import
# -------------------------
RUN python -c "from torch.distributed.checkpoint import HuggingFaceStorageWriter; print('Checkpoint API OK')"
RUN python -c "import torch; print('PyTorch version:', torch.__version__)"

WORKDIR /workspace