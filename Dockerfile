FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Clone TorchTitan first
RUN git clone https://github.com/pytorch/torchtitan.git /workspace/torchtitan

WORKDIR /workspace/torchtitan

# Install TorchTitan's non-torch dependencies only
RUN pip install -r requirements.txt --no-deps || true

# Install package in editable mode without pulling in pinned torch
RUN pip install -e . --no-deps

# Install supporting packages (no torch dependency)
RUN pip install \
    numpy packaging typing_extensions pyyaml \
    protobuf sentencepiece accelerate \
    huggingface_hub safetensors fsspec

# NOW install PyTorch nightly LAST — this wins over anything above
# Use cu124 to match the base image's CUDA 12.4
RUN pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu124 \
    --force-reinstall

# Verify
RUN python -c "from torch.distributed.checkpoint import HuggingFaceStorageWriter; print('OK')"
RUN python -c "import torch; print('PyTorch:', torch.__version__)"

WORKDIR /workspace