FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Clone TorchTitan
RUN git clone https://github.com/pytorch/torchtitan.git /workspace/torchtitan

WORKDIR /workspace/torchtitan

# Install TorchTitan deps without pulling in its pinned torch
RUN pip install -r requirements.txt --no-deps || true
RUN pip install -e . --no-deps

# Install supporting packages
RUN pip install \
    numpy packaging typing_extensions pyyaml \
    protobuf sentencepiece accelerate \
    huggingface_hub safetensors fsspec

# Install torch nightly ONLY (no torchvision/torchaudio — not needed for LLM training)
# Pin to a specific nightly date to avoid mismatched package versions
RUN pip install --pre torch \
    --index-url https://download.pytorch.org/whl/nightly/cu124 \
    --force-reinstall

# Verify
RUN python -c "from torch.distributed.checkpoint import HuggingFaceStorageWriter; print('HuggingFaceStorageWriter OK')"
RUN python -c "import torch; print('PyTorch:', torch.__version__)"

WORKDIR /workspace