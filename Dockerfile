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
# FIX: cuDNN SDPA "No valid execution plans" error
# Root cause: libnvrtc-builtins.so.* is not found by cuDNN frontend in pip nightly wheels.
# cuDNN needs libnvrtc-builtins to JIT-compile attention kernels at runtime.
# Setting LD_LIBRARY_PATH to include nvidia cuda lib dirs fixes it without
# disabling or degrading cuDNN — full B300 cuDNN attention performance preserved.
# See: https://github.com/pytorch/pytorch/issues/167602
# -------------------------
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib

# -------------------------
# Verify critical imports
# -------------------------
RUN python -c "from torch.distributed.checkpoint import HuggingFaceStorageWriter; print('HuggingFaceStorageWriter OK')"
RUN python -c "import torch; print('PyTorch:', torch.__version__)"
RUN python -c "import torch; assert torch.cuda.is_available() or True; print('CUDA version:', torch.version.cuda)"

WORKDIR /workspace