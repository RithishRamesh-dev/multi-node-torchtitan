FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN git clone https://github.com/pytorch/torchtitan.git /workspace/torchtitan

WORKDIR /workspace/torchtitan

RUN pip install -r requirements.txt --no-deps || true
RUN pip install -e . --no-deps

RUN pip install smart_open
RUN pip install --pre torchdata \
    --index-url https://download.pytorch.org/whl/nightly/cpu \
    --no-deps

RUN pip install \
    numpy packaging typing_extensions pyyaml \
    protobuf sentencepiece accelerate \
    huggingface_hub safetensors fsspec datasets \
    docstring_parser tyro \
    absl-py tensorboard

# Install PyTorch nightly cu128 for B300 (sm_100) — must be LAST
RUN pip install --pre torch \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    --force-reinstall

# -------------------------
# B300 (sm_103) SDPA workaround
#
# The NVIDIA B300 GPU has compute capability sm_103. As of May 2026, NO official
# PyTorch release or NVIDIA container includes sm_103 in TORCH_CUDA_ARCH_LIST —
# not PyTorch 2.7, not PyTorch nightly, not NGC 25.10. This is an ecosystem-wide
# gap confirmed by NVIDIA engineers.
#
# Impact: cuDNN Frontend cannot build valid execution plans for SDPA on B300
# because it has no sm_103-compiled kernels. This causes:
#   RuntimeError: cuDNN Frontend error: No valid execution plans built
#
# Fix: Disable cuDNN SDPA and use Flash Attention instead. On B300, Flash
# Attention runs sm_100 (B200-compatible) kernels which work correctly.
# This is NOT a performance regression vs cuDNN since cuDNN SDPA fails entirely.
# It will be resolved when NVIDIA ships sm_103 support in official containers.
#
# Reference: https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.enable_cudnn_sdp
# -------------------------
RUN echo "import torch; torch.backends.cuda.enable_cudnn_sdp(False)" \
    > /opt/conda/lib/python3.11/site-packages/sitecustomize.py

# Verify
RUN python -c "from torch.distributed.checkpoint import HuggingFaceStorageWriter; print('HuggingFaceStorageWriter OK')"
RUN python -c "import torch; print('PyTorch:', torch.__version__)"
RUN python -c "import torch; print('cuDNN SDPA enabled:', torch.backends.cuda.cudnn_sdp_enabled())"
RUN python -c "import torch; print('Flash SDPA available:', torch.backends.cuda.is_flash_attention_available())"

WORKDIR /workspace