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
# B300 (sm_103) cuDNN SDPA workaround
#
# The NVIDIA B300 GPU (sm_103) has no compiled cuDNN SDPA kernels in any
# official PyTorch or NVIDIA container as of May 2026 — confirmed by NVIDIA
# engineers. cuDNN Frontend fails with "No valid execution plans built."
#
# Fix: Patch TorchTitan's train.py to disable cuDNN SDPA at startup and fall
# back to Flash Attention (which runs sm_100/B200 kernels correctly on B300).
# This patch is in the source code, so pip reinstalls cannot overwrite it.
# -------------------------
RUN sed -i 's/^import torch$/import torch\ntorch.backends.cuda.enable_cudnn_sdp(False)  # B300 sm_103 workaround: no cuDNN SDPA kernels for sm_103 yet/' \
    /workspace/torchtitan/torchtitan/train.py && \
    head -5 /workspace/torchtitan/torchtitan/train.py

# Verify the patch is in place
RUN grep -n "enable_cudnn_sdp" /workspace/torchtitan/torchtitan/train.py

# Verify imports
RUN python -c "from torch.distributed.checkpoint import HuggingFaceStorageWriter; print('HuggingFaceStorageWriter OK')"
RUN python -c "import torch; print('PyTorch:', torch.__version__)"

WORKDIR /workspace