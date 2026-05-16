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
# Upgrade pip tooling (NOT torch)
# -------------------------
RUN pip install --upgrade pip setuptools wheel

# -------------------------
# Clone TorchTitan
# -------------------------
RUN git clone https://github.com/pytorch/torchtitan.git /workspace/torchtitan

WORKDIR /workspace/torchtitan

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
# Install PyTorch nightly cu128 for B300 (sm_100) support — must be LAST
# -------------------------
RUN pip install --pre torch \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    --force-reinstall

# -------------------------
# FIX: cuDNN SDPA "No valid execution plans" error
#
# Root cause: pip nightly torch wheel installs nvidia-cuda-nvrtc with hash-mangled
# library names (libnvrtc-builtins-HASH.so.12.8) that cuDNN frontend cannot find
# when it tries to dlopen libnvrtc-builtins.so.12.8 at runtime.
#
# Fix 1: Add /usr/local/cuda/lib64 (system CUDA, has canonical names) to LD_LIBRARY_PATH.
# Fix 2: Create canonical symlinks for hash-named files in pip nvidia packages.
#
# See: https://github.com/pytorch/pytorch/issues/167602
#      https://github.com/pytorch/pytorch/issues/99781
# -------------------------
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

RUN for f in $(find /opt/conda/lib/python3.11/site-packages/nvidia -name 'libnvrtc-builtins-*.so*' 2>/dev/null); do \
      dir=$(dirname "$f"); \
      canonical="$dir/$(basename $f | sed 's/libnvrtc-builtins-[0-9a-f]*/libnvrtc-builtins/')"; \
      [ ! -e "$canonical" ] && ln -s "$f" "$canonical" && echo "Linked: $canonical"; \
    done; ldconfig; true

# -------------------------
# Verify
# -------------------------
RUN python -c "from torch.distributed.checkpoint import HuggingFaceStorageWriter; print('HuggingFaceStorageWriter OK')"
RUN python -c "import torch; print('PyTorch:', torch.__version__)"
RUN ldconfig -p | grep libnvrtc-builtins || echo "WARNING: libnvrtc-builtins not in ldconfig — LD_LIBRARY_PATH will handle it at runtime"

WORKDIR /workspace