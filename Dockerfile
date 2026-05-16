FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

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

# -----------------------------------------------------------------------
# DO NOT install PyTorch nightly. Use stable 2.7.0 from the base image.
#
# Root cause of all previous failures:
# We were overriding the base image's stable torch==2.7.0+cu128 with
# a nightly build. PyTorch nightly has a regression in the backward
# pass of Flash Attention on B300 (sm_103) via
# _scaled_dot_product_fused_attention_overrideable, producing:
#   "tensor has non-zero elements but data not allocated yet"
#
# PyTorch 2.7.0 STABLE explicitly added Blackwell SDPA support (PR #145602:
# "Add Blackwell support to SDPA" — sm_100 and sm_120 archs). The stable
# release Flash Attention backward works correctly on B300.
#
# HuggingFaceStorageWriter is NOT needed: it's only used when
# checkpoint.last_save_in_hf=True, which is False (the default) in the
# llama3_8b config. Nightly was never required.
# -----------------------------------------------------------------------

# Patch attention.py: remove cuDNN (no sm_103 kernels in cuDNN)
# Keep FLASH_ATTENTION + MATH fallback — both work in stable 2.7.0
RUN python - << 'PYEOF'
import ast

path = "/workspace/torchtitan/torchtitan/models/common/attention.py"
with open(path) as f:
    content = f.read()

old = """            self.sdpa_backends = [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.MATH,
            ]"""

new = """            self.sdpa_backends = [
                SDPBackend.FLASH_ATTENTION,  # stable 2.7.0 has working FA backward on B300
                SDPBackend.MATH,             # fallback (not needed at seq=8192 but safe to list)
            ]"""

assert old in content, "Pattern not found"
content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
ast.parse(content)
print("Patch OK")
PYEOF

RUN python -c "import ast; ast.parse(open('/workspace/torchtitan/torchtitan/models/common/attention.py').read()); print('syntax OK')"
RUN grep -n "SDPBackend\|sdpa_backends" /workspace/torchtitan/torchtitan/models/common/attention.py
RUN python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"
RUN python -c "from torch.distributed.checkpoint import HuggingFaceStorageWriter; print('HuggingFaceStorageWriter OK')"

WORKDIR /workspace