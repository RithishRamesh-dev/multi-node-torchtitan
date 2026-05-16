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

RUN pip install --pre torch \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    --force-reinstall

# -------------------------
# B300 (sm_103) SDPA fix
#
# Error history and root causes:
#  1. cuDNN SDPA (CUDNN_ATTENTION): no sm_103 kernels → hard crash. REMOVED.
#  2. MATH backend: O(seq²) memory → 8GB per attention call at seq=8192 → OOM. REMOVED.
#  3. Flash Attention + any AC mode: "tensor data not allocated" at backward.
#     This error occurs WITH the checkpoint_wrapper active (any AC mode).
#     Root cause: the ptd_checkpoint_wrapper interacts badly with Flash
#     Attention's backward on B300 in nightly torch.
#
# Fix: Flash Attention ONLY (no cuDNN, no MATH), combined with AC mode=none
#      in the YAML (set via --activation_checkpoint.mode none).
#      Without the checkpoint_wrapper, Flash Attention forward+backward
#      runs as pure eager autograd — no wrapper interference.
#      Flash Attention is O(n) memory so no OOM risk at seq=8192.
# -------------------------
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
                SDPBackend.FLASH_ATTENTION,  # B300: cuDNN has no sm_103 kernels; MATH is O(seq^2) OOM
            ]"""

assert old in content, "Pattern not found"
content = content.replace(old, new)

with open(path, "w") as f:
    f.write(content)

ast.parse(content)
print("Patch OK — FLASH_ATTENTION only")
PYEOF

RUN python -c "import ast; ast.parse(open('/workspace/torchtitan/torchtitan/models/common/attention.py').read()); print('syntax OK')"
RUN grep -n "SDPBackend\|sdpa_backends" /workspace/torchtitan/torchtitan/models/common/attention.py
RUN python -c "from torch.distributed.checkpoint import HuggingFaceStorageWriter; print('imports OK')"
RUN python -c "import torch; print('PyTorch:', torch.__version__)"

WORKDIR /workspace