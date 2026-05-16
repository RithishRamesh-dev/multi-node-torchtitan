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
# B300 (sm_103) SDPA fix — force MATH backend only
#
# Root cause (confirmed by full codebase analysis):
#
# Problem 1 — cuDNN SDPA forward fails:
#   attention.py sdpa_backends=[CUDNN, FLASH, MATH] with set_priority=True.
#   cuDNN has no sm_103 kernels → hard crash, no fallthrough.
#
# Problem 2 — Flash Attention backward fails:
#   On B300 with PyTorch nightly, Flash Attention dispatches through
#   _scaled_dot_product_fused_attention_overrideable (listed in
#   activation_checkpoint.py line 40 as a save_op). This op returns tensors
#   with deferred/lazy storage allocation on new architectures. When autograd
#   traverses the backward graph, it encounters these unallocated tensors and
#   throws: "tensor has non-zero elements but data not allocated yet".
#   This happens at loss.backward() on step 1 — not during AC recompute.
#   It fails with BOTH selective and full AC modes.
#
# Fix: Force SDPBackend.MATH only.
#   MATH is pure PyTorch eager autograd — no custom CUDA kernels, no lazy
#   storage, no dispatch to _fused_attention_overrideable. backward works
#   correctly on any GPU architecture. Performance is lower (~3-4x vs Flash)
#   but this is the only end-to-end working path on B300 with nightly torch.
#
# Memory note: With AC disabled (mode=none in YAML), activation memory for
#   8B model at seq=8192 is ~12GB per GPU — fine on B300's 267GB.
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
                SDPBackend.MATH,  # B300 workaround: MATH is only backend with working backward on sm_103/nightly
            ]"""

assert old in content, "Pattern not found — attention.py structure may have changed"
content = content.replace(old, new)

with open(path, "w") as f:
    f.write(content)

ast.parse(content)
print("Patch OK")
print("Active backends:", [l.strip() for l in content.splitlines() if "SDPBackend." in l and "import" not in l and "sdpa_backends" not in l])
PYEOF

RUN python -c "import ast; ast.parse(open('/workspace/torchtitan/torchtitan/models/common/attention.py').read()); print('syntax OK')"
RUN grep -n "SDPBackend\|sdpa_backends" /workspace/torchtitan/torchtitan/models/common/attention.py
RUN python -c "from torch.distributed.checkpoint import HuggingFaceStorageWriter; print('imports OK')"
RUN python -c "import torch; print('PyTorch:', torch.__version__)"

WORKDIR /workspace