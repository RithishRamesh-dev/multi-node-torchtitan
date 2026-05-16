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
# B300 (sm_103) cuDNN SDPA fix
#
# Root cause (confirmed by reading TorchTitan source):
#   attention.py ScaledDotProductAttention.__init__ sets:
#     self.sdpa_backends = [CUDNN_ATTENTION, FLASH_ATTENTION, MATH]
#   with set_priority=True — PyTorch tries cuDNN FIRST.
#   cuDNN SDPA has no sm_103 kernels for B300, so it fails hard.
#   PyTorch does NOT fall through to Flash Attention on failure.
#
# Fix: Remove CUDNN_ATTENTION from the default list in attention.py.
#   This is a one-line sed on a known exact string.
#   After the patch: [FLASH_ATTENTION, MATH] — Flash runs on B300 via sm_100.
#
# Verified by reading:
#   torchtitan/models/common/attention.py lines 274-279
# -------------------------
RUN sed -i '/SDPBackend.CUDNN_ATTENTION,/d' \
    /workspace/torchtitan/torchtitan/models/common/attention.py

# Confirm CUDNN_ATTENTION is gone and FLASH_ATTENTION remains
RUN grep -n "SDPBackend\|sdpa_backends" \
    /workspace/torchtitan/torchtitan/models/common/attention.py

# Validate Python syntax
RUN python -c "
import ast
with open('/workspace/torchtitan/torchtitan/models/common/attention.py') as f:
    src = f.read()
ast.parse(src)
print('attention.py syntax OK')
"

RUN python -c "from torch.distributed.checkpoint import HuggingFaceStorageWriter; print('HuggingFaceStorageWriter OK')"
RUN python -c "import torch; print('PyTorch:', torch.__version__)"

WORKDIR /workspace