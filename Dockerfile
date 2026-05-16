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
# Root cause: TorchTitan's ScaledDotProductAttention sets:
#   sdpa_backends = [CUDNN_ATTENTION, FLASH_ATTENTION, MATH]
# with set_priority=True — cuDNN is tried first and fails hard on B300.
# PyTorch does NOT fall through to Flash Attention on failure.
#
# Fix: Remove CUDNN_ATTENTION from the default list.
# After patch: [FLASH_ATTENTION, MATH] — Flash runs on B300 via sm_100.
# -------------------------
RUN sed -i '/SDPBackend.CUDNN_ATTENTION,/d' /workspace/torchtitan/torchtitan/models/common/attention.py

RUN grep -n "SDPBackend\|sdpa_backends" /workspace/torchtitan/torchtitan/models/common/attention.py

RUN python -c "import ast; ast.parse(open('/workspace/torchtitan/torchtitan/models/common/attention.py').read()); print('attention.py syntax OK')"

RUN python -c "from torch.distributed.checkpoint import HuggingFaceStorageWriter; print('HuggingFaceStorageWriter OK')"
RUN python -c "import torch; print('PyTorch:', torch.__version__)"

WORKDIR /workspace