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
# DO NOT install PyTorch nightly — use stable 2.7.0 from the base image.
#
# PyTorch 2.7.0 stable landed PR #145602 "Add Blackwell support to SDPA"
# which properly compiles Flash Attention kernels for sm_100/sm_120.
# Flash Attention backward works correctly on B300 in stable 2.7.0.
#
# PyTorch nightly has a regression: Flash Attention on B300 dispatches
# through _scaled_dot_product_fused_attention_overrideable with lazy tensor
# storage — backward crashes with "data not allocated yet" on every AC mode.
#
# HuggingFaceStorageWriter/Reader ARE NOT in stable 2.7.0. They were added
# later. They are only used when checkpoint.last_save_in_hf=True (default:
# False, not set in llama3_8b config). We make their imports lazy/conditional
# so TorchTitan starts up fine on stable 2.7.0.
# -----------------------------------------------------------------------

# Patch 1: attention.py — remove cuDNN (no sm_103 kernels), keep FLASH + MATH
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
                SDPBackend.FLASH_ATTENTION,  # stable 2.7.0 has working FA on B300 (PR #145602)
                SDPBackend.MATH,
            ]"""

assert old in content, "Pattern not found in attention.py"
content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
ast.parse(content)
print("attention.py patch OK")
PYEOF

# Patch 2: checkpoint.py — make HuggingFace imports lazy (only needed when last_save_in_hf=True)
RUN python - << 'PYEOF'
import ast

path = "/workspace/torchtitan/torchtitan/components/checkpoint.py"
with open(path) as f:
    content = f.read()

old = """from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
)"""

new = """try:
    from torch.distributed.checkpoint import HuggingFaceStorageWriter
    from torch.distributed.checkpoint._consolidate_hf_safetensors import (
        consolidate_safetensors_files_on_every_rank,
    )
except ImportError:
    HuggingFaceStorageWriter = None  # type: ignore[assignment,misc]
    consolidate_safetensors_files_on_every_rank = None  # type: ignore[assignment]"""

assert old in content, "Pattern not found in checkpoint.py"
content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
ast.parse(content)
print("checkpoint.py patch OK")
PYEOF

# Patch 3: protocols/state_dict_adapter.py — make HuggingFaceStorageReader lazy
RUN python - << 'PYEOF'
import ast

path = "/workspace/torchtitan/torchtitan/protocols/state_dict_adapter.py"
with open(path) as f:
    content = f.read()

old = "from torch.distributed.checkpoint import HuggingFaceStorageReader"
new = """try:
    from torch.distributed.checkpoint import HuggingFaceStorageReader
except ImportError:
    HuggingFaceStorageReader = None  # type: ignore[assignment,misc]"""

assert old in content, "Pattern not found in state_dict_adapter.py"
content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
ast.parse(content)
print("state_dict_adapter.py patch OK")
PYEOF

RUN python -c "
import ast
for p in [
    '/workspace/torchtitan/torchtitan/models/common/attention.py',
    '/workspace/torchtitan/torchtitan/components/checkpoint.py',
    '/workspace/torchtitan/torchtitan/protocols/state_dict_adapter.py',
]:
    ast.parse(open(p).read())
    print(f'syntax OK: {p}')
"
RUN python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"
RUN python -c "from torchtitan.components.checkpoint import CheckpointManager; print('CheckpointManager import OK')"
RUN python -c "from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter; print('StateDictAdapter import OK')"

WORKDIR /workspace