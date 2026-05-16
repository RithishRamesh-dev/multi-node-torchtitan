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
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.MATH,
            ]"""
assert old in content, "Pattern not found in attention.py"
content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
ast.parse(content)
assert "SDPBackend.CUDNN_ATTENTION" not in content, "cuDNN still present"
assert "SDPBackend.FLASH_ATTENTION" in content, "FLASH missing"
print("attention.py patch OK")
PYEOF

# Patch 2: checkpoint.py — wrap HuggingFace imports in try/except (not in stable 2.7.0)
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
assert "except ImportError:" in content, "except block missing"
assert "HuggingFaceStorageWriter = None" in content, "fallback missing"
print("checkpoint.py patch OK")
PYEOF

# Patch 3: state_dict_adapter.py — wrap HuggingFaceStorageReader in try/except
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
assert "except ImportError:" in content, "except block missing"
assert "HuggingFaceStorageReader = None" in content, "fallback missing"
print("state_dict_adapter.py patch OK")
PYEOF

# Final validation: syntax-check all patched files
RUN python - << 'PYEOF'
import ast
files = [
    "/workspace/torchtitan/torchtitan/models/common/attention.py",
    "/workspace/torchtitan/torchtitan/components/checkpoint.py",
    "/workspace/torchtitan/torchtitan/protocols/state_dict_adapter.py",
]
for p in files:
    ast.parse(open(p).read())
    print("syntax OK:", p)
print("All patches validated")
PYEOF

RUN python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)"

WORKDIR /workspace