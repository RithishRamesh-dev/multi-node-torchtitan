FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

RUN apt-get update && apt-get install -y \
    git wget curl build-essential ninja-build \
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

# Install PyTorch nightly — required for TorchTitan's DCP APIs
# (DefaultStager, StagingOptions, HuggingFaceStorageWriter, AsyncCheckpointerType
#  are all post-2.7.0 stable additions, present only in nightly)
RUN pip install --pre torch \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    --force-reinstall

# Install flash-attn package targeting sm_100 (B200), which also runs on B300 (sm_103)
# via CUDA forward compatibility. This gives flash-attn its own proper CUDA kernels
# with a correct backward pass, bypassing PyTorch nightly's broken
# _scaled_dot_product_fused_attention_overrideable dispatch on B300.
RUN pip install packaging ninja
RUN TORCH_CUDA_ARCH_LIST="10.0" pip install flash-attn --no-build-isolation

# Patch attention.py: remove cuDNN (no sm_103 kernels) and remove MATH (OOM at seq=8192)
# Keep only FLASH_ATTENTION — now backed by flash-attn's proper sm_100 kernels
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
            ]"""
assert old in content, "Pattern not found in attention.py"
content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
ast.parse(content)
assert "SDPBackend.CUDNN_ATTENTION" not in content
assert "SDPBackend.FLASH_ATTENTION" in content
print("attention.py patch OK")
PYEOF

# Validate
RUN python - << 'PYEOF'
import ast
ast.parse(open("/workspace/torchtitan/torchtitan/models/common/attention.py").read())
print("syntax OK: attention.py")
PYEOF

RUN python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)"
RUN python -c "import flash_attn; print('flash-attn:', flash_attn.__version__)"

WORKDIR /workspace