FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    openssh-server \
    libibverbs-dev \
    rdma-core \
    infiniband-diags \
    ibverbs-utils \
    iproute2 \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Install TorchTitan
WORKDIR /workspace

RUN git clone https://github.com/pytorch/torchtitan.git

WORKDIR /workspace/torchtitan

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN pip install \
    datasets \
    transformers \
    sentencepiece \
    tiktoken \
    wandb \
    tensorboard

ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV NCCL_DEBUG=WARN
ENV NCCL_IB_DISABLE=0
ENV NCCL_NET_GDR_LEVEL=2
ENV NCCL_SOCKET_IFNAME=eth0
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1

CMD ["/bin/bash"]