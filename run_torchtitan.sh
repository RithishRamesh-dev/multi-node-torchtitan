#!/bin/bash
set -ex

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

export NCCL_IB_HCA=fabric0,fabric1,fabric2,fabric3,\
fabric4,fabric5,fabric6,fabric7,\
fabric8,fabric9,fabric10,fabric11,\
fabric12,fabric13,fabric14,fabric15

export NCCL_CROSS_NIC=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_NET_GDR_LEVEL=2

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=29500

NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=8

cd /workspace/torchtitan

cd /workspace/torchtitan

torchrun \
  --nnodes=4 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  -m torchtitan.train \
  --job.config_file ./configs/llama3_8b_32gpu.toml