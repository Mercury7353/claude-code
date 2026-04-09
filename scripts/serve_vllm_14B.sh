#!/bin/bash
#SBATCH -J vllm_qwen3_14B
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH --mem=80G
#SBATCH -c 8
#SBATCH -o logs/vllm_14B_%j.out
#SBATCH -e logs/vllm_14B_%j.err

echo "=== vLLM Qwen3-14B on port 8010 ==="
source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool

python -m vllm.entrypoints.openai.api_server \
    --model /nfs/hpc/share/zhanyaol/models/Qwen3-14B \
    --host 0.0.0.0 --port 8010 \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --enable-reasoning --reasoning-parser deepseek_r1
