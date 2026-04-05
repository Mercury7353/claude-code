#!/bin/bash
#SBATCH -J vllm_qwen3_8b_8008
#SBATCH -A hw-grp
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/vllm8008_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/vllm8008_%j.err

MODEL_PATH="/nfs/hpc/share/zhanyaol/models/Qwen3-8B"
PORT=8008
SERVER_FILE="/nfs/hpc/share/zhanyaol/claude-code/vllm_server_8008.json"

echo "=== vLLM Server 7: Qwen3-8B port $PORT ==="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"

module load cuda/12.1 2>/dev/null || true
source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

NODE_IP=$(hostname -I | awk '{print $1}')
cat > "$SERVER_FILE" << EOF
{"host":"$NODE_IP","port":$PORT,"model":"$MODEL_PATH","url":"http://${NODE_IP}:${PORT}","job_id":"$SLURM_JOB_ID","node":"$SLURMD_NODENAME"}
EOF
echo "URL: http://${NODE_IP}:${PORT}"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port $PORT \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --served-model-name qwen3-8b \
  --trust-remote-code \
  --enable-prefix-caching \
  2>&1 | tee /nfs/hpc/share/zhanyaol/claude-code/logs/vllm_server7_${SLURM_JOB_ID}.log
