#!/bin/bash
#SBATCH -J evopool_e9_aflow
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 48:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e9_aflow_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e9_aflow_%j.err

# E9: AFlow with fixed async_llm (httpx + multi-server + 90s timeout)
SEED=42

echo "=== EvoPool E9 AFlow (httpx fix, all servers) ==="
echo "Job: ${SLURM_JOB_ID} | Node: $(hostname) | $(date)"
echo "Seed: $SEED"
echo "Git: $(git -C /nfs/hpc/share/zhanyaol/claude-code log --oneline -1)"

source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool
cd /nfs/hpc/share/zhanyaol/claude-code

# Wait for new servers (4/5) to be fully serving before starting AFlow.
# Servers 8001/8002 expire at ~midnight; we need long-lived servers.
wait_server_ready() {
    local url="$1"
    local max_wait=600  # 10 minutes
    local elapsed=0
    echo "  Waiting for $url to be ready..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -sf --max-time 5 "${url}/health" > /dev/null 2>&1 || \
           curl -sf --max-time 5 "${url}/v1/models" > /dev/null 2>&1; then
            echo "  $url is ready! (${elapsed}s)"
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
    echo "  WARNING: $url not ready after ${max_wait}s"
    return 1
}

echo "Waiting for new vLLM servers (4 and 5) to be ready..."
# Wait for server_4.json to exist (written at server start), then wait for health
for attempt in $(seq 1 40); do
    if [ -f "vllm_server_4.json" ]; then break; fi
    echo "  Waiting for vllm_server_4.json... attempt $attempt/40"
    sleep 15
done

URLS=""
if [ -f "vllm_server_4.json" ]; then
    URL4=$(python3 -c "import json; d=json.load(open('vllm_server_4.json')); print(d['url'])" 2>/dev/null)
    if [ -n "$URL4" ] && wait_server_ready "$URL4"; then
        URLS="$URL4"
    fi
fi

if [ -f "vllm_server_5.json" ]; then
    URL5=$(python3 -c "import json; d=json.load(open('vllm_server_5.json')); print(d['url'])" 2>/dev/null)
    if [ -n "$URL5" ] && wait_server_ready "$URL5"; then
        URLS="$URLS${URLS:+,}$URL5"
    fi
fi

# Fall back to old servers (8001/8002) if new ones not available
if [ -z "$URLS" ]; then
    echo "WARNING: New servers not ready, falling back to old servers"
    for f in vllm_server_2.json vllm_server_3.json; do
        if [ -f "$f" ]; then
            URL=$(python3 -c "import json; d=json.load(open('$f')); print(d['url'])" 2>/dev/null)
            [ -n "$URL" ] && URLS="${URLS}${URLS:+,}${URL}"
        fi
    done
fi

if [ -z "$URLS" ]; then
    echo "ERROR: No vLLM servers available."
    exit 1
fi

echo "Using servers: $URLS"
export EVOPOOL_LOCAL_LLM_URLS="$URLS"
export HF_DATASETS_OFFLINE=1
mkdir -p results/e9

python -u run_experiment.py \
    --condition "aflow" \
    --benchmark aflow_stream \
    --n_per_domain 100 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed "$SEED" \
    --output_dir results/e9/

echo "=== Done: aflow | $(date) ==="
