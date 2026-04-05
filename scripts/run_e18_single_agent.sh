#!/bin/bash
#SBATCH -J evopool_e18_single
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 4:00:00
#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e18_single_agent_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e18_single_agent_%j.err

# E18: Single-agent baseline (lower bound — no pool, no memory, no collaboration)
# Establishes the value floor for all EvoPool machinery
echo "=== E18: Single-Agent Baseline ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
echo "Git: $(git -C /nfs/hpc/share/zhanyaol/claude-code log --oneline -1)"

source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool
cd /nfs/hpc/share/zhanyaol/claude-code

URL6=""
URL7=""
# Try servers 6/7 first, then fall back to 8007/8008 if newer ones are available
for jf in "vllm_server_6.json" "vllm_server_8007.json"; do
    [ -f "$jf" ] && URL6=$(python3 -c "import json; d=json.load(open('$jf')); print(d['url'])" 2>/dev/null) && break
done
for jf in "vllm_server_7.json" "vllm_server_8008.json"; do
    [ -f "$jf" ] && URL7=$(python3 -c "import json; d=json.load(open('$jf')); print(d['url'])" 2>/dev/null) && break
done

# Verify selected servers are actually responding
check_server() {
    local url="$1"
    curl -s --max-time 5 "${url}/v1/models" | grep -q qwen 2>/dev/null
}
for jf in "vllm_server_8007.json" "vllm_server_6.json"; do
    [ -f "$jf" ] && candidate=$(python3 -c "import json; d=json.load(open('$jf')); print(d['url'])" 2>/dev/null)
    check_server "$candidate" && URL6="$candidate" && echo "Server A: $URL6 (from $jf)" && break
done
for jf in "vllm_server_8008.json" "vllm_server_7.json"; do
    [ -f "$jf" ] && candidate=$(python3 -c "import json; d=json.load(open('$jf')); print(d['url'])" 2>/dev/null)
    check_server "$candidate" && URL7="$candidate" && echo "Server B: $URL7 (from $jf)" && break
done

if [ -z "$URL6" ] || [ -z "$URL7" ]; then
    echo "ERROR: No responsive vLLM servers found"
    exit 1
fi
echo "Using servers: $URL6, $URL7"
export EVOPOOL_LOCAL_LLM_URLS="$URL6,$URL7"
export HF_DATASETS_OFFLINE=1
mkdir -p results/e18

python -u run_experiment.py \
    --condition "single_agent" \
    --benchmark aflow_stream \
    --n_per_domain 100 \
    --pool_size 1 \
    --team_size 1 \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/e18/

echo "=== Done: single_agent | $(date) ==="
