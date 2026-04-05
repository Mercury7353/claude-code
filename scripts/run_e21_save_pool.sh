#!/bin/bash
#SBATCH -J evopool_e21_pool
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 12:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e21_save_pool_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e21_save_pool_%j.err

# E21: EvoPool-full replica of E17, but with:
#   1. --save_pool: saves final evolved pool state for E20 warm-start
#   2. verify stats logging: records codream_generated/verified per task for analysis
#
# This run is scientifically identical to E17 (same condition, seed, benchmark).
# Minor numerical differences due to LLM non-determinism are expected and acceptable.
# The pool state from E21 will be used as warm-start for E20.
echo "=== E21: EvoPool-full (pool state save + verify stats) ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
echo "Git: $(git -C /nfs/hpc/share/zhanyaol/claude-code log --oneline -1)"

source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool
cd /nfs/hpc/share/zhanyaol/claude-code

URL6=""
URL7=""
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
mkdir -p results/e21

python -u run_experiment.py \
    --condition "evopool_full" \
    --benchmark aflow_stream \
    --n_per_domain 100 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --save_pool results/e21/evopool_pool_state.json \
    --output_dir results/e21/

echo "=== Done: E21 (pool saved to results/e21/evopool_pool_state.json) | $(date) ==="
