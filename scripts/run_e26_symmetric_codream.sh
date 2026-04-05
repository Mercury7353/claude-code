#!/bin/bash
#SBATCH -J evopool_e26_sym
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 10:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e26_symmetric_codream_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e26_symmetric_codream_%j.err

# E26: EvoPool with Symmetric CoDream (Claim D ablation)
#
# Tests: does asymmetric routing (standard) outperform symmetric broadcasting?
# Symmetric mode: ALL insights (L1+L2+L3) broadcast to ALL agents equally.
#
# Prediction (Claim D):
#   - Symmetric CoDream should HURT performance vs asymmetric (E17)
#   - Pool homogenization: all agents converge → diversity collapses
#   - Team selection provides less benefit (all agents similar)
#   - Expected: ~0.855-0.865 (lower than E17 0.874)
#   - HQA may be most affected (requires complementary specialists)
echo "=== E26: Symmetric CoDream (Claim D ablation: asymmetric vs symmetric) ==="
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
mkdir -p results/e26

python -u run_experiment.py \
    --condition "evopool_symmetric_codream" \
    --benchmark aflow_stream \
    --n_per_domain 100 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/e26/

echo "=== Done: Symmetric CoDream | $(date) ==="
