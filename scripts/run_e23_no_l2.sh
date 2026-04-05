#!/bin/bash
#SBATCH -J evopool_e23_nol2
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 8:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e23_no_l2_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e23_no_l2_%j.err

# E23: EvoPool CoDream without L2 (no subdomain accumulation)
# Tests Claim A: whether L2 subdomain insights drive within-domain learning.
#
# Prediction:
#   - HQA Q1→Q4 trend REDUCED or eliminated (no L2 within-domain accumulation)
#   - HQA first 5 tasks remain high (L3 from GSM8K still active)
#   - DROP Q1→Q4 trend reduced
#   - Independent tasks (MATH/HE): no change (L2 wasn't helping them anyway)
#
# E17 (full) - E22 (no-L3) = L3 value
# E22 (no-L3) - E23 (no-L2, no-L3) = L2 value   [if E23 also disables L3 for clean isolation]
# Actually E23 as defined: disables L2 only, keeps L3 → compares L2 vs L3 contribution
echo "=== E23: EvoPool -L2 (no subdomain accumulation) ==="
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
    check_server "$candidate" && URL6="$candidate" && echo "Server A: $URL6" && break
done
for jf in "vllm_server_8008.json" "vllm_server_7.json"; do
    [ -f "$jf" ] && candidate=$(python3 -c "import json; d=json.load(open('$jf')); print(d['url'])" 2>/dev/null)
    check_server "$candidate" && URL7="$candidate" && echo "Server B: $URL7" && break
done

if [ -z "$URL6" ] || [ -z "$URL7" ]; then
    echo "ERROR: No responsive vLLM servers found"
    exit 1
fi
export EVOPOOL_LOCAL_LLM_URLS="$URL6,$URL7"
export HF_DATASETS_OFFLINE=1
mkdir -p results/e23

python -u run_experiment.py \
    --condition "evopool_no_l2" \
    --benchmark aflow_stream \
    --n_per_domain 100 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/e23/

echo "=== Done: EvoPool -L2 | $(date) ==="
