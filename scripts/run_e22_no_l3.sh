#!/bin/bash
#SBATCH -J evopool_e22_nol3
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 8:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e22_no_l3_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e22_no_l3_%j.err

# E22: EvoPool CoDream without L3 (no cross-domain broadcast to persona)
# Tests Claim B: whether L3 broadcast causes the cross-domain transfer observed in E17.
#
# Prediction:
#   - HQA first 5 tasks DROP from 0.933 toward ~0.650 (no L3 priming from GSM8K)
#   - HQA Q1→Q4 trend PRESERVED (L2 subdomain accumulation still works)
#   - DROP first 5 tasks DROP from 0.800 toward ~0.600
#   - Overall mean: between E15b (0.763) and E17 (0.874)
#
# Compare: E17 (full, 0.874) vs E22 (no-L3) → isolates L3 cross-domain transfer value
#          E22 (no-L3) vs E15b (no-CoDream, 0.763) → isolates L2 within-domain value
echo "=== E22: EvoPool -L3 (no cross-domain broadcast) ==="
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
mkdir -p results/e22

python -u run_experiment.py \
    --condition "evopool_no_l3" \
    --benchmark aflow_stream \
    --n_per_domain 100 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/e22/

echo "=== Done: EvoPool -L3 | $(date) ==="
