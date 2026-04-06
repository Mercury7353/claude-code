#!/bin/bash
#SBATCH -J evopool_e25_enh
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 10:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e25_enhanced_codream_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e25_enhanced_codream_%j.err

# E25: EvoPool with Enhanced CoDream (tests whether CoDream can help independent tasks)
#
# Three enhancements over E17 (standard CoDream):
#   1. Disagreement trigger: CoDream also runs when agents DISAGREE (max-min score >= 0.5)
#      even if team avg > 0.6. Captures [0,1,1] teams where 1 agent failed.
#      For MATH (83% accuracy): expected ~42% disagreement → 5x more CoDream triggers
#
#   2. Success extraction: when a successful agent is in the CoDream session,
#      they extract "what worked" (positive strategy) rather than "what failed"
#      This is the richest signal for independent tasks
#
#   3. domain_general insights (L2.5): new tier between L2 (subdomain) and L3 (general)
#      "For any math: verify answer by substitution" → injected for ALL MATH tasks
#      This is higher-bandwidth than subdomain but doesn't pollute other domains
#
# Prediction:
#   - MATH/GSM8K: significant improvement over E17 (+0.03-0.05 per domain)
#   - HQA/DROP: similar or slightly better (disagreement trigger helps here too)
#   - Overall: should exceed E17's 0.874
echo "=== E25: Enhanced CoDream (disagreement trigger + success extraction + domain_general) ==="
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
mkdir -p results/e25

python -u run_experiment.py \
    --condition "evopool_enhanced_codream" \
    --benchmark aflow_stream \
    --n_per_domain 100 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/e25/

echo "=== Done: Enhanced CoDream | $(date) ==="
