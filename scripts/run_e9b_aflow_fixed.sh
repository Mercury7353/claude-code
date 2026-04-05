#!/bin/bash
#SBATCH -J evopool_e9b_aflow
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 6:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e9b_aflow_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e9b_aflow_%j.err

# E9b: AFlow baseline with corrected format instructions
# Fixes E9 bug: MATH instruction now requires \boxed{} format; GSM8K requires #### <number>
# Without these, evaluator couldn't parse 48/100 MATH and ~26/100 GSM8K answers → zero scores
# AFlow QA domains (HQA=0.832, DROP=0.820) were correct and unchanged.
echo "=== E9b: AFlow fixed format ==="
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
mkdir -p results/e9b

python -u run_experiment.py \
    --condition "aflow" \
    --benchmark aflow_stream \
    --n_per_domain 100 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/e9b/

echo "=== Done: AFlow fixed | $(date) ==="
