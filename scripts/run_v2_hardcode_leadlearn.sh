#!/bin/bash
#SBATCH -J evopool_v2_hardcode_leadlearn
#SBATCH -A hw-grp
#SBATCH -p hw-grp,share,preempt
#SBATCH --gres=gpu:0
#SBATCH -t 2-00:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/v2_hardcode_leadlearn_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/v2_hardcode_leadlearn_%j.err

echo "=== v2_hardcode_leadlearn (with leader learning) ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool
cd /nfs/hpc/share/zhanyaol/claude-code

URL6=""; URL7=""
check_server() { curl -s --max-time 5 "${1}/v1/models" | grep -q qwen 2>/dev/null; }
for jf in "vllm_server_8007.json" "vllm_server_6.json"; do
    [ -f "$jf" ] && candidate=$(python3 -c "import json; d=json.load(open('$jf')); print(d['url'])" 2>/dev/null)
    check_server "$candidate" && URL6="$candidate" && echo "Server A: $URL6" && break
done
for jf in "vllm_server_8008.json" "vllm_server_7.json"; do
    [ -f "$jf" ] && candidate=$(python3 -c "import json; d=json.load(open('$jf')); print(d['url'])" 2>/dev/null)
    check_server "$candidate" && URL7="$candidate" && echo "Server B: $URL7" && break
done
[ -z "$URL6" ] || [ -z "$URL7" ] && echo "ERROR: No responsive vLLM servers" && exit 1
export EVOPOOL_LOCAL_LLM_URLS="$URL6,$URL7"
export HF_DATASETS_OFFLINE=1
mkdir -p results/v2_hardcode_leadlearn

python -u run_experiment.py \
    --condition "evopool_full" \
    --benchmark hard_code_stream --n_per_domain all --domains mbpp,humaneval,code_contests \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/v2_hardcode_leadlearn/

echo "=== Done: v2_hardcode_leadlearn | $(date) ==="
