#!/bin/bash
#SBATCH -A hw-grp
#SBATCH -p hw-grp,share,preempt
#SBATCH --gres=gpu:0
#SBATCH -t 2-00:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -J ms_memcollab_s123
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/ms_memcollab_s123_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/ms_memcollab_s123_%j.err

echo "=== ms_memcollab_s123: memcollab seed=123 ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool
cd /nfs/hpc/share/zhanyaol/claude-code
URLS=""
check_server() { curl -s --max-time 5 "${1}/v1/models" | grep -q qwen 2>/dev/null; }
for port in 8007 8008 8009 8010; do
    jf="vllm_server_${port}.json"
    if [ -f "$jf" ]; then
        candidate=$(python3 -c "import json; d=json.load(open(\"$jf\")); print(d[\"url\"])" 2>/dev/null)
        if check_server "$candidate"; then
            [ -n "$URLS" ] && URLS="${URLS},"
            URLS="${URLS}${candidate}"
        fi
    fi
done
[ -z "$URLS" ] && echo "ERROR: No vLLM servers" && exit 1
export EVOPOOL_LOCAL_LLM_URLS="$URLS"
export HF_DATASETS_OFFLINE=1
mkdir -p results/ms_memcollab_s123

python -u run_experiment.py \
    --condition "memcollab" \
    --benchmark hard_math_stream \
    --n_per_domain all \
    --domains "math_hard,aime_2022,aime_2023,aime_2024,aime_2025" \
    --pool_size 20 --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed 123  \
    --output_dir results/ms_memcollab_s123/

echo "=== Done: ms_memcollab_s123 | $(date) ==="
