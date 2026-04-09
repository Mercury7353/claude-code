#!/bin/bash
#SBATCH -A hw-grp
#SBATCH -p hw-grp,share,preempt
#SBATCH --gres=gpu:0
#SBATCH -t 2-00:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -J uniform_nocd
#SBATCH -o logs/uniform_nocd_%j.out
#SBATCH -e logs/uniform_nocd_%j.err
echo "=== EvoPool w/o CoDream uniform execution ==="
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
mkdir -p results/uniform_nocd_hardmath
python -u run_experiment.py \
    --condition "evopool_no_codream" --uniform_execution \
    --benchmark hard_math_stream --n_per_domain all \
    --domains "math_hard,aime_2022,aime_2023,aime_2024,aime_2025" \
    --pool_size 20 --team_size 3 --backbone_llm "qwen3-8b" --seed 42 \
    --output_dir results/uniform_nocd_hardmath/
echo "=== Done: uniform_nocd | $(date) ==="
