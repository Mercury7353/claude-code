#!/bin/bash
#SBATCH -J scale_sa_4b
#SBATCH -A hw-grp
#SBATCH -p hw-grp,share,preempt
#SBATCH --gres=gpu:0
#SBATCH -t 2-00:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -o logs/scale_sa_4b_%j.out
#SBATCH -e logs/scale_sa_4b_%j.err

echo "=== scale_sa_4b: single_agent with Qwen3-4B ==="
source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool
cd /nfs/hpc/share/zhanyaol/claude-code

# Use dedicated port for this model size
URLS=""
check_server() { curl -s --max-time 5 "${1}/v1/models" | grep -q qwen 2>/dev/null; }
candidate="http://$(scontrol show hostname $(squeue -u zhanyaol -n vllm_qwen3_4B -o %N -h 2>/dev/null) 2>/dev/null):8009"
# Fallback: try dgxh-1 directly
for try_url in "$candidate" "http://10.217.117.45:8009"; do
    if check_server "$try_url"; then
        URLS="$try_url"
        echo "Found server: $URLS"
        break
    fi
done
[ -z "$URLS" ] && echo "ERROR: No Qwen3-4B server on port 8009" && exit 1
export EVOPOOL_LOCAL_LLM_URLS="$URLS"
export HF_DATASETS_OFFLINE=1
mkdir -p results/scale_sa_4b

python -u run_experiment.py \
    --condition "single_agent" \
    --benchmark hard_math_stream --n_per_domain all \
    --domains "math_hard,aime_2022,aime_2023,aime_2024,aime_2025" \
    --pool_size 20 --team_size 3 \
    --backbone_llm "qwen3-4b" \
    --seed 42 \
    --output_dir results/scale_sa_4b/

echo "=== Done: scale_sa_4b | $(date) ==="
