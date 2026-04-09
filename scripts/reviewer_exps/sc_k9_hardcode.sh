#!/bin/bash
#SBATCH -A hw-grp
#SBATCH -p hw-grp,share,preempt
#SBATCH --gres=gpu:0
#SBATCH -t 2-00:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -J sc_k9_hcode
#SBATCH -o logs/sc_k9_hcode_%j.out
#SBATCH -e logs/sc_k9_hcode_%j.err
echo "=== SC k=9 on hard code ==="
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
mkdir -p results/sc_k9_hardcode
python -u run_experiment.py \
    --condition "self_consistency" --sc_k 9 \
    --benchmark hard_code_stream --n_per_domain all \
    --backbone_llm "qwen3-8b" --seed 42 \
    --output_dir results/sc_k9_hardcode/
echo "=== Done: sc_k9_hcode | $(date) ==="
