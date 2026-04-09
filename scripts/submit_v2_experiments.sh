#!/bin/bash
# Submit EvoPool v2 experiments (post-redesign)
cd /nfs/hpc/share/zhanyaol/claude-code

COMMON_SBATCH="-A hw-grp --gres=gpu:0 --mem=64G -c 4"
COMMON_ENV="source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh && conda activate tool && cd /nfs/hpc/share/zhanyaol/claude-code"
SERVER_SETUP='URL6=""; URL7=""; check_server() { curl -s --max-time 5 "${1}/v1/models" | grep -q qwen 2>/dev/null; }; for jf in "vllm_server_8007.json" "vllm_server_6.json"; do [ -f "$jf" ] && candidate=$(python3 -c "import json; d=json.load(open('\''$jf'\'')); print(d['\''url'\''])" 2>/dev/null); check_server "$candidate" && URL6="$candidate" && break; done; for jf in "vllm_server_8008.json" "vllm_server_7.json"; do [ -f "$jf" ] && candidate=$(python3 -c "import json; d=json.load(open('\''$jf'\'')); print(d['\''url'\''])" 2>/dev/null); check_server "$candidate" && URL7="$candidate" && break; done; [ -z "$URL6" ] || [ -z "$URL7" ] && echo "ERROR: No vLLM" && exit 1; export EVOPOOL_LOCAL_LLM_URLS="$URL6,$URL7"; export HF_DATASETS_OFFLINE=1'

submit_exp() {
    local name="$1" dir="$2" condition="$3" extra="$4"
    mkdir -p "results/$dir"
    sbatch -J "evopool_${name}" -p hw-grp,share,preempt -t 24:00:00 $COMMON_SBATCH \
        -o "logs/${name}_%j.out" -e "logs/${name}_%j.err" \
        --wrap "$COMMON_ENV; $SERVER_SETUP; echo '=== $name ==='; echo \"Job: \$SLURM_JOB_ID | Node: \$(hostname) | \$(date)\"; python -u run_experiment.py --condition $condition --benchmark aflow_stream --n_per_domain 10 --pool_size 20 --team_size 3 --backbone_llm qwen3-8b --seed 42 --output_dir results/$dir/ $extra; echo '=== Done: $name | \$(date) ==='"
}

echo "Submitting EvoPool v2 experiment battery..."

# E-new-4: Experience injection impact (FIRST — if this fails, everything is moot)
submit_exp "e_new4_exp_on" "e_new4" "evopool_full" ""
# Note: need an "experience OFF" condition — use noCoDream with a flag to skip experience injection
# For now, noCoDream still has experience buffer from self-play

# E-new-1: Agent Differentiation Validation (full system)
submit_exp "e_new1_full" "e_new1" "evopool_full" ""

# E-new-3a: CoDream Full 5-phase
submit_exp "e_new3a_codream" "e_new3a" "evopool_full" ""

# E-new-3b: No CoDream (ablation)
submit_exp "e_new3b_nocd" "e_new3b" "evopool_no_codream" ""

# E-new-3c: Single agent baseline
submit_exp "e_new3c_sa" "e_new3c" "single_agent" "--pool_size 1 --team_size 1"

# E-new-5: vs DyLAN baseline
submit_exp "e_new5_dylan" "e_new5_dylan" "dylan" ""

echo "All jobs submitted!"
