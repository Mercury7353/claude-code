#!/bin/bash
# Multi-seed hard math experiments (Priority E1 + E2 from review)
# Submits 3 seeds x 5 methods = 15 jobs
# Plus 2 randomized ordering runs

cd /nfs/hpc/share/zhanyaol/claude-code

COMMON_HEADER='#!/bin/bash
#SBATCH -A hw-grp
#SBATCH -p hw-grp,share,preempt
#SBATCH --gres=gpu:0
#SBATCH -t 2-00:00:00
#SBATCH --mem=64G
#SBATCH -c 4'

SERVER_SETUP='source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
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
export HF_DATASETS_OFFLINE=1'

# Define methods and seeds
declare -A METHODS=(
    ["evopool"]="evopool_full"
    ["nocd"]="evopool_no_codream"
    ["leadlearn"]="evopool_full"  # with leader learning flag
    ["sc"]="self_consistency"
    ["memcollab"]="memcollab"
)

SEEDS=(42 123 7)

echo "=== Creating multi-seed experiment scripts ==="

for method in evopool nocd leadlearn sc memcollab; do
    cond="${METHODS[$method]}"
    for seed in "${SEEDS[@]}"; do
        # Skip seed 42 for methods we already have
        if [ "$seed" -eq 42 ] && [ "$method" != "leadlearn" ]; then
            echo "Skipping $method seed $seed (already have results)"
            continue
        fi
        
        name="ms_${method}_s${seed}"
        script="scripts/generated/${name}.sh"
        mkdir -p scripts/generated results/${name}
        
        extra_args=""
        if [ "$method" == "leadlearn" ]; then
            extra_args="--leader_learning"
        fi
        
        cat > "$script" << EOF
${COMMON_HEADER}
#SBATCH -J ${name}
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/${name}_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/${name}_%j.err

echo "=== ${name}: ${method} seed=${seed} ==="
echo "Job: \$SLURM_JOB_ID | Node: \$(hostname) | \$(date)"
${SERVER_SETUP}
mkdir -p results/${name}

python -u run_experiment.py \\
    --condition "${cond}" \\
    --benchmark hard_math_stream \\
    --n_per_domain all \\
    --domains "math_hard,aime_2022,aime_2023,aime_2024,aime_2025" \\
    --pool_size 20 --team_size 3 \\
    --backbone_llm "qwen3-8b" \\
    --seed ${seed} ${extra_args} \\
    --output_dir results/${name}/

echo "=== Done: ${name} | \$(date) ==="
EOF
        chmod +x "$script"
        echo "Created: $script"
    done
done

# Randomized ordering experiments (E2)
for method in evopool nocd; do
    cond="${METHODS[$method]}"
    name="shuffle_${method}_s42"
    script="scripts/generated/${name}.sh"
    mkdir -p results/${name}
    
    cat > "$script" << EOF
${COMMON_HEADER}
#SBATCH -J ${name}
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/${name}_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/${name}_%j.err

echo "=== ${name}: ${method} SHUFFLED ordering ==="
echo "Job: \$SLURM_JOB_ID | Node: \$(hostname) | \$(date)"
${SERVER_SETUP}
mkdir -p results/${name}

python -u run_experiment.py \\
    --condition "${cond}" \\
    --benchmark hard_math_stream \\
    --n_per_domain all \\
    --domains "math_hard,aime_2022,aime_2023,aime_2024,aime_2025" \\
    --pool_size 20 --team_size 3 \\
    --backbone_llm "qwen3-8b" \\
    --seed 42 --shuffle_all \\
    --output_dir results/${name}/

echo "=== Done: ${name} | \$(date) ==="
EOF
    chmod +x "$script"
    echo "Created: $script"
done

echo ""
echo "=== Summary ==="
echo "Multi-seed scripts: $(ls scripts/generated/ms_*.sh 2>/dev/null | wc -l)"
echo "Shuffle scripts: $(ls scripts/generated/shuffle_*.sh 2>/dev/null | wc -l)"
echo ""
echo "To submit all: for f in scripts/generated/*.sh; do sbatch \$f; done"
echo "To submit just new seeds: for f in scripts/generated/ms_*_s123.sh scripts/generated/ms_*_s7.sh; do sbatch \$f; done"
