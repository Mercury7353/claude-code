#!/bin/bash
#SBATCH -J evopool_arc3_sa
#SBATCH -A hw-grp
#SBATCH -p hw-grp,share,preempt
#SBATCH --gres=gpu:0
#SBATCH -t 2-00:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/arc3_sa_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/arc3_sa_%j.err

echo "=== arc3_sa ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
cd /nfs/hpc/share/zhanyaol/claude-code

# Use system python 3.13 (has arc-agi + evopool)
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

python3 -u run_arc_agi3.py \
    --pool_size 5 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --max_steps 50 \
    --warmup_plays 30 \
    --seed 42 \
    --games "r11l,vc33,cd82,sb26,ft09,sc25,tn36,tr87,tu93,ls20,bp35,su15,ar25,sp80,lp85,g50t,s5i5,sk48,cn04,ka59,m0r0,re86,dc22,lf52,wa30" --output_dir results/arc3/ --single_agent

echo "=== Done: arc3_sa | $(date) ==="
