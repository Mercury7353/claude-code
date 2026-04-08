#!/bin/bash
#SBATCH -J arc3_evopool_gpt54
#SBATCH -A hw-grp
#SBATCH -p hw-grp,share,preempt
#SBATCH --gres=gpu:0
#SBATCH -t 2-00:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/arc3_evopool_gpt54_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/arc3_evopool_gpt54_%j.err

echo "=== ARC-AGI-3 EvoPool with GPT-5.4 ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"

source /nfs/hpc/share/zhanyaol/.env  # loads OPENAI_API_KEY
cd /nfs/hpc/share/zhanyaol/claude-code

python3 -u run_arc_agi3.py \
    --pool_size 5 \
    --team_size 3 \
    --backbone_llm "gpt-5.4" \
    --max_steps 50 \
    --warmup_plays 10 \
    --games "r11l,vc33,cd82,sb26,ft09,sc25,tn36,tr87,tu93,ls20,bp35,su15,ar25,sp80,lp85,g50t,s5i5,sk48,cn04,ka59,m0r0,re86,dc22,lf52,wa30" \
    --seed 42 \
    --output_dir results/arc3_gpt54/

echo "=== Done | $(date) ==="
