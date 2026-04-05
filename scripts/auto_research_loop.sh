#!/bin/bash
# Autonomous research loop — runs every 30min, tracks all experiments
LOG=/nfs/hpc/share/zhanyaol/claude-code/logs/auto_research.log
echo "[$(date)] Auto-research loop tick" >> $LOG

cd /nfs/hpc/share/zhanyaol/claude-code
source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool

# Show slurm job status
echo "[$(date)] Job queue:" >> $LOG
squeue -u zhanyaol --format="%.10i %.9P %.15j %.8u %.8T %.6M" 2>/dev/null >> $LOG

python3 - >> $LOG 2>&1 << 'EOF'
import json, glob, os

def summarize_dir(exp_dir, label):
    files = sorted(glob.glob(f"results/{exp_dir}/*.json"))
    if not files:
        print(f"=== {label}: no results yet ===")
        return
    results = {}
    for f in files:
        name = os.path.basename(f).replace("_aflow_stream_seed","_s").replace(".json","")
        d = json.load(open(f))
        s = d.get("all_scores", [])
        n = len(s)
        mean = sum(s)/n if n else 0
        dom = d.get("domain_scores", {})
        dom_clean = {}
        for k,v in dom.items():
            dom_clean[k] = (sum(v)/len(v) if isinstance(v, list) and v else v) if v else 0
        results[name] = {"mean": mean, "n": n, "domain": dom_clean}

    print(f"=== {label} ({len(files)}/{len(files)} complete) ===")
    for k, v in sorted(results.items(), key=lambda x: -x[1]["mean"]):
        dom_str = " ".join(f"{v['domain'].get(d,0):.2f}" for d in ["gsm8k","hotpotqa","mbpp","math","humaneval","drop"])
        print(f"  {k:<35} {v['mean']:.3f} ({v['n']:2}/60) | {dom_str}")

    evopool = [v["mean"] for k,v in results.items() if "evopool_full" in k and v["n"]==60]
    aflow = [v["mean"] for k,v in results.items() if k.startswith("aflow") and v["n"]==60]
    if evopool and aflow:
        ev = sum(evopool)/len(evopool)
        af = sum(aflow)/len(aflow)
        status = "✅ EvoPool WINS!" if ev > af else f"❌ EvoPool behind by {af-ev:.3f}"
        print(f"  EvoPool-full: {ev:.3f} | AFlow: {af:.3f} | {status}")

summarize_dir("e2", "E2 (fork+codream fix)")
summarize_dir("e3", "E3 (oracle-free embed)")
summarize_dir("e5", "E5 (all fixes + correct eval)")
EOF
