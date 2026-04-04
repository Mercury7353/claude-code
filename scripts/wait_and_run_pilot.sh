#!/bin/bash
# Wait for the vLLM server to come up, then launch the pilot experiment.
# Run this in the background: bash scripts/wait_and_run_pilot.sh &

SERVER_FILE="/nfs/hpc/share/zhanyaol/claude-code/vllm_server.json"
MAX_WAIT=1800   # seconds
INTERVAL=15

echo "Waiting for vLLM server (checking $SERVER_FILE every ${INTERVAL}s, max ${MAX_WAIT}s)..."

elapsed=0
while [ $elapsed -lt $MAX_WAIT ]; do
    if [ -f "$SERVER_FILE" ]; then
        URL=$(python3 -c "import json; d=json.load(open('$SERVER_FILE')); print(d['url'])" 2>/dev/null)
        if [ -n "$URL" ]; then
            # Test the server is responsive
            if curl -sf "$URL/health" > /dev/null 2>&1 || curl -sf "$URL/v1/models" > /dev/null 2>&1; then
                echo "vLLM server is UP at $URL"
                break
            fi
        fi
    fi
    echo "  Still waiting... (${elapsed}s elapsed)"
    sleep $INTERVAL
    elapsed=$((elapsed + INTERVAL))
done

if [ $elapsed -ge $MAX_WAIT ]; then
    echo "Timeout waiting for vLLM server. Exiting."
    exit 1
fi

echo ""
echo "=== Launching Pilot Experiment ==="
cd /nfs/hpc/share/zhanyaol/claude-code
mkdir -p logs results/pilot

MODEL="qwen3-8b"
export EVOPOOL_LOCAL_LLM_URL="$URL"

CONDITIONS=("evopool_full" "evopool_no_codream" "dylan" "agentnet")

for COND in "${CONDITIONS[@]}"; do
    echo ""
    echo ">>> Running: $COND"
    python run_experiment.py \
        --condition "$COND" \
        --benchmark gsm8k_stream \
        --n_tasks 30 \
        --pool_size 10 \
        --team_size 3 \
        --backbone_llm "$MODEL" \
        --seed 42 \
        --output_dir results/pilot/ \
        2>&1 | tee logs/pilot_${COND}.log
    echo ">>> Done: $COND (exit $?)"
done

echo ""
echo "=== Pilot Complete. Results: ==="
python3 -c "
import json, glob, os
files = sorted(glob.glob('results/pilot/*.json'))
if not files:
    print('  No results yet.')
else:
    print(f'  {\"System\":<35} {\"Mean\":>6} {\"Final\":>6} {\"Slope\":>8}')
    print('  ' + '-'*58)
    for f in files:
        d = json.load(open(f))
        s = d['summary']
        name = s['system'][:35]
        print(f'  {name:<35} {s[\"mean_score\"]:>6.3f} {s[\"final_score\"]:>6.3f} {s[\"learning_slope\"]:>8.4f}')
"
