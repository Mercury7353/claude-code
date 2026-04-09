#!/bin/bash
# vLLM server watchdog — checks every 5 min, restarts if down
LOG="/nfs/hpc/share/zhanyaol/claude-code/logs/vllm_watchdog.log"
cd /nfs/hpc/share/zhanyaol/claude-code

check_server() {
    local url="$1"
    curl -s --max-time 10 "${url}/v1/models" 2>/dev/null | grep -q qwen
}

while true; do
    url7=$(python3 -c "import json; d=json.load(open('vllm_server_8007.json')); print(d['url'])" 2>/dev/null)
    url8=$(python3 -c "import json; d=json.load(open('vllm_server_8008.json')); print(d['url'])" 2>/dev/null)

    s7_ok=false; s8_ok=false
    check_server "$url7" && s7_ok=true
    check_server "$url8" && s8_ok=true

    if $s7_ok && $s8_ok; then
        echo "[$(date)] Both servers OK ($url7, $url8)" >> "$LOG"
    else
        echo "[$(date)] SERVER DOWN! s7=$s7_ok s8=$s8_ok — restarting" >> "$LOG"
        if ! $s7_ok; then
            sbatch scripts/serve_vllm_8007.sh >> "$LOG" 2>&1
            echo "[$(date)] Submitted vLLM 8007 restart" >> "$LOG"
        fi
        if ! $s8_ok; then
            sbatch scripts/serve_vllm_8008.sh >> "$LOG" 2>&1
            echo "[$(date)] Submitted vLLM 8008 restart" >> "$LOG"
        fi
        # Wait for servers to come up
        sleep 180
    fi
    sleep 300  # check every 5 min
done
