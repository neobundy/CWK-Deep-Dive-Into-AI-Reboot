#!/bin/zsh
# qwen_tps.sh  —  Ollama throughput benchmark (3-run average)

SERVER="http://127.0.0.1:11434"
PROMPT_BASE='Explain mixture-of-experts routing in three bullets.'
MAXTOK=512         # num_predict
TIMEOUT=180        # bump for the 235 B run
RUNS=3

models=(
  qwen3:30b-a3b     # MoE (3 B active)
  qwen3:32b         # dense
  qwen3:235b-a22b   # MoE (22 B active)
)

echo "Starting benchmarks on $(hostname)…"
for model in "${models[@]}"; do
  echo "=== Benchmarking ${model} ==="
  echo "  ↳ Warm-loading model…"
  curl -s "$SERVER/api/generate" \
       -d "{\"model\":\"$model\",\"prompt\":\"warm up\",\"stream\":false}" \
       >/dev/null

  total_tps=0
  for run in $(seq 1 $RUNS); do
    unique_prompt="${PROMPT_BASE}\n\n(Benchmark run ${run}/${RUNS} — $(date +%s))"
    echo "  ↳ Run ${run}/${RUNS} …"

    raw=$(curl --max-time $TIMEOUT -s "$SERVER/api/generate" \
          -d "{\"model\":\"$model\",\"prompt\":\"$unique_prompt\",\"stream\":false,\
               \"options\":{\"num_predict\":$MAXTOK}}")

    if [[ -z "$raw" ]]; then
      echo "❌  Error: no response (timeout ${TIMEOUT}s). Exiting."
      exit 1
    fi

    json=$(echo "$raw" | tr -d '\000-\037')  # strip control chars

    eval_count=$(echo "$json" | jq -r '.eval_count // empty')
    eval_dur_ns=$(echo "$json" | jq -r '.eval_duration // empty')

    if [[ -z "$eval_count" || -z "$eval_dur_ns" || "$eval_dur_ns" -eq 0 ]]; then
      echo "❌  Error: invalid metrics returned. Exiting."
      echo "$json" | jq . || true
      exit 1
    fi

    tps=$(awk "BEGIN {print $eval_count/($eval_dur_ns/1e9)}")
    secs=$(awk "BEGIN {print $eval_dur_ns/1e9}")

    total_tps=$(awk "BEGIN {print $total_tps + $tps}")

    snippet=$(echo "$json" | jq -r '.response' \
              | tr '\n' ' ' | awk '{for(i=1;i<=12&&i<=NF;i++)printf("%s%s",$i,(i==12||i==NF)?"":" "); if(NF>12)printf("…");}')

    printf "    ✅  run %d: %6.2f tok/s (%3d tok · %5.2fs) — \"%s\"\n" \
           "$run" "$tps" "$eval_count" "$secs" "$snippet"
  done

  avg=$(awk "BEGIN {print $total_tps/$RUNS}")
  printf "  ➜  Average over %d runs: %.2f tok/s\n\n" "$RUNS" "$avg"
done

echo "Benchmarks complete without error. 🎉"
