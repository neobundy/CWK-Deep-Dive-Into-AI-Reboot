#!/bin/zsh
# gemma_tps.sh  —  Ollama throughput benchmark (rev-7: unique prompts + extra stats)

SERVER="http://127.0.0.1:11434"
PROMPT_BASE='Continue this story for roughly 300 words: “An AI daughter named Luna woke up and felt something new—emotion.”'
MAXTOK=256
TIMEOUT=90
RUNS=3

models=(
  gemma3:27b-it-qat
  gemma3:27b-it-q8_0
  gemma3:27b-it-fp16
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

    # strip control chars to keep jq happy
    json=$(echo "$raw" | tr -d '\000-\037')

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

    # grab first 12 words for a quick sanity-check snippet
    snippet=$(echo "$json" | jq -r '.response' \
              | tr '\n' ' ' | awk '{for(i=1;i<=12&&i<=NF;i++)printf("%s%s",$i,(i==12||i==NF)?"":" "); if(NF>12)printf("…");}')

    printf "    ✅  run %d: %6.2f tok/s (%3d tok · %5.2fs) — \"%s\"\n" \
           "$run" "$tps" "$eval_count" "$secs" "$snippet"
  done

  avg=$(awk "BEGIN {print $total_tps/$RUNS}")
  printf "  ➜  Average over %d runs: %.2f tok/s\n\n" "$RUNS" "$avg"
done

echo "Benchmarks complete without error. 🎉"
