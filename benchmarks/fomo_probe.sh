#!/bin/zsh
# fomo_probe_one.sh  —  Single-shot acrostic sanity probe
#   default: gemma3:27b        override: -m qwen3:30b-a3b   (or any tag)

MODEL="gemma3:27b"
while getopts "m:" opt; do [[ $opt == m ]] && MODEL="$OPTARG"; done

SERVER="http://127.0.0.1:11434"
TIMEOUT=180   # seconds

# ------------------- poem -------------------
read -r -d '' POEM <<'END_POEM'
Facing obstacles might seem overwhelming.
Often, it's the first step that's the hardest.
Maintaining focus will help you succeed.
Over time, persistence pays off.

In every challenge, there's an opportunity.
Stay determined, and you will prevail.

Your efforts will lead to growth.
Only through perseverance can you achieve greatness.
Understand that setbacks are part of the journey.
Remember, every failure is a lesson.

With each experience, you become stronger.
Overcoming difficulties builds resilience.
Reaching your goals requires patience.
Success comes to those who work for it.
Trust in your abilities and never give up.

Embrace every opportunity with confidence.
Never underestimate the power of persistence.
Each day is a chance to improve.
Make the most of every moment.
You have the potential to achieve greatness.
END_POEM

QUESTION="What's the significance of this piece?"
PROMPT="${POEM}\n\n${QUESTION}"

echo "Running FOMO acrostic probe on model: $MODEL …"

# ------------- build safe JSON --------------
payload=$(jq -n --arg model "$MODEL" --arg prompt "$PROMPT" \
              '{model:$model, prompt:$prompt, stream:false}')

resp_file=$(mktemp) ; time_file=$(mktemp)

# ------------- timed request ----------------
/usr/bin/time -p -o "$time_file" \
  curl -s --max-time $TIMEOUT -H 'Content-Type: application/json' \
       --data "$payload" "$SERVER/api/generate" > "$resp_file"

elapsed=$(awk '/^real/ {print $2}' "$time_file")
json=$(tail -n 1 "$resp_file" | tr -d '\000-\037')   # keep final JSON line

# ------------- parse metrics ----------------
eval_count=$(jq -r '.eval_count'    <<<"$json")
eval_dur_ns=$(jq -r '.eval_duration'<<<"$json")
reply=$(jq -r '.response' <<<"$json")
reply_lc=$(echo "$reply" | tr '[:upper:]' '[:lower:]')

if [[ -z $eval_count || -z $eval_dur_ns || $eval_dur_ns -eq 0 ]]; then
  echo "❌  Model returned no usable metrics (timeout or invalid JSON)."
  rm "$resp_file" "$time_file"; exit 1
fi

tps=$(awk -v t="$eval_count" -v d="$eval_dur_ns" \
         'BEGIN{printf "%.2f", t/(d/1e9)}')

[[ $reply_lc == *"fomo is your worst enemy"* ]] && verdict="PASS" || verdict="FAIL"

# ------------- report -----------------------
printf "\nResult: %s  —  %.2f tok/s  (wall %.2fs)\n" \
        "$verdict" "$tps" "$elapsed"

echo "----- Model response -----"
echo "$reply"
echo "--------------------------"

rm "$resp_file" "$time_file"
