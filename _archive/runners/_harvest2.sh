#!/bin/bash
cd /c/Users/andre/Documents/gemma-4-omni
declare -A TXT
TXT[generated_001]="Il perizoma è dorato all'esterno e dipinto di rosso all'interno."
TXT[generated_005]="Venne sepolto nella basilica di San Vito."
for F in generated_001 generated_005; do
  for i in $(seq 1 25); do
    OUT=$(timeout 280 agy -p "Ascolta questo audio: @C:/Users/andre/Documents/gemma-4-omni/_wav_v3/${F}.wav

Testo atteso: \"${TXT[$F]}\"

Rispondi ESATTAMENTE:
TRASCRIZIONE: <quello che senti>
MATCH: <percentuale>
QUALITA: <naturale / artefatti / robotica / rumore>
VERDETTO: INTELLIGIBILE / PARZIALE / INCOMPRENSIBILE" \
      --model gemini-3.7-flash-high --print-timeout 4m --dangerously-skip-permissions 2>&1 | tail -6)
    if echo "$OUT" | grep -q "TRASCRIZIONE"; then
      echo "$OUT" > "_agy_out_t11h_listen/${F}.txt"
      echo "=== $F OK al tentativo $i ==="; echo "$OUT"; break
    fi
    echo "[$F try$i] fail"; sleep $((RANDOM % 90 + 30))
  done
done
echo "HARVEST2_FINITO"
