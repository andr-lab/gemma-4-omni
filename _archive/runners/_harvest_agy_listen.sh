#!/bin/bash
# Harvester: per ogni file ritenta finché ottiene TRASCRIZIONE (sfrutta i successi
# sporadici dell'API audio agy). Salva output per file. Max 30 tentativi/file.
cd /c/Users/andre/Documents/gemma-4-omni
mkdir -p _agy_out_t11h_listen
declare -A TXT
TXT[generated_001]="Il perizoma è dorato all'esterno e dipinto di rosso all'interno."
TXT[generated_003]="Questa ipotesi venne scartata."
TXT[generated_005]="Venne sepolto nella basilica di San Vito."
TXT[oracle_001]="Il perizoma è dorato all'esterno e dipinto di rosso all'interno."
TXT[oracle_003]="Questa ipotesi venne scartata."

for F in generated_001 generated_003 generated_005 oracle_001 oracle_003; do
  DONE_FILE="_agy_out_t11h_listen/${F}.txt"
  if grep -q "TRASCRIZIONE" "$DONE_FILE" 2>/dev/null; then echo "[skip] $F già fatto"; continue; fi
  OK=0
  for i in $(seq 1 30); do
    T="${TXT[$F]}"
    OUT=$(timeout 280 agy -p "Ascolta questo audio: @C:/Users/andre/Documents/gemma-4-omni/_wav_v3/${F}.wav

Testo atteso: \"${T}\"

Rispondi ESATTAMENTE in questo formato:
TRASCRIZIONE: <quello che senti letteralmente>
MATCH: <percentuale parole corrispondenti>
QUALITA: <naturale / artefatti lievi / robotica / rumore>
VERDETTO: INTELLIGIBILE / PARZIALE / INCOMPRENSIBILE" \
      --model gemini-3.7-flash-high --print-timeout 4m --dangerously-skip-permissions 2>&1 | tail -8)
    if echo "$OUT" | grep -q "TRASCRIZIONE"; then
      echo "=== $F (tentativo $i) ===" 
      echo "$OUT" > "$DONE_FILE"
      echo "$OUT"
      OK=1; break
    fi
    echo "[$F try$i] fail: $(echo "$OUT" | head -1)"
    sleep $((RANDOM % 60 + 30))
  done
  [ $OK -eq 0 ] && echo "[GIVE UP] $F dopo 30 tentativi"
done
echo "HARVEST_COMPLETATO"
