#!/bin/bash
# Una chiamata agy per file (limite: 1 attachment audio per chiamata)
cd /c/Users/andre/Documents/gemma-4-omni
mkdir -p _agy_out_t11h_listen
for F in generated_001 generated_003 generated_005 oracle_001 oracle_003; do
  echo "=== $F ==="
  case $F in
    generated_001) TXT="Il perizoma è dorato all'esterno e dipinto di rosso all'interno.";;
    generated_003) TXT="Questa ipotesi venne scartata.";;
    generated_005) TXT="Venne sepolto nella basilica di San Vito.";;
    *) TXT="(controllo GT - stesso testo del generated corrispondente)";;
  esac
  agy -p "Ascolta questo audio con attenzione: @C:/Users/andre/Documents/gemma-4-omni/_wav_v3/${F}.wav

Testo atteso (finestra di validazione CV26): \"$TXT\"

RISPONDI IN QUESTO FORMAT:
TRASCRIZIONE: <quello che senti letteralmente, parola per parola>
MATCH: <percentuale stimata parole corrispondenti al testo atteso>
QUALITA: <naturale / artefatti lievi / robotica / rumore>
VERDETTO: INTELLIGIBILE / PARZIALE / INCOMPRENSIBILE" \
      --model gemini-3.7-flash-high --print-timeout 8m --dangerously-skip-permissions \
      > "_agy_out_t11h_listen/${F}.txt" 2>&1 || echo "FAIL $F"
  tail -6 "_agy_out_t11h_listen/${F}.txt"
done
echo "TUTTI_COMPLETATI"
