#!/bin/bash
# Riprova la capacità audio di agy ogni 15 min, max 32 tentativi (8h). Esce al primo successo.
cd /c/Users/andre/Documents/gemma-4-omni
for i in $(seq 1 32); do
  OUT=$(timeout 250 agy -p "Ascolta: @C:/Users/andre/Documents/gemma-4-omni/_wav_v3/oracle_003.wav e scrivi AUDIO_OK se senti voce." --model gemini-3.7-flash-high --print-timeout 4m --dangerously-skip-permissions 2>&1 | tail -2)
  echo "[$i] $(date +%H:%M) $OUT"
  if echo "$OUT" | grep -qiE "AUDIO_OK|voce"; then echo "AUDIO_CAPABILITA_RIPRISTINATA al tentativo $i"; exit 0; fi
  sleep 900
done
echo "TIMEOUT 8h: audio ancora non disponibile"
exit 1
