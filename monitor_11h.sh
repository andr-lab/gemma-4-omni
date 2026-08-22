#!/bin/bash
# monitor_11h.sh — attende il termine del training adapter v3 sulla 3090, poi esce (risveglia l'agente)
# Controlla anche che il builder THv5 non sia morto prematuramente.
ADAPTER_DONE=false
for i in $(seq 1 240); do
  # container di training attivo? (cerca train_adapter_v3_cv26 tra i processi host docker)
  RUNNING=$(ssh -o BatchMode=yes gpu3090 "ps aux | grep train_adapter_v3_cv26 | grep -v grep | wc -l" 2>/dev/null || echo "ERR")
  THV5=$(ssh -o BatchMode=yes gpu3090 "grep -c 'FATTO' /tmp/job_thv5_build.log 2>/dev/null; ps aux | grep build_turnhead_v5 | grep -v grep | wc -l" 2>/dev/null | tr '\n' ' ')
  echo "[$i] $(date +%H:%M) adapter_running=$RUNNING thv5_status=$THV5"
  if [ "$RUNNING" = "0" ]; then
    echo "ADAPTER_TRAINING_TERMINATO al ciclo $i"
    ssh -o BatchMode=yes gpu3090 "ls -la ~/gemma-4-omni/checkpoints/gemma4_duplex_sota_v3/ 2>/dev/null; tail -5 \$(ls -t /tmp/*.log 2>/dev/null | head -1)"
    exit 0
  fi
  sleep 300
done
echo "TIMEOUT monitor (20h): esco comunque per verifica manuale"
exit 0
