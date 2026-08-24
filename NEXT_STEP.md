# NEXT STEP — dopo la ricerca letteratura (2026-08-24)

## Verdetto ricerca (docs/DESIGN_FULLDUPLEX_LETTERATURA.md)

**Il pattern architetturale è corretto** (cb0 semantico + residui acustici = pattern dominante 2025-26:
CaT-TTS dual-LM, Moshi semantic distillation, WER Trap → decoupling). **Il collo è triplice**:

1. **INFORMAZIONE (causa primaria)**: h_sub (output adapter) NON contiene l'info cb0
   (probe: lineare 3.7%, MLP 3.2%, bix ~6.6 loss vs random 7.62; transfer codec_head 0.08%).
   L'adapter è stato addestrato per i residui acustici, non per la semantica cb0.
2. **SCALA**: predittore cb0 = decoder piccolo su ~40h; Qwen3-TTS usa backbone 1.7B su 5M ore.
   Ma: accuracy token di parlato intrinsecamente bassa (2509.17021) — il top-1 assoluto non è il problema.
3. **IMPLEMENTATIVO (mancanza, non bug)**: scheduled sampling mai attivato (gate 25% mai raggiunto);
   manca l'hybrid training TF+free-running (2509.17021: WER 6.23→4.21) e l'input corruption (2607.29363).

## Piano d'azione (ordine di costo crescente)

- [ ] **P1. Input corruption** sul trainer AR: corrompere ε=0.1-0.2 dei token cb0 GT nel prefix
      (sostituzione con campioni dalla distribuzione o token casuali). ~2 righe. Evidenza: 2607.29363.
- [ ] **P2. Hybrid training** (TF + free-running con prompt protection + EOS-adaptive, SENZA gate).
      Evidenza: 2509.17021. Costo: 1.5× compute.
- [ ] **P3. Ancora testuale (inner monologue)**: condizionare il predittore cb0 anche sul testo T
      generato da Gemma (o embedding del testo). Fix strutturale "free ≈ ancorato".
      Evidenza: Moshi 2410.00037, Mini-Omni Reasoner 2508.15827.
- [ ] **P4. cb0-pressure sull'adapter** (ritrain): loss ausiliaria semantica (predizione cb0 o allineamento
      WavLM) durante il training dell'adapter. Evidenza: SemBridge 2608.07462 + nostri probe.
- [ ] **P5. Opzione B** se dopo P1-P4 top-1 < 15%: proiezione addestrata h_sub→spazio backbone +
      backbone 1.7B congelato + codec_head. Evidenza: Qwen2.5-Omni 2503.20215, DS2ST-LM 2601.16023.
- [ ] **P6. Turn-taking**: special tokens + policy duplex (backchannel/barge-in/yield/interrupt).
      Evidenza: DuplexCascade 2603.09180, survey 2606.19453.

## Vincoli operativi
- GPU 3090: llama-server utente (PID 422361) occupa ~13.5GB → per i training serve fermarlo (chiedere).
- Cache: `cache_ars_full.pt` (h_sub 2048) valida per P1/P2; per P3 serve estrarre testo/embedding Gemma.
- Commit: `docs/DESIGN_FULLDUPLEX_LETTERATURA.md` nuovo.
