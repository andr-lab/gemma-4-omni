# P4D — Frozen-Qwen Future Conditioning

## Vincolo

Qwen3-TTS Talker, voice cloning, native voice design, SubTalker e Code2Wav
restano frozen e byte-identical al checkpoint originale. Nessuna LoRA Qwen e
nessuna modifica ai pesi o al codice del Talker in questa fase.

GemmaListener e V9 continuano a consumare un frame PCM 16 kHz/80 ms ogni 80 ms.
L'eventuale latenza appartiene solo alla catena speaker/playout.

## STEP 0 — P4-T4 corretto

Il gate P4-T4 usa le soglie originali del piano:

- `causal_CE <= full_CE * 1.10`
- `causal_top1 >= full_top1 - 0.05`
- `causal_top5 >= full_top5 - 0.03`

Il rerun reale `P4/run8` al commit `269f4cf` è **FAIL**: CE ratio `1.3047` e
top-5 drop `6.953 pp`. Il risultato precedente PASS era un bug del verifier,
che controllava solo la presenza di una differenza causale.

## Esperimenti frozen

`tools/kernel_gates/p4d_checks.py` confronta FULL con prefisso causale più un
lookahead reale di `L` frame, equivalente a `L * 80 ms` di speaker playout.
Valuta inoltre anchor causali deterministiche, senza parametri addestrati:
zero e primo frame raw fissato.

Risultati RTX 3090, 100 sequenze/6.400 frame per variante:

| anchor | lookahead | delay | CE ratio | top-1 drop | top-5 drop | gate |
|---|---:|---:|---:|---:|---:|---|
| zero | 8 | 640 ms | 1.1127 | 0.953 pp | 2.094 pp | FAIL |
| zero | 12 | 960 ms | 1.0957 | 0.625 pp | 1.563 pp | PASS |
| first raw | 4 | 320 ms | 1.1142 | 0.781 pp | 2.203 pp | FAIL |
| first raw | 7 | 560 ms | 1.0992 | 0.500 pp | 1.813 pp | PASS |
| first raw | 8 | 640 ms | 1.0950 | 0.359 pp | 1.844 pp | PASS |

Il minimo candidato osservato è `first_raw_frame + 7 frame` (560 ms),
senza modificare Qwen. `first_raw_frame` è causale e fisso dall'inizio della
sequenza; `oracle_text` è usato solo come baseline diagnostico e non è valido
come anchor runtime.

## Playout

`duplex_kernel/playout.py` implementa `SpeakerPlayoutBuffer`: riceve PCM
24 kHz in frame esatti da 80 ms, ritarda/releasea per frame ID, rifiuta gap,
duplicati, shape errate e PCM non finito. È speaker-only e non tocca il clock
del listener. Il delay runtime è opzionale (`0` di default) tramite
`GEMMA4_OMNI_FROZEN_QWEN_PLAYOUT_DELAY_FRAMES` o `playout_delay_frames` nella
sessione realtime.

Questa infrastruttura non viene dichiarata soluzione completa finché il
Talker runtime non consuma una sequenza V9 frame-stateful con il prefisso
`<= t+L`; il gate P4D dimostra prima la sufficienza del conditioning frozen,
non introduce una scorciatoia che ritardi Gemma.

## Next step

Integrare il lookahead nel percorso speaker stateful mantenendo il Talker
immutato; valutare un `FuturePrefixDrafter` solo se il delay minimo accettabile
non è sufficiente. Nessun QLoRA prima della decisione P4D.
