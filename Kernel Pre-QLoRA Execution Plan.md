# Kernel Pre-QLoRA Execution Plan
## Preparazione del kernel Gemma 4 Omni per continuous full-duplex frame-by-frame

**Status:** implementation specification  
**Baseline repository:** `andr-lab/gemma-4-omni`  
**Baseline commit:** `5534b09a05ae1d5c55f832a33be15fb826fe6c80`  
**Scope:** tutto ciò che deve essere completato **prima** del QLoRA duplex di Gemma.

---

# 1. Obiettivo

Preparare il kernel affinché l'architettura runtime finale sia già sostanzialmente completa prima di addestrare Gemma.

Il QLoRA dovrà cambiare principalmente la **policy appresa da Gemma**, non richiedere una nuova riscrittura di scheduler, cache, V9, Talker o WebSocket.

Architettura target:

```text
                     ┌─────────────────────────────┐
MIC 16 kHz ─────────►│ Gemma continuous listener   │
                     │ persistent multimodal state │
SELF AUDIO REF ─────►│                             │
                     └──────────────┬──────────────┘
                                    │
                              Gemma hidden_t
                                    │
                     ┌──────────────┴───────────────┐
                     │                              │
                     ▼                              ▼
               Control Policy                 V9.step()
          LISTEN / SPEAK / YIELD                   │
          CONTINUE / BACKCHANNEL                   │
          REPLAN / HARD_STOP                       ▼
                     │                    dynamic conditioning
                     ▼                              │
                SpeakerGate                        ▼
                     │                       Qwen Talker.step()
                     │                              │
              LISTEN ─── no-op                      ▼
              SPEAK ───────────────────────► CB0 + CB1..15
                                                   │
                                                   ▼
                                               Code2Wav
                                                   │
                                                   ▼
                                               PCM 24 kHz
```

Prima del QLoRA:

```text
TurnDetectionHead
        ↓
   TurnHeadPolicy
        ↓
    SpeakerGate
```

Dopo il QLoRA:

```text
Gemma control tokens
        ↓
     GemmaPolicy
        ↓
    SpeakerGate
```

**Il kernel sottostante non deve cambiare.**

---

# 2. Vincoli architetturali non negoziabili

1. **TurnDetectionHead è temporanea.** Può restare funzionante fino al QLoRA, ma solo dietro un'interfaccia `TurnPolicy`. Nessun nuovo componente core può dipendere direttamente da `TurnHeadRuntime`.

2. **V9, Qwen Talker, SubTalker e Code2Wav restano frozen.** Nessun retraining deve essere introdotto prima che il causal-prefix ablation dimostri che è necessario.

3. **LISTEN non significa generare codici di silenzio con Qwen.** Significa non schedulare il Talker.

4. **Gemma deve continuare a ricevere audio mentre Qwen parla.**

5. **Clock logico duplex: 12.5 Hz = un tick ogni 80 ms.**

6. **Steering cadence e PCM/network cadence sono indipendenti.**
   - steering: 80 ms;
   - Code2Wav: può inizialmente lavorare su 2–4 frame;
   - WebSocket: può aggregare 160–320 ms.

7. **Ogni risposta, segmento e tick deve essere identificabile.** Un worker stale non può modificare una sessione nuova.

8. **La history contiene ciò che è stato realmente emesso**, non tutto ciò che Gemma aveva pianificato prima di un'interruzione.

9. **Nessuna fase GPU può essere chiusa con `FakeEngine`.** I test contract esistenti restano utili ma attualmente verificano solo protocollo/API. 

---

# 3. Regola sui test e sulle metriche

Una fase **NON è completata** perché:

- il codice "sembra funzionare";
- un agente dichiara una latenza;
- viene allegato uno screenshot;
- un log contiene un numero scritto manualmente;
- passano test con mock quando la fase riguarda GPU/model state.

Ogni fase deve avere un comando eseguibile che:

```text
1. esegue i test;
2. raccoglie raw events;
3. calcola automaticamente le metriche;
4. applica automaticamente le soglie;
5. restituisce exit code != 0 se una soglia fallisce;
6. produce un report JSON.
```

Formato minimo di ogni report:

```json
{
  "schema_version": 1,
  "phase": "P3",
  "git_sha": "...",
  "git_dirty": false,
  "timestamp_utc": "...",
  "hostname": "...",
  "gpu_name": "...",
  "gpu_uuid": "...",
  "cuda_version": "...",
  "driver_version": "...",
  "torch_version": "...",
  "transformers_version": "...",
  "python_version": "...",
  "seed": 1234,
  "input_manifest_sha256": "...",
  "raw_trace_sha256": "...",
  "metrics": {},
  "thresholds": {},
  "passed": true
}
```

Il summary deve essere **ricalcolato dal raw trace** da un verifier separato.

Required tooling:

```text
tools/kernel_gates/run_phase.py
tools/kernel_gates/verify_report.py
tools/kernel_gates/build_manifest.py
artifacts/kernel_gates/<git_sha>/<phase>/
```

`verify_report.py` deve fallire se:

- `git_dirty=true`;
- manca il raw trace;
- l'hash non coincide;
- mancano metriche obbligatorie;
- una soglia non è rispettata.

Per benchmark comparativi, baseline e candidate devono essere misurati:

```text
stesso host
stessa GPU
stessi weights
stesso input manifest
stessi sampling parameters
stesso seed
```

---

# PHASE P0 — Baseline verificabile e harness di misurazione

## Obiettivo

Creare il punto zero quantitativo prima di modificare ulteriormente il kernel.

Il kernel corrente ha già manual Thinker decode, streaming audio callback e primitive KV, ma non esiste ancora un gate GPU completo. 

## Task

Creare:

```text
tests/kernel/
tests/gpu/
tools/kernel_gates/
artifacts/kernel_gates/
```

Mantenere i contract test esistenti.

Aggiungere un benchmark GPU con almeno 20 input fissi e manifest versionato.

Registrare almeno:

```text
think_time_ms
v9_time_ms
talker_time_ms
ttfa_ms
speaker_rtf
generated_cb0_frames
generated_pcm_samples
cuda_peak_allocated_bytes
cuda_peak_reserved_bytes
nan_count
inf_count
exceptions
```

## Test obbligatori

### P0-T1 — CPU contract regression

Tutti i contract test correnti devono passare.

```text
pytest tests/test_duplex_kernel_contract.py
```

Gate:

```text
failed_tests = 0
```

### P0-T2 — Real-model GPU smoke

Almeno 20/20 richieste reali devono completarsi usando i weights effettivi.

Gate:

```text
success_rate = 1.000
nan_count = 0
inf_count = 0
empty_audio_responses = 0
uncaught_exceptions = 0
```

### P0-T3 — Reproducible manifest

Ripetere lo stesso benchmark due volte.

Gate:

```text
input_manifest_sha256_run1 == input_manifest_sha256_run2
git_sha_run1 == git_sha_run2
```

Le metriche prestazionali vengono registrate ma **non hanno ancora soglia di miglioramento**.

## Fase chiusa quando

Esiste un `P0_report.json` verificabile e `verify_report.py` restituisce `0`.

---

# PHASE P1 — Control plane: TurnPolicy, SpeakerGate e generation fencing

## Obiettivo

Rendere la TurnDetectionHead completamente sostituibile senza toccare il kernel.

Attualmente `server.py` conosce direttamente `TurnHeadRuntime` e decide direttamente `BARGE_IN`. 

Questo deve sparire dal core.

## API richiesta

```python
class ControlAction(Enum):
    LISTEN = ...
    SPEAK = ...
    CONTINUE = ...
    YIELD = ...
    BACKCHANNEL = ...
    REPLAN = ...
    HARD_STOP = ...


class TurnPolicy(Protocol):
    def step(self, state: PolicyInput) -> PolicyDecision:
        ...


class SpeakerGate:
    def apply(self, decision: PolicyDecision, session: DuplexSession):
        ...
```

Implementazioni iniziali:

```text
TurnHeadPolicy
ScriptedPolicy
NoOpPolicy
```

Implementazione futura:

```text
GemmaPolicy
```

## Generation fencing

Aggiungere:

```text
session_generation_id
response_id
speaker_segment_id
frame_id
```

Ogni callback/state write deve verificare che il proprio generation ID sia ancora attivo.

## Test obbligatori

### P1-T1 — Core independence

Ricerca automatica nel codice.

Gate:

```text
direct TurnHeadRuntime imports outside turn_policy/turnhead_adapter.py = 0
direct "BARGE_IN" branching inside server core = 0
```

### P1-T2 — State-machine property test

Generare almeno 10.000 sequenze casuali di `ControlAction`.

Gate:

```text
illegal_state_transitions = 0
uncaught_exceptions = 0
```

### P1-T3 — Cancellation race

Almeno 1.000 volte:

```text
start response A
cancel/yield A
start response B
allow worker A to finish late
```

Gate:

```text
stale_audio_events_emitted = 0
stale_text_events_emitted = 0
stale_history_writes = 0
stale_kv_writes = 0
stale_session_mutations = 0
```

### P1-T4 — History correctness

Interrompere una risposta dopo una quantità nota di output.

Gate:

```text
history_assistant_text == actually_emitted_text
planned_but_not_emitted_text_in_history = 0
```

## Fase chiusa quando

TurnHead può essere attivata/disattivata solo tramite config:

```text
turn_policy=turnhead
turn_policy=scripted
turn_policy=none
```

senza modificare server, engine o speaker loop.

---

# PHASE P2 — DuplexScheduler a 80 ms e audio ingress indipendente

## Obiettivo

Creare il clock su cui vivrà l'architettura finale.

L'audio in ingresso non deve più essere concettualmente legato a request/response.

## Task

Implementare:

```python
DuplexScheduler
DuplexTick
AudioIngressQueue
FrameClock
```

Un frame input standard:

```text
16 kHz × 80 ms = 1280 sample PCM
```

Ogni tick deve avere:

```text
frame_id
capture_start_ns
capture_end_ns
enqueue_ns
listener_start_ns
listener_end_ns
policy_ns
speaker_start_ns
speaker_end_ns
```

## Test obbligatori

### P2-T1 — Virtual clock correctness

Con clock virtuale:

```text
10 s → esattamente 125 tick
60 s → esattamente 750 tick
```

Gate:

```text
missing_ticks = 0
duplicate_ticks = 0
out_of_order_ticks = 0
```

### P2-T2 — Real clock stability

Run di almeno 60 secondi.

Gate:

```text
deadline_miss_rate <= 0.5%
tick_jitter_p99 <= 20 ms
```

### P2-T3 — Lossless input transport

Inviare PCM numerato frame-per-frame.

Gate:

```text
input_frames_received == input_frames_enqueued
input_frames_enqueued == input_frames_consumed
dropped_input_frames = 0
duplicate_input_frames = 0
out_of_order_input_frames = 0
```

### P2-T4 — Input durante output

Far durare artificialmente lo speaker almeno 5 secondi e contemporaneamente inviare mic frames.

Gate:

```text
input_frames_during_speaking > 0
dropped_frames_during_speaking = 0
```

L'attuale comportamento che considera l'audio arrivato durante la risposta come appartenente semplicemente al turno successivo deve essere eliminato dal percorso principale. 

---

# PHASE P3 — GemmaListener continuo e self-listening buffer

## Obiettivo

Gemma deve ricevere il flusso audio anche mentre il sistema parla, **prima ancora che il QLoRA le insegni cosa farne**.

Non è richiesta ancora una buona policy conversazionale.

È richiesta una topologia runtime corretta.

## Task

Creare:

```python
GemmaListener
GemmaListenerState
AssistantReferenceBuffer
```

API:

```python
listener.step(
    user_pcm_t,
    assistant_reference_t,
    state
) -> ListenerOutput
```

`ListenerOutput` deve includere almeno:

```text
hidden_t
audio_timestamp
listener_state_version
finite
```

Il percorso principale realtime non deve richiedere:

```text
PCM → transcribe_audio() → text → Thinker
```

L'ASR può restare come API legacy/debug, non come prerequisito del listener.

## Self listening

Memorizzare l'audio realmente emesso con timestamp.

Default:

```text
assistant_reference delay = 1 frame = 80 ms
```

Nessun frame futuro può essere visibile a Gemma.

## Test obbligatori

### P3-T1 — Exact audio accounting

Replay di almeno 5 minuti.

Gate:

```text
samples_received == samples_accounted_for
listener_input_frames == expected_frames
dropped_listener_frames = 0
```

### P3-T2 — Listener while speaking

Durante almeno 30 secondi di output speaker:

```text
listener_updates_while_speaking == input_frames_while_speaking
```

Gate:

```text
coverage = 100%
```

### P3-T3 — Hidden health

Almeno 10.000 listener updates.

Gate:

```text
nonfinite_hidden_count = 0
empty_hidden_count = 0
```

### P3-T4 — Listener step latency

Misurare con CUDA events dove applicabile.

Gate isolato:

```text
listener_step_gpu_ms_p95 <= 80 ms
```

Se questo gate fallisce, il kernel non è pronto per steering a 12.5 Hz.

### P3-T5 — Self-reference causality

Per ogni tick `t`:

```text
reference visible at t <= audio emitted at t-1
```

Gate:

```text
future_reference_leaks = 0
wrong_reference_frame = 0
```

---

# PHASE P4 — V9 realmente incrementale

## Obiettivo

Sostituire il percorso realtime:

```text
hidden sequence
→ mean
→ interpolate to 16
→ V9
```

attualmente ancora presente nel kernel 

con:

```python
h_pfx_t = v9.step(h_raw_t, h_tx_anchor, v9_state)
```

V9 deve mantenere:

```text
CausalConv last 2 states
KV layer 0
KV layer 1
KV layer 2
KV layer 3
RoPE position
```

Per il test di equivalenza iniziale, `h_tx_anchor` deve restare fisso lungo la sequenza. Non introdurre ancora un `h_tx_t` dinamico non visto nel training.

## Test obbligatori

### P4-T1 — Batch vs incremental equivalence

Per almeno 100 sequenze reali di validation:

```text
batch causal V9(h[0:T])
vs
step(h0), step(h1), ... step(hT)
```

Calcolare automaticamente:

```text
mean_abs_error
max_abs_error
normalized_rmse
cosine_similarity per frame
```

Gate BF16:

```text
mean_abs_error <= 0.01
cosine_similarity_min >= 0.999
```

Se necessario, il test matematico di causality deve essere ripetuto FP32.

### P4-T2 — No future leakage

Creare due sequenze:

```text
A = [same prefix <=t | future A]
B = [same prefix <=t | future B]
```

Gate:

```text
V9_output_A[<=t] == V9_output_B[<=t]
```

FP32:

```text
max_abs_difference <= 1e-5
```

### P4-T3 — V9 step performance

Almeno 10.000 step dopo warmup.

Gate:

```text
v9_step_gpu_ms_p95 <= 10 ms
```

### P4-T4 — Causal-prefix ablation Qwen

Questo è un **blocco obbligatorio prima del QLoRA**.

Usare validation data già esistenti e confrontare:

```text
FULL:
audio_t può attendere a tutto h_pfx[0:T]

CAUSAL:
audio_t può attendere solo h_pfx[0:t]
```

Minimo:

```text
>= 100 sequenze
>= 5.000 frame valutati
```

Metriche:

```text
CB0 cross entropy
CB0 top-1 accuracy
CB0 top-5 accuracy
per-frame KL divergence
metriche per quartile temporale
```

Gate per poter continuare con V9/Qwen frozen:

```text
causal_CE <= full_CE × 1.10

causal_top1 >= full_top1 - 5 percentage points

causal_top5 >= full_top5 - 3 percentage points
```

### Se P4-T4 fallisce

**STOP.**

Non iniziare QLoRA Gemma.

Aprire una decisione architetturale separata per causal adaptation V9/Qwen.

Non nascondere il fallimento cambiando soglie nel report.

---

# PHASE P5 — Persistent Qwen frame decoder e Dynamic Conditioning KV

## Obiettivo

Qwen deve diventare un decoder frame-stateful.

Non deve più essere richiamato come una sequenza di micro-TTS indipendenti.

L'attuale separazione tra parte ICL statica e `text + h_pfx + BOS` dinamica è una buona base. 

## Oggetto runtime

Creare:

```python
SpeakerSegment
TalkerState
DynamicConditioningKV
```

`SpeakerSegment`:

```text
segment_id
generation_id
text_plan
static_icl_kv
conditioning_slots
bos_state
acoustic_kv
frames_generated
emitted_frames
```

## Conditioning memory

Riservare inizialmente **64 V9 frame slots**, coerenti con le finestre temporali usate nel training V9.

Gli slot non ancora disponibili devono essere masked.

Layout concettuale:

```text
[STATIC ICL]
[PFX_00]
[PFX_01]
...
[PFX_63]
[BOS]
[AUDIO_00]
[AUDIO_01]
...
```

Al frame `t`, `AUDIO_t` può vedere:

```text
STATIC
PFX_00 ... PFX_t
BOS
AUDIO_00 ... AUDIO_t-1
```

ma **non** `PFX_t+1...63`.

## API

```python
talker.open_segment(text_plan, state)

talker.insert_conditioning(h_pfx_t, frame_id, state)

cb0_t, hidden_t = talker.step(state)

talker.close_segment(reason, state)
```

## Test obbligatori

### P5-T1 — Dynamic KV vs offline causal reference

Costruire una reference eager con attention mask causale esplicita.

Per almeno:

```text
100 sequenze
32+ frame per sequenza
```

confrontare logits CB0 frame-per-frame.

Gate:

```text
top1_token_match = 100%

mean_abs_logit_difference <= 0.02

logit_cosine_similarity_min >= 0.999
```

### P5-T2 — Acoustic KV persistence

Per un segmento di N frame:

```text
acoustic_kv_position incrementa una volta per frame
```

Gate:

```text
unexpected_cache_resets = 0
duplicate_cache_appends = 0
```

### P5-T3 — LISTEN costs zero Talker forwards

Forzare `LISTEN` per almeno 10 secondi.

Instrumentare il numero reale di forward Qwen.

Gate:

```text
qwen_forward_calls_during_listen = 0
subtalker_calls_during_listen = 0
code2wav_calls_during_listen = 0
```

Gemma deve comunque continuare:

```text
listener_updates_during_listen == expected_listener_ticks
```

### P5-T4 — Qwen frame deadline

Almeno 10.000 frame dopo warmup.

Misurare:

```text
Qwen main step
+
CB0 sampling
+
SubTalker residual generation
```

escludendo Code2Wav.

Gate:

```text
speaker_step_gpu_ms_p95 <= 70 ms
speaker_step_gpu_ms_p99 <= 80 ms
deadline_miss_rate <= 1%
```

### P5-T5 — Valid code range

Gate:

```text
invalid_cb0_codes = 0
invalid_residual_codes = 0
nonfinite_logits = 0
```

---

# PHASE P6 — Speaker semantics: silence, yield, replan e backchannel

## Obiettivo

Implementare le semantiche finali indipendentemente dalla policy che le sceglie.

Usare `ScriptedPolicy` per i gate. Non usare la qualità della TurnHead come criterio di correttezza del kernel.

## Semantica obbligatoria

### LISTEN

```text
Gemma listener: ON
V9: può aggiornare state se necessario
Qwen: OFF
PCM output: niente nuovo speech
```

### SPEAK

Apre un nuovo `SpeakerSegment`.

### CONTINUE

Avanza il segmento corrente di un frame.

### YIELD

Termina il segmento alla prossima boundary sicura.

### BACKCHANNEL

Apre un segmento breve mentre Gemma continua ad ascoltare.

### REPLAN

Chiude il segmento corrente e invalida il piano non emesso.

### HARD_STOP

Emergency stop immediato entro i limiti della pipeline audio.

## Test obbligatori

### P6-T1 — Silence gating

10 secondi di LISTEN.

Gate:

```text
generated_speech_frames = 0
speaker_model_forwards = 0
listener_frames_processed = expected
```

### P6-T2 — Yield latency

Produrre YIELD in punti pseudocasuali durante 1.000 segmenti.

Misurare:

```text
policy_decision_timestamp
last_generated_cb0_timestamp
last_emitted_pcm_timestamp
```

Gate:

```text
decision_to_last_cb0_p95 <= 80 ms

decision_to_last_pcm_p95 <= 320 ms
```

### P6-T3 — No ghost audio

Gate:

```text
audio_packets_generated_by_old_segment_after_allowed_stop_window = 0
```

### P6-T4 — Backchannel mechanics

Con `ScriptedPolicy`:

```text
LISTEN
BACKCHANNEL
LISTEN
```

per almeno 1.000 eventi.

Gate:

```text
failed_backchannel_segments = 0
listener_updates_lost_during_backchannel = 0
illegal_segment_state_transitions = 0
```

### P6-T5 — Replan

Interrompere segment A e aprire segment B.

Gate:

```text
A frames emitted after B activation = 0
A state writes after invalidation = 0
A unspoken text inserted into history = 0
```

---

# PHASE P7 — PCM streaming e packetization indipendente dal steering

## Obiettivo

Eliminare il coupling attuale:

```text
8 codec frame = 640 ms = audio callback
```

Il kernel corrente invia infatti Code2Wav per gruppi di 8 frame. 

Target:

```text
model steering      80 ms
Code2Wav            160–320 ms
network packet      160–320 ms
```

## Task

Aggiungere:

```text
AudioFrameTimeline
DecodeAccumulator
PacketSequenceNumber
segment-aware flush
crossfade/stop handling
```

## Test obbligatori

### P7-T1 — Exact PCM timeline

Gate:

```text
duplicate_packet_ids = 0
missing_packet_ids = 0
pcm_gap_samples = 0
pcm_overlap_samples = 0
out_of_order_packets = 0
```

### P7-T2 — Audio health

Almeno 20 minuti aggregati di output.

Gate:

```text
nan_samples = 0
inf_samples = 0
clipping_pct <= 0.1%
```

### P7-T3 — Control-to-output latency

Da `SPEAK`:

```text
control_to_first_cb0_p95 <= 80 ms
control_to_first_pcm_p95 <= 320 ms
```

### P7-T4 — Intelligibility regression

Usare un ASR fisso e versionato su un set testuale fisso.

Confrontare candidate vs P0 baseline sullo stesso hardware/config.

Gate:

```text
WER_candidate <= WER_baseline + 0.05 absolute
```

e:

```text
WER_candidate <= 0.25
```

Il transcript ASR completo per ogni sample deve essere presente nel raw trace.

---

# PHASE P8 — Integrated duplex soak

## Obiettivo

Provare che tutte le primitive funzionano contemporaneamente per sessioni lunghe.

La policy principale del test deve essere `ScriptedPolicy`, affinché il test misuri il kernel e non la qualità ancora insufficiente del turn model.

La TurnHead viene testata separatamente come compatibility shim.

## Scenario minimo

Durata:

```text
>= 30 minuti
```

Deve contenere ripetutamente:

```text
LISTEN
SPEAK
CONTINUE
YIELD
BACKCHANNEL
REPLAN
HARD_STOP
```

mentre il mic continua sempre a inviare frame.

## Test obbligatori

### P8-T1 — Zero-loss duplex

Gate:

```text
dropped_input_frames = 0
duplicate_input_frames = 0
out_of_order_input_frames = 0

stale_events = 0
illegal_state_transitions = 0

nan_count = 0
inf_count = 0
uncaught_exceptions = 0
```

### P8-T2 — Scheduler stability sotto carico reale

Gate:

```text
80ms_tick_deadline_miss_rate <= 1%

control_to_cb0_p95 <= 80 ms
```

### P8-T3 — Bounded memory

Dopo 5 minuti di warmup, misurare per almeno altri 25 minuti.

Gate:

```text
cuda_allocated_growth <= 256 MiB

session_KV_size never exceeds configured bound
```

Non usare `max_memory_allocated()` da solo per rilevare leak: registrare anche `memory_allocated()` periodicamente.

### P8-T4 — Speaking real-time capability

Per ogni speaker segment:

```text
speaker_RTF =
speaker_compute_wall_time /
emitted_audio_duration
```

Gate:

```text
speaker_RTF_p50 <= 1.0
speaker_RTF_p95 <= 1.10
```

Se il benchmark usa concurrency, il report deve indicare esplicitamente come viene attribuito il wall time.

### P8-T5 — TurnHead compatibility

Eseguire il kernel con:

```text
turn_policy=turnhead
```

Gate:

```text
TurnHead loads successfully
policy decisions reach SpeakerGate
kernel crash count = 0
direct TurnHead dependency in speaker/listener core = 0
```

**Non esiste un gate di accuracy TurnHead.**

La TurnHead è un bridge temporaneo, non il target finale.

---

# PHASE P9 — Freeze dell'interfaccia pre-QLoRA

## Obiettivo

Congelare l'interfaccia che il QLoRA dovrà imparare a pilotare.

Dopo questa fase non devono essere necessari cambiamenti strutturali al kernel per sostituire TurnHead con Gemma.

## Contratto GemmaPolicy

Preparare:

```python
class GemmaPolicy(TurnPolicy):
    def step(self, listener_output, generation_state) -> PolicyDecision:
        ...
```

Azioni finali:

```text
LISTEN
SPEAK
CONTINUE
YIELD
BACKCHANNEL
REPLAN
HARD_STOP
```

Il mapping esatto ai control token verrà collegato al checkpoint QLoRA.

## Runtime training trace

Il kernel deve poter esportare un trace a 80 ms:

```json
{
  "frame_id": 120,
  "user_audio_start": 9.600,
  "user_audio_end": 9.680,
  "assistant_audio_reference_frame": 119,
  "speaker_active": true,
  "speaker_segment_id": "seg_...",
  "control_action": "CONTINUE",
  "gemma_hidden_available": true,
  "v9_frame_index": 120,
  "qwen_frame_generated": true
}
```

Questo schema deve diventare il ponte fra runtime e dataset SmoothConv/QLoRA.

## Test obbligatori

### P9-T1 — Trace round-trip

Almeno 100.000 frame.

Gate:

```text
frames_written == frames_read
timestamp_mismatches = 0
frame_id_mismatches = 0
invalid_actions = 0
```

### P9-T2 — Policy swap

Eseguire lo stesso scripted scenario con:

```text
ScriptedPolicy
TurnHeadPolicy
GemmaPolicy stub
```

Gate:

```text
kernel code-path modifications required = 0
```

Il cambio deve essere solo configurazione/injection.

### P9-T3 — Core dependency audit

Gate:

```text
TurnHead references inside:
  scheduler = 0
  speaker = 0
  V9 = 0
  Talker = 0
  GemmaListener = 0
  WebSocket protocol core = 0
```

TurnHead può esistere soltanto nel proprio adapter.

---

# 4. Gate finale PRE_QLORA_READY

Il QLoRA Gemma **non deve iniziare** finché non esiste:

```text
artifacts/kernel_gates/<git_sha>/PRE_QLORA_READY.json
```

generato automaticamente da:

```bash
python tools/kernel_gates/verify_pre_qlora_ready.py \
    artifacts/kernel_gates/<git_sha>/
```

Il verifier deve:

```text
caricare P0 ... P9
verificare SHA
verificare raw traces
ricomputare metriche
applicare thresholds
verificare stesso git_sha
verificare git_dirty=false
```

Output ammesso:

```json
{
  "pre_qlora_ready": true,
  "git_sha": "...",
  "all_required_phases": [
    "P0", "P1", "P2", "P3", "P4",
    "P5", "P6", "P7", "P8", "P9"
  ],
  "failed_phases": []
}
```

Qualunque altro risultato significa:

```text
DO NOT START GEMMA QLoRA
```

---

# 5. Ordine di implementazione obbligatorio

```text
P0  Measurement baseline
 ↓
P1  TurnPolicy + SpeakerGate + generation fencing
 ↓
P2  80 ms DuplexScheduler + continuous RX
 ↓
P3  continuous GemmaListener + self-reference
 ↓
P4  incremental V9 + causal-prefix ablation
 ↓
P5  persistent Qwen + Dynamic Conditioning KV
 ↓
P6  LISTEN/YIELD/BACKCHANNEL/REPLAN semantics
 ↓
P7  PCM streaming decoupled from model clock
 ↓
P8  30-minute integrated duplex soak
 ↓
P9  freeze GemmaPolicy/training-trace contract
 ↓
PRE_QLORA_READY
 ↓
SmoothConv QLoRA Gemma
```

Non saltare P4-T4.

È il test che stabilisce se l'assunzione:

```text
Gemma trainable
V9 frozen
Qwen frozen
```

è realmente sostenibile.

---

# 6. Cose da NON fare durante questo piano

Non:

```text
retrain TurnDetectionHead
aggiungere un nuovo VAD decision model
aggiungere un nuovo acoustic interruption classifier
fare additive latent injection dentro gli audio embeddings Qwen
retrain Qwen preventivamente
retrain V9 preventivamente
usare ASR→testo come unico listener realtime
considerare WebSocket always-open equivalente a Gemma always-listening
considerare 640ms packetization equivalente a 80ms steering
salvare in history testo mai realmente pronunciato
chiudere phase gate usando solo FakeEngine
```

---

# 7. Definition of Done prima del QLoRA

Il kernel è pronto quando il seguente scenario funziona già, usando `ScriptedPolicy` o temporaneamente `TurnHeadPolicy`:

```text
utente parla
    ↓
Gemma riceve ogni frame

policy = SPEAK
    ↓
Qwen apre segmento

Gemma continua a ricevere il mic
mentre Qwen produce frame ogni 80 ms

policy = CONTINUE
    ↓
Qwen continua con acoustic KV persistente

policy = YIELD
    ↓
Qwen si ferma entro il budget previsto
Gemma non si è mai fermata

policy = LISTEN
    ↓
Qwen non esegue forward
Gemma continua a processare mic

policy = BACKCHANNEL
    ↓
breve SpeakerSegment
Gemma continua ad ascoltare

policy = REPLAN
    ↓
vecchio generation_id invalidato
nessun ghost audio/state
nuovo segmento può partire
```

A quel punto il QLoRA SmoothConv deve soltanto sostituire:

```text
policy temporanea
```

con:

```text
policy appresa negli hidden/control logits di Gemma
```

Il comportamento fisico del kernel deve essere già pronto.

---

# 8. Principio finale

**Il QLoRA non deve essere usato per compensare un kernel non-causale o non-duplex.**

Prima si dimostra meccanicamente:

```text
continuous listening
80 ms clock
causal V9
persistent Qwen
speaker gating
clean yield/replan
bounded KV
real-time compute
```

con test eseguibili e metriche generate automaticamente.

Solo dopo si addestra Gemma a decidere:

```text
quando ascoltare
quando parlare
quando continuare
quando cedere
quando fare backchannel
quando replannare
```

Questo mantiene separati i due problemi:

```text
KERNEL = capacità fisica full-duplex
QLoRA  = comportamento conversazionale full-duplex
```