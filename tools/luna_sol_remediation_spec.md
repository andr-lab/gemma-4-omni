# Specifica Operativa di Bonifica per GPT-6 Luna Max (Sol Blueprint)

## Obiettivo
Implementare le misure correttive e di protezione identificate dalla diagnosi architetturale di **GPT 6.1 Sol** per eliminare lo spike di latenza isolato (1430ms) durante il Turn 1 (long response playout continuo >= 40s) e garantire il superamento deterministico dei 3 run consecutivi della suite di accettazione (`tools/run_acceptance_suite.py --runs 3`).

## Vincoli Assoluti (Contratto Sol & AGENTS.md)
1. **Conservare il significato di `input_audio_buffer.consumed`**: MAI anticipare o falsificare l'ACK prima del completamento effettivo del frame.
2. **Nessun timeout cosmetico**: Non aggiungere timeout su Future o fallback a decisioni precedenti che altererebbero la provenienza temporale e la semantica del barge-in.
3. **Nessun `gc.collect()` in thread background durante il servizio attivo**: Il GC di CPython arresta il mondo (GIL) per tutti i thread.
4. **Preservare l'integrità dei 177 unit test** (`test_gemma_native_control6_policy.py`, `test_shared_listener_tick_dispatcher.py`, `test_real_dialogue_simulator.py`, `test_client_consumed_pcm_ack.py`).

---

## 1. Gestione Centralizzata del GC (CPython Cyclic GC) in `duplex_kernel/server.py`
In Python 3.11, il GC di Generazione 2 (`gc.collect(2)`) analizza l'intero heap di oggetti tracciabili. Durante sessioni lunghe con migliaia di frame audio e oggetti temporanei, una raccolta automatica Gen 2 può congelare l'interprete (GIL) per oltre 1.4s.
### Implementazione:
1. **GC Pause Tracker**:
   - Installare una singola callback `gc.callbacks.append(gc_callback)` a livello di processo in `duplex_kernel/server.py`.
   - Struttura `GcPauseSample`:
     - `sequence: int`
     - `generation: int`
     - `start_ns: int`
     - `end_ns: int`
     - `collected: int`
     - `uncollectable: int`
   - Salvare gli ultimi 256 campioni in una deque circolare thread-safe (`gc_pause_ring`).
   - La callback deve essere O(1), senza lock pesanti, I/O, logging o chiamate CUDA.
2. **Session-Aware GC Control**:
   - All'avvio di una sessione duplex realtime (o quando una sessione diventa attiva): se `gc.isenabled()`, invocare `gc.disable()`.
   - Tenere traccia del conteggio delle sessioni duplex attive (`active_duplex_sessions`).
   - Quando non ci sono più sessioni duplex attive (o alla chiusura/drain di una sessione): eseguire un `gc.collect()` controllato in un momento sicuro (fuori dal loop audio critico) e riabilitare il GC se non ci sono altre sessioni attive.

---

## 2. Rimozione Sincronizzazioni Host-Device Bloccanti in `NativeControl6LoRA.forward`
In `training/duplex_qlora/native_control6.py`:
- Alle righe 170 e 186:
  ```python
  if not torch.isfinite(normalized_hidden).all():
      raise ValueError("normalized_hidden contains non-finite values")
  ...
  if not torch.isfinite(phase_t).all():
      raise ValueError("phase_t contains non-finite values")
  ```
- **Problema**: Su tensori CUDA, `if not torch.isfinite(...).all():` forza un `cudaStreamSynchronize()` sincrono immediato sul thread host per valutare il valore booleano! Se la GPU è impegnata nella generazione audio/text dell'assistente, questo blocca il thread CPU del listener in attesa che la GPU finisca tutti i kernel pendenti.
- **Risoluzione**:
  - Quando `normalized_hidden` e `phase_t` sono su device CUDA, NON forzare una sincronizzazione host intermedia sincrona con `if not bool(...)`.
  - In modalità CUDA/fused readback, il controllo di finitezza è già calcolato e verificato atomicamente in un unico pacchetto di readback host in `gemma_native_control6_policy.py:438-460` (`packed_scores` contenente `finite_flag` e `probabilities_finite_value`).
  - Assicurarsi che su CPU o quando non in fused mode i controlli rimangano conformi, ma su CUDA non ci siano sync booleani intermedi ridondanti.

---

## 3. Strumentazione e Correlazione Temporale
In `duplex_kernel/listener_microbatch_worker.py` e `duplex_kernel/gemma_native_control6_batch_policy.py`:
- Misurare `wall_start_ns`, `wall_end_ns` e `thread_cpu_start_ns`, `thread_cpu_end_ns` attorno a `mixed_source_policy`.
- Pubblicare nei diagnostici del tick se un evento GC (`gc_pause_ms`) si è sovrapposto al tick.
- Mantenere la metrica invariata `listener_batch_control6_policy_ms`.

---

## 4. Verifica
1. Eseguire tutti i test unitari pertinenti con pytest.
2. Sincronizzare i file modificati su Mondaini.
3. Riavviare il container o il servizio su Mondaini.
4. Eseguire la suite di accettazione (`tools/run_acceptance_suite.py --runs 3`).
