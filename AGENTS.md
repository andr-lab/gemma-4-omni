# AGENTS.md — Gemma 4 Full-Duplex Omni Pipeline

## Regole operative di base

- L'ambiente primario di calcolo GPU ed esecuzione è la macchina remota **Mondaini** (host `mondaini` / RTX 3090 24GB VRAM) via Docker isolato.
- Non scaricare mai modelli di dimensioni superiori a 1GB sulla macchina di sviluppo locale Windows. I download pesanti avvengono esclusivamente sulla macchina Linux remota o nei volumi container.
- **Divieto Assoluto di Burocrazia e Gate Formali**: Non reintrodurre manifest di SHA, gate formali di certificazione, verificatori a cascata bloccanti o file di piano rigidi che congelano il lavoro. Lo sviluppo è guidato da test funzionali e contratti snelli.
- Ogni modifica a codice o architettura deve aggiornare direttamente `docs/ARCHITECTURE.md` o i test pertinenti, senza creare documenti orfani o report di stato usa-e-getta.

---

## Principi di sviluppo obbligatori (istruzione per tutti gli agenti)

Ogni agente che lavora su questo repository deve applicare i tre principi seguenti:

### 1. Doubt-Driven Development (DDD)
Una risposta sicura non è una risposta corretta. Ogni decisione non banale viene verificata con rigore prima di essere considerata valida.
- Una decisione è **non banale** se introduce o modifica logica ramificata, attraversa un confine di modulo, tocca parametri numerici critici o ha un raggio d'impatto irreversibile.
- Processo: isolare l'artefatto minimo e il contratto; sottoporre a verifica avversariale ("trova cosa c'è di sbagliato, non limitarti a confermare"); riconciliare i punti emersi ed eliminare il codice non necessario.
- Non applicare DDD a operazioni meccaniche (rinominare, formattare, spostare file, eseguire tool).

### 2. Source-Driven Development (SDD)
Ogni decisione di codice specifica per un framework o libreria deve essere fondata sulla documentazione ufficiale o sul codice sorgente effettivo verificato.
- Isolare lo stack e le versioni esatte dal runtime o dai file di configurazione (`docker-compose.yml`, `pyproject.toml`).
- Verificare i contratti leggendo direttamente il codice implementativo nei moduli target.
- Se un comportamento non è documentato o verificato a runtime, dichiararlo esplicitamente invece di fare assunzioni.

### 3. Test-Driven Development (TDD)
Comportamento guidato da test snelli e mirati: RED -> GREEN -> REFACTOR.
- Prima scrivere o isolare il test che fallisce o cattura la proprietà richiesta; poi implementare la modifica minima per farlo passare; infine rifattorizzare rimuovendo codice difensivo o residuo.
- Su GPU RTX 3090: eseguire sempre uno smoke test breve (pochi batch o frame) prima di lanciare job pesanti o sessioni lunghe.

---

## Esecuzione su RTX 3090 (Host `mondaini` via Docker)

L'esecuzione avviene sulla macchina remota Linux con NVIDIA GeForce RTX 3090 (24GB VRAM), raggiungibile via SSH (host `mondaini`).

### Vincoli operativi e di sicurezza (obbligatori)
- **Container non-root**: eseguire sempre come utente `uid:gid 1000:1000` (`appuser`) per garantire che i file creati nei volumi montati mantengano permessi coerenti.
- **Shared memory**: usare sempre `--ipc=host` per prevenire errori di bus (`SIGBUS`) nel DataLoader PyTorch durante il trasferimento dei batch tra processi worker.
- **Isolamento di rete e privilegi**:
  - MAI usare `--privileged`.
  - MAI usare `--network host`. Per il serving REST usare esclusivamente il port mapping esplicito `-p 8000:8000`.
- **Volume di lavoro**: montare sempre il repository host in `/workspace` (`-v /home/administrator/gemma-4-omni:/workspace`).
- **Preservazione dei Checkpoint**: MAI cancellare o sovrascrivere checkpoint in `checkpoints/` (inclusi adapter QLoRA e Projector FiLM v2) senza esplicita richiesta dell'utente.

### Comandi standard su `mondaini`
```bash
# Avvio del servizio speech di produzione
./start_speech_service_3090.sh

# Verifica health endpoint
curl -s http://localhost:8000/health
```

---

## Delega Operativa a Codex CLI (GPT-6 Luna xhigh)

Tutte le implementazioni complesse non banali (> 50 righe, logica core multi-file, refactoring estesi) vengono affidate a **Codex CLI** con modello **`gpt-6-luna`** a massimo sforzo di ragionamento (**`xhigh`**).

### Regole di Delega
1. **Modello e Ragionamento**: usare rigorosamente `gpt-6-luna` con reasoning `xhigh`.
2. **Timeout Illimitato**: non impostare timeout artificiali brevi; lasciare a Codex il tempo necessario per il completamento.
3. **Contratti Tecnici Rigorosi**: passare a Codex specifiche deterministiche, interfacce, vincoli e criteri di verifica; non delegare decisioni architetturali di alto livello aperte.
4. **Strumento di Esecuzione**:
   `$null | codex exec -m gpt-6-luna -c model_reasoning_effort="xhigh" --dangerously-bypass-approvals-and-sandbox -C "<workspace>" "<task_spec>"`
   oppure tramite lo script dedicato in `tools/codex_orchestrator.py`.
