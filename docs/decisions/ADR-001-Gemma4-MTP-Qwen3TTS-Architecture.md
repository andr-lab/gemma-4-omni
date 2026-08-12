# ADR-001: Architettura Full-Duplex Speech-to-Speech a Teste Parallele (Dual-Head) con Gemma 4, MTP e Qwen3-TTS

## Stato
**Superato da ADR-002 (Superseded by ADR-002)**

## Data
2026-08-04

## Contesto e Validazione Scientifica Approfondita

L'analisi dei due paper cardine della letteratura Speech LLM (**Mini-Omni** [*arXiv:2408.16725*] e **LLaMA-Omni** [*arXiv:2409.06666*]) fornisce indicazioni cruciali per la correttezza metodologica dell'architettura **Gemma-4-Omni**:

---

### 1. Lezioni dal Paper Mini-Omni (arXiv:2408.16725)

1. **Codebook Delay Pattern (MusicGen-style)**:
   - **Problema**: La generazione parallela di 15 codebook RVQ nello stesso passo temporale senza priorità gerarchica porta ad instabilità acustica e "Acoustic Drift".
   - **Soluzione**: Applicazione del **Delay Pattern** (ritardo progressivo di $k$ token per il $k$-esimo codebook). Il Codebook 0 (semantico) viene generato al tempo $t$, il Codebook 1 al tempo $t+1$, il Codebook 2 al tempo $t+2$, e così via. Questo stabilizza la sintesi vocale e migliora la qualità audio senza aumentare la latenza perceptible.
2. **Backbone Congelato + Audio Projection Heads**:
   - Conferma che mantenere il backbone LLM quantizzato (Gemma 4 4-bit QAT) ed addestrare esclusivamente gli adattatori e le teste acustiche parallele è l'approccio ottimale per preservare le capacità di ragionamento del modello.

---

### 2. Lezioni dal Paper LLaMA-Omni (arXiv:2409.06666)

1. **Upsampling degli Hidden States & Decodifica Non-Autoregressiva (NAR)**:
   - Poiché la sequenza di token acustici è più lunga del testo, gli hidden states dell'LLM ($H$) vengono passati attraverso uno strato di **Upsampling 1D-Conv** prima di alimentare il decodificatore streaming (ConvNet / Code2Wav).
2. **Allineamento tramite Loss Combinata (CTC + Cross-Entropy)**:
   - LLaMA-Omni dimostra che combinare una **CTC Loss (Connectionist Temporal Classification)** con la Cross-Entropy standard permette agli hidden states del testo di allinearsi naturalmente alla frequenza di campionamento acustica a 12.5 Hz.

---

## Diagramma dell'Architettura Aggiornato (Dual-Head + Delay Pattern)

```
                               +-----------------------------------+
                               |     Input Audio Grezzo (16 kHz)   |
                               +-----------------------------------+
                                                 |
                                                 v
                        +-------------------------------------------------+
                        |   Gemma 4 12B Backbone (4-bit QAT / QLoRA)      |
                        |          Pesi Vocabolario Invariati             |
                        +-------------------------------------------------+
                                                 |
                                                 v
                                        Hidden States (H)
                                                 |
                                  Upsampling 1D Conv (12.5 Hz)
                                                 |
                       +-------------------------+-------------------------+
                       |                                                   |
                       v                                                   v
        +------------------------------+                +------------------------------------+
        |       Text LM Head           |                |   Parallel Audio Codebook Heads    |
        |  (Pesi Gemma Congelati/LoRA) |                | (Delay Pattern Shift: Codebook k)  |
        +------------------------------+                +------------------------------------+
                       |                                                   |
                       v                                                   v
               Predicts Text Token                               Predicts RVQ Audio Codebooks
                                                                      (12.5 Hz / 80ms step)
                                                                                   |
                                                                                   v
                                                                +------------------------------------+
                                                                |   Qwen3-TTS Code2Wav ConvNet       |
                                                                |       Streaming Synthesis          |
                                                                +------------------------------------+
```

---

## Scheda Tecnica e Decisioni Metodologiche

| Metodo | Fonte Scientifico | Applicazione in Gemma-4-Omni |
| :--- | :--- | :--- |
| **Parallel Projection Heads** | Mini-Omni & LLaMA-Omni | 15 strati lineari in BF16/FP32 collegati agli hidden states di Gemma 4. |
| **Codebook Delay Pattern** | Mini-Omni (MusicGen-style) | Sfasamento di 1 step tra i 15 livelli RVQ per stabilizzare l'acustica. |
| **Weighted Loss (Silenzio)** | Mini-Omni / Moshi | Abbattimento del peso della loss ($\approx 0.2$) per i token di silenzio. |
| **Upsampling 1D-Conv** | LLaMA-Omni | Adattamento di frequenza tra hidden states dell'LLM e stream acustico a 12.5 Hz. |
