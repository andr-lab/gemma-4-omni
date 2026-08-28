# ADR-013: Protocollo Diagnostico Post-Training — Manifold Alignment & SLERP Oracle Probe

## Status
**Accepted** (Implementato e pronto in `tools/diagnostics_manifold_oracle.py`)

## Data
2026-08-26

## Contesto e Origine
Durante il training di **Gemma4-Omni V6** (`TextAnchoredProjection` su Gemma 4 E2B + Qwen3-TTS 1.7B Talker congelato), la loss semantica su $CB_0$ è scesa da $13.26$ a un minimo locale di $4.49$ (sotto la soglia di pura entropia uniforme $\ln(4096) = 8.32$). Tuttavia, la loss sui residui acustici $CB_1..CB_{15}$ è rimasta relativamente alta ($res \approx 8.2$), producendo negli step intermedi campioni audio con timbro realistico ma balbettii fonetici veloci che ingannano il modello linguistico di Whisper (allucinazioni di trascrizione).

Una revisione tecnica esterna ha suggerito che $h_{cond}$ stia operando in un regime di **Cross-Model Latent Interface Learning**, in cui l'informazione semantica grossolana è sufficiente ad attivare il `codec_head` ma la geometria dello spazio latente non rispetta ancora la distribuzione, la scala e la dinamica attesa dal Talker Qwen.

Per evitare modifiche cieche a iperparametri (LR, batch, architettura) prima di avere prove empiriche certe, è stato definito un protocollo diagnostico rigoroso sottoposto a **Doubt-Driven Development (DDD)** e verificato con **Source-Driven Development (SDD)** sulla documentazione ufficiale e sul paper di Qwen3-TTS ([arXiv:2601.15621](https://arxiv.org/abs/2601.15621)).

---

## Decisione Architetturale

Implementare lo script standalone [`tools/diagnostics_manifold_oracle.py`](file:///c:/Users/andre/Documents/gemma-4-omni/tools/diagnostics_manifold_oracle.py) da eseguire a fine training V6 (o sui checkpoint intermedi `step1000`/`step2000`).

Lo script realizza **4 sonde diagnostiche quantitative disaccoppiate**:

### 1. Metrica 1: Allineamento Geometrico di Manifold (Output Space)
* **Cosa confronta**: L'attivazione `past_hidden` prodotta dal transformer a 28 layer del Talker stimolato da $h_{cond}$ (regime Omni) vs l'attivazione `past_hidden` prodotta dal Talker stimolato dal prompt testuale nativo (regime Native).
* **Perché all'uscita e non all'ingresso (Risoluzione del Category Error DDD)**:
  In Qwen3-TTS nativo, il testo entra nel prefill (`icl_input_embed`). Nel nostro V6, $h_{cond}$ entra per-frame come residuo sommato a `codec_agg_{t-1}`. Confrontare $h_{cond}$ (input pre-layer 0) con `past_hidden` (output post-layer 28) sarebbe un errore di categoria. Confrontando `past_hidden` contro `past_hidden` si misura la risposta finale del modello a parità di piano di riferimento.
* **Misure statistiche**:
  * Cosine Similarity media nel tempo: $\text{Cosine}(past\_hidden_{omni}[t], past\_hidden_{native}[t])$.
  * Rapporto delle norme euclidee: $\|past\_hidden_{omni}\|_2 / \|past\_hidden_{native}\|_2$.
  * Rango Effettivo SVD (entropia spettrale dei valori singolari) per individuare eventuale collasso dimensionale.
  * Smoothness temporale (derivata prima $\|\Delta past\_hidden / \Delta t\|_2$) per misurare jitter o rumore inter-frame.

### 2. Metrica 2: Cross-Entropy Disaggregata dei Codebook
* **Isolamento delle formanti dal micro-rumore spettrale**:
  * $\mathcal{L}_{cb0}$: Cross-Entropy su $CB_0$ (vocab 4096, semantica/fonetica).
  * $\mathcal{L}_{res\_primary}$: Cross-Entropy sui primi 3 livelli $CB_1..CB_3$ (vocab 2048, responsabili diretti di F0 pitch contour e formanti vocali).
  * $\mathcal{L}_{res\_high}$: Cross-Entropy sui livelli residui $CB_4..CB_{15}$ (vocab 2048, micro-dettaglio acustico ad alta frequenza e rumore naturale).

### 3. Metrica 3: SLERP Oracle Interpolation Curve
* **Interpolazione Sferica Normalizzata**:
  Per testare la risposta del Talker a transizioni graduali tra il regime nativo e il regime Omni senza subire il collasso di norma euclidea $\approx -10.5\%$ tipico delle medie lineari in 2048D:
  $$h_{mix}[t] = \text{SLERP}(past\_hidden_{native}[t], past\_hidden_{omni}[t], \alpha) \quad \text{per } \alpha \in [0.0, 0.25, 0.50, 0.75, 1.0]$$
* **Diagnosi Differenziale**:
  * Se a $\alpha = 0.5$ e $\alpha = 1.0$ la residual CE su $CB_1..CB_3$ rimane bassa $\implies$ il manifold geometrico è compatibile e la proiezione è corretta.
  * Se la loss residua esplode appena $\alpha > 0 \implies$ mismatch di scala/covarianza latente nel manifold del Talker.

### 4. Metrica 4: Misura dell'Exposure Bias Autoregressivo
* Valutazione in generazione libera (greedy + temperature decay) per tracciare a quale frame temporale $t$ un eventuale errore su $\hat{CB}_0$ propaga instabilità al sub-talker rispetto al teacher forcing.

---

## Fondamento Scientifico (Source-Driven)

1. **Alibaba Qwen3-TTS Technical Report ([arXiv:2601.15621](https://arxiv.org/abs/2601.15621))**:
   * *Dual-Track Architecture*: Conferma che Qwen3-TTS 12Hz adotta un'architettura gerarchica con Backbone per $CB_0$ a 12.5 Hz (80ms/frame) e MTP Sub-Talker (5 layer causal transformer) per $CB_1..CB_{15}$.
   * *Frame Synchrony*: La frequenza a 12.5 Hz è bit-pari col downsampling $8\times$ dell'encoder Conformer di Gemma 4 E2B.
2. **Qwen Official Codebase (`qwen_tts/core/models/modeling_qwen3_tts.py`)**:
   * Utilizzo delle API native `talker.model(...)`, `talker.codec_head(...)` e `talker.forward_sub_talker_finetune(...)` per estrazione on-the-fly senza dipendenza da cache degradate.

---

## Conseguenze e Decisioni Operative

### 1. Esecuzione Obbligatoria a Fine Training V6
Appena il container `v6_thinker_talker_run` su RTX 3090 termina i 2000 step, lo script [`tools/diagnostics_manifold_oracle.py`](file:///c:/Users/andre/Documents/gemma-4-omni/tools/diagnostics_manifold_oracle.py) **deve essere eseguito sui checkpoint (`step1000`, `step2000`)** per produrre la matrice quantitativa oggettiva e diagnosticare l'esatto stato di compatibilità del manifold prima di qualsiasi decisione architetturale su V7.

### 2. Standardizzazione delle Metriche di Eval per le Prossime Run (V7+)
Nelle prossime sessioni di training, l'eval in-loop online basata su Whisper ASR viene **ufficialmente sostituita** dalla sonda di allineamento manifold e Cross-Entropy disaggregata:
* **Perché**: Whisper ASR introduce latenza elevata (~20s), consuma VRAM aggiuntiva (~1GB) e produce allucinazioni linguistiche fuorvianti su babble fonetico.
* **Cosa si valuta nel loop continuo (ogni 100 step)**:
  - $\text{Cosine}(past\_hidden_{omni}, past\_hidden_{nat})$
  - $\text{Acc}_{top1}(CB_0)$ & $\mathcal{L}_{cb0}$ (Semantica / Fonetica)
  - $\mathcal{L}_{res1..3}$ (Formanti vocali e Pitch contour F0)
  - $\mathcal{L}_{res4..15}$ (Micro-rumore spettrale)
  - Tempo di calcolo: $< 50\text{ms}$ (zero overhead, zero rischio OOM).
* **Ruolo dei file audio `.wav`**: La sintesi audio tramite Code2Wav viene limitata ai soli checkpoint principali (es. ogni 500 step e checkpoint finale) ad esclusivo scopo di audit e ascolto percettivo umano.
