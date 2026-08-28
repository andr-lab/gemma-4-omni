# ADR-021: Standardizzazione delle Metriche di Fedeltà Acustica e Soluzioni alla Distorsione da Disallineamento nei Modelli Speech-to-Speech

- **Stato**: Accettato
- **Data**: 27 Agosto 2026
- **Autori**: Gemma 4 Omni Team
- **Contesto**: Valutazione della fedeltà vocale e diagnosi della distorsione negli Speech-LLM con backbone congelati

---

## 1. Standardizzazione delle Metriche Acustiche

Per tutte le valutazioni di sintesi audio da Gemma 4 Omni, la sola trascrizione ASR (WER/CER) è insufficiente poiché un segnale con parole intelligibili può comunque presentare distorsione severa (iper-compressione, rumore di intermodulazione, saturazione periodica).

La pipeline adotta ufficialmente la suite di metriche implementata in `tools/measure_thd_distortion.py`:

| Metrica | Formula / Metodo | Valore Target (Voce Naturale) | Soglia di Allarme (Distorsione) |
| :--- | :--- | :---: | :---: |
| **Crest Factor** | $\frac{\text{Peak}}{\text{RMS}}$ su finestra intera | **$5.0\text{--}8.5$** | **$< 3.5$** (Iper-compressione/Onda quadra) |
| **HNR** | Harmonics-to-Noise Ratio via autocorrelazione dei frame voiced (40ms) | **$> 3.0\text{ dB}$** (Tipico $4\text{--}15\text{ dB}$) | **$< 0.0\text{ dB}$** (Rumore non-armonico dominante) |
| **THD+N** | $\frac{100}{1 + 10^{\text{HNR}/10}}\%$ (Energia non armonica + rumore) | **$< 30\%$** | **$> 45\%$** (Distorsione percepibile) |
| **Spectral Flatness** | Entropia di Wiener $\frac{\exp(\mathbb{E}[\ln S(f)])}{\mathbb{E}[S(f)]}$ | **$< 0.10$** | **$> 0.25$** (Rumore bianco / Fritto) |
| **Rumble Sub-Bass** | $\frac{\int_0^{80\text{Hz}} |S(f)|^2 df}{\int |S(f)|^2 df} \times 100\%$ | **$< 1.0\%$** | **$> 10.0\%$** (Rombo/Drift continuo a bassa frequenza) |

---

## 2. Indagine Letteratura arXiv: Come gli Altri Risolvono la Distorsione dei Residui Acustici

Dall'analisi sistematica dei principali Speech-to-Speech LLM ([Freeze-Omni](https://arxiv.org/abs/2411.00774), [LLaMA-Omni](https://arxiv.org/abs/2409.06666), [Moshi](https://arxiv.org/abs/2410.00037), [CosyVoice 2](https://arxiv.org/abs/2412.10117), [F5-TTS](https://arxiv.org/abs/2410.06885), [Mini-Omni 2](https://arxiv.org/abs/2410.11190)):

### A. Il Problema Fondamentale dell'Autoregressione su 16 Codebook RVQ
I modelli che tentano di generare $16$ quantizzatori RVQ in cascata autoregressiva da un backbone congelato soffrono di:
1. **Accumulo esponenziale di errore (Exposure Bias)**: Qualsiasi piccola divergenza nello stato latente $h_t$ fa sfasare i quantizzatori profondi ($cb_4 \dots cb_{15}$), producendo rumore di intermodulazione nel vocoder neuronale (SnakeBeta).
2. **Assenza di pause naturali**: Senza un meccanismo esplicito di silenzio / energy gating, la proiezione continua eccita il vocoder a ogni singolo frame, facendo collassare il Crest Factor a $2.1\text{--}2.5$.

### B. Le 3 Soluzioni Architetturali Adottate nella Letteratura SOTA

1. **Decoupling Semantico / Flow Matching (CosyVoice 2 [arXiv:2412.10117] / F5-TTS [arXiv:2410.06885])**:
   - L'LLM predice **esclusivamente un flusso di token semantici a singolo livello** (12.5Hz o 25Hz con FSQ).
   - Tutti i dettagli acustici, armoniche e fase vengono sintetizzati da un modulo **Conditional Flow Matching (DiT)** non-autoregressivo, eliminando all'origine la cascata RVQ a 16 livelli e la relativa distorsione.
2. **Single-Codebook / NAR CTC (Freeze-Omni [arXiv:2411.00774] & LLaMA-Omni [arXiv:2409.06666])**:
   - Freeze-Omni impiega **TiCodec a singolo codebook (1024 token a 40Hz)** escludendo a monte la necessità di predire 15 residui gerarchici attraverso 40 layer congelati.
   - LLaMA-Omni usa un decoder non-autoregressivo (NAR Transformer) con loss **CTC** su unità discrete HuBERT + vocoder HiFi-GAN con duration predictor.
3. **Acoustic Delay & Entropy Gating (Moshi [arXiv:2410.00037])**:
   - Introduce un **ritardo acustico ($\tau = 1$ o $2$ frame)** tra i token semantici e quelli acustici per ridurre il vincolo di informazione istantanea.
   - Utilizza un **Depth Transformer** con parametri dedicati per ciascun livello di quantizzazione e uno spazio dedicato di token di silenzio per azzerare l'entropia del rumore nelle pause.
