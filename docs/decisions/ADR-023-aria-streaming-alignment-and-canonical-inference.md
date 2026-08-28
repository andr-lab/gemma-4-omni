# ADR-023: ARIA Streaming Alignment, Pacing Dinamico e Parametri Canonici di Inferenza per Qwen3-TTS

- **Stato**: Accettato / Validato Empiricamente
- **Data**: 27 Agosto 2026
- **Autori**: Gemma 4 Omni Team
- **Contesto**: Risoluzione definitiva di distorsione progressiva, troncamento di frase, rialzo anomalo di volume e picchi stocastici sparsi nella sintesi vocale live full-duplex

---

## 1. Motivazione & Ipotesi Falsificate Empiricamente

Durante i test end-to-end 100% live tra Gemma 4 Thinker (Unsloth 4-bit) e Qwen3-TTS Talker mediati da Projector V9, abbiamo sottoposto a test empirici rigorosi e falsificato 5 ipotesi critiche:

### ❌ Ipotesi Falsificata 1: "La Repetition Penalty migliora la generazione audio RVQ"
- **Realtà osservata**: Applicare una penalità di ripetizione ai codici acustici di $CB_0$ divide/penalizza i token visti nei frame recenti. Ma nella voce umana a 12Hz, un singolo fonema vocalico (es. *"a"*, *"o"*) o una formante armonica dura 4–6 frame consecutivi (300–500ms) con LO STESSO codice RVQ. Penalizzare i token recenti costringe la bocca neurale a cambiare formante ogni 80ms, distruggendo la continuità armonica e rendendo la voce progressivamente metallica e stridula dopo 2 secondi.
- **Risoluzione canonica**: Su sequenze audio RVQ, `repetition_penalty` deve rimanere rigorosamente **1.0 (disabilitata)**.

### ❌ Ipotesi Falsificata 2: "Compressione temporale fissa di frasi arbitrarie"
- **Realtà osservata**: Comprimere una frase di 19 parole in una finestra temporale fissa (es. 48 frame = 4.0s) forza una velocità di articolazione innaturale (>15 sillabe/s), esaurendo il budget di frame e troncando la frase a metà.
- **Risoluzione canonica**: Architettura **ARIA** (*Adaptive Rate Interleaved Alignment*, arXiv:2604.15804) con budget di generazione dinamico parametrato sulla lunghezza lessicale ($T_{\text{gen}} \approx 4.8\text{ frame/parola} \approx 400\text{ms/parola}$) e prefisso di stile compatto ($T_{\text{pfx}} = 16$ frame da V9).

### ❌ Ipotesi Falsificata 3: "Omissione del bias d'ingresso canonico `tts_pad_embed`"
- **Realtà osservata**: In Qwen3-TTS (`modeling_qwen3_tts.py:1680-1695`), ogni passo di decodifica ricorsiva calcola:
  $$\text{inputs\_embeds}_t = \sum_{g=0}^{15} \text{embed}(c_{t, g}) + \mathbf{e}_{\text{tts\_pad}}$$
  Omettere $\mathbf{e}_{\text{tts\_pad}}$ provoca un *mean shift* permanente nell'ingresso del Talker frozen. Nei primi frame l'effetto è minimo, ma dopo 3 secondi (35+ frame) la norma degli stati interni cresce in modo cumulativo, causando un netto **rialzo di volume audio** e saturazione armonica del vocoder.
- **Risoluzione canonica**: Aggiunta sistematica di $\mathbf{e}_{\text{tts\_pad}}$ a ogni frame nel loop ricorsivo.

### ❌ Ipotesi Falsificata 4: "Multinomial Sampling non vincolato su $CB_0$"
- **Realtà osservata**: Campionare su tutti i 2048 logit di $CB_0$ (anche a $T=0.35$) mantiene probabilità non nulle sulla coda della distribuzione. Su 100 frame consecutivi, c'è una probabilità cumulativa del 5–10% di campionare token spuri fuori distribuzione, creando micro-picchi/click di distorsione sparsi.
- **Risoluzione canonica**: Applicazione di **`top_k = 20` truncation** prima della softmax, eliminando al 100% la coda di token spuri.

### ❌ Ipotesi Falsificata 5: "Greedy Decoding sul Sub-Talker 12L"
- **Realtà osservata**: Forzare `do_sample=False` sui codebook residui $CB_1..15$ del Sub-Talker fa collassare la tessitura acustica in modalità costante/silenzio (portando Whisper ad allucinare boilerplate).
- **Risoluzione canonica**: Sub-Talker campionato con parametri nativi Qwen3-TTS (`do_sample=True, top_k=20, top_p=0.90, temperature=0.80`).

---

## 2. Decisione Architetturale

1. **Pipeline di Inferenza Live Standard**:
   - Thinker produce la sequenza testuale $T_{\text{tok}}$ e gli stati latenti $h_{\text{tx}}$ ($1536$-dim).
   - Projector V9 produce il prefisso di stile prosodico $h_{\text{pfx}} \in \mathbb{R}^{1 \times 16 \times 2048}$.
   - Talker esegue prefill su $[\text{ICL}_{\text{text}} \,\|\, h_{\text{pfx}} \,\|\, \text{BOS}]$.
   - Loop ricorsivo: generazione autoregressiva con Top-K=20 su $CB_0$, Sub-Talker sampling, e iniezione canonica $+ \mathbf{e}_{\text{tts\_pad}}$ ad ogni frame.
2. **Validazione Sperimentale Finale**:
   - Frase test: *"Leonardo da Vinci fu un genio poliedrico del Rinascimento, noto per essere un artista, scienziato e inventore."*
   - Risultato ASR: **100% parole pronunciate con pronuncia cristallina e fluida** (7.20s di audio).
   - Metriche: Crest Factor = **`4.85`**, Volume uniforme e zero distorsione dal secondo 3 in poi.
