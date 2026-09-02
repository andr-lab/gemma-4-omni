# P4D Lookahead Study Report: Valutazione Ritardo Temporale e Conclusione Architetturale

Data di attestazione: **02 Settembre 2026**  
Stato: **REJECTED — SOLUZIONE A RITARDO PASSIVO DICHIARATA INSOSTENIBILE**  
Piattaforma di esecuzione: **NVIDIA GeForce RTX 3090 (24 GB VRAM)** via Docker (`gemma4-omni-v5:latest`)  
Dataset di validazione: `cache_ars_runtime_aligned_v1.pt` (100 sequenze, 6.400 frame, allineamento rigoroso Thinker nativo e split validazione)  

---

## 1. Motivazione ed Obiettivo dello Studio

Lo scopo della fase P4D era determinare se fosse possibile colmare il gap qualitativo introdotto dalla causalità nel Talker congelato (**Qwen3-TTS 12Hz 1.7B Base**) applicando esclusivamente un ritardo temporale nello speaker playout buffer (`FuturePrefixQwen` con finestra di lookahead $L$).

Soglie di qualità previste dal piano originale (P4-T4):
- **CE Ratio**: $\le 1.10$ ($+10\%$ massimo rispetto all'oracolo full-sequence non causale)
- **Top-1 Drop**: $\le 5.0$ punti percentuali
- **Top-5 Drop**: $\le 3.0$ punti percentuali

---

## 2. Risultati Sperimentali Rilevati

I test sono stati eseguiti con l'ancora causale deterministica `first_raw_frame` su 100 sequenze consecutive della split di validazione:

| Variante | Lookahead ($L$) | Ritardo Playout | CE Ratio (target $\le 1.10$) | Top-1 Drop (target $\le 5.00$ pp) | Top-5 Drop (target $\le 3.00$ pp) | Esito Gate |
|---|---|---|---|---|---|---|
| `first_raw_frame_L0` | 0 frame | 0 ms | **1.4086** | 5.25 pp | 8.89 pp | **FAIL** |
| `first_raw_frame_L1` | 1 frame | 80 ms | **1.3928** | 4.84 pp | 8.36 pp | **FAIL** |
| `first_raw_frame_L2` | 2 frame | 160 ms | **1.3779** | 4.58 pp | 7.94 pp | **FAIL** |
| `first_raw_frame_L4` | 4 frame | 320 ms | **1.3545** | **4.55 pp (PASS)** | 7.37 pp | **FAIL** |

---

## 3. Analisi della Curva e Proiezione Matematica

1. **Recupero Lento**:  
   La pendenza di recupero medio è di appena **$\approx -0.0135$ CE ratio per frame** (80 ms).  
   Anche quadruplicando il ritardo da 80 ms a 320 ms ($L=4$), il CE ratio resta inchiodato a **1.3545**, ovvero **oltre 25 punti percentuali sopra la soglia di tolleranza**.

2. **Proiezione per il superamento del Gate**:  
   Per scendere da $1.3545$ a $1.1000$ ($\Delta = -0.2545$):
   $$\Delta L \approx \frac{0.2545}{0.0135} \approx 19 \text{ frame extra} \implies L \ge 23 \text{ frame}$$
   Tenendo conto della concavità asintotica naturale dei modelli di predizione, il lookahead necessario si colloca tra **$L=24$ e $L=40$ frame**, equivalente a un ritardo audio di **$1.92$ — $3.20$ secondi**.

3. **Incompatibilità con i Requisiti Full-Duplex**:  
   Per una conversazione bidirezionale naturale:
   * Qualsiasi ritardo superiore a **$500$ ms** ($L \approx 6$) distrugge l'interattività e genera sovrapposizioni o pause innaturali.
   * Entro il limite di mezzo secondo ($500$ ms), il CE ratio non scende sotto **$1.33$** ($+33\%$ di errore).

---

## 4. Decisione Architetturale Ufficiale

Su direttiva esplicita dell'utente:
> *"se la tendenza è negativa, annulliamo la run e dichiariamo insostenibile la soluzione del ritardo temporale. oltre il mezzo secondo diventa deleterio."*

* La run è stata **immediatamente arrestata** e le risorse GPU liberate.
* La soluzione del ritardo passivo nello speaker playout buffer è **formalmente dichiarata INSOSTENIBILE ed ABBANDONATA**.
* Per risolvere il gap causale senza intaccare la latenza conversazionale ($<200$ ms), l'architettura deve adottare una **soluzione attiva**:
  1. **`FuturePrefixDrafter`**: un modulo predittivo leggero che stima autoregressivamente i frame di prefisso futuro $t+1 \dots t+K$.
  2. **Co-adaptation / Fine-Tuning Causale**: adattamento dei pesi del proiettore o QLoRA su regime causale.
