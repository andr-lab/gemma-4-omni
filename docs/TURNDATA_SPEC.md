# Specifica Tecnica del Dataset per la TurnStateHead (Stage 3)
## Formato, Generazione Ground Truth a 12.5 Hz e Campionamento Bilanciato

- **Progetto**: Gemma 4 Full-Duplex Omni Pipeline (`C:\Users\andre\Documents\gemma-4-omni`)
- **Modulo**: STAGE 3 — `TurnStateHead` (5 stati di turno)
- **Data**: 20 Agosto 2026
- **Autore**: Ingegnere ML Senior / Lead Architect

---

## 1. Obiettivo e Specifiche del Segnale di Supervisione

La `TurnStateHead` richiede una supervisione discreta a livello di frame ($T=64$, 5.12s a 12.5 Hz = 80ms per frame) per apprendere la transizione di stato tra 5 dinamiche conversazionali fondamentali.

### 1.1. Spazio delle Classi (5 Stati di Turno)

```
        ┌────────────────────────────────────────────────────────┐
        │                 5 STATI DELLA TURNSTATEHEAD             │
        ├──────┬───────────────────────┬─────────┬───────────────┤
        │ ID   │ Nome Simbolico        │ Peso CE │ Dinamica      │
        ├──────┼───────────────────────┼─────────┼───────────────┤
        │  0   │ STATE_SPEAK           │   1.0   │ Monologo Ag.  │
        │  1   │ STATE_YIELD           │   8.0   │ Cessione Turno│
        │  2   │ STATE_BARGE_IN        │   6.0   │ Interruzione  │
        │  3   │ STATE_USER_BACKCHANNEL│   3.0   │ Feedback Ut.  │
        │  4   │ STATE_EMIT_BACKCHANNEL│   4.0   │ Assenso Agente│
        └──────┴───────────────────────┴─────────┴───────────────┘
```

1. **`0: STATE_SPEAK`**: L'agente ha il turno e sta parlando, mentre l'utente e silenzioso o in ascolto passivo. La sintesi $h_t$ procede a piena banda.
2. **`1: STATE_YIELD`**: L'agente sta per concludere la propria frase (ultimi 3 frame = ~240ms prima dell'arresto dell'emissione). Segnala al client streaming di prepararsi alla ricezione utente.
3. **`2: STATE_BARGE_IN`**: L'utente inizia a parlare mentre l'agente e in emissione attiva, con durata $>400\text{ms}$ (presa di parola competitiva). Trigger immediato di stop/muting sul sub-talker e flush del buffer.
4. **`3: STATE_USER_BACKCHANNEL`**: L'utente emette un breve feedback ("mhm", "si", "ok") durante il parlato dell'agente con durata $<400\text{ms}$. L'agente NON interrompe la sintesi vocale.
5. **`4: STATE_EMIT_BACKCHANNEL`**: L'agente e in ascolto passivo da oltre 1.5s e rileva una pausa prosodica/sintattica dell'utente ($300-800\text{ms}$). Trigger per la riproduzione di un breve clip di assenso a latenza zero.

---

## 2. Derivazione Algoritmica delle Label `turn_labels`

Dati:
- $T$: Numero di frame per finestra (default: 64 frame agente @ 12.5 Hz = 5.12s).
- $\text{FrameDuration} = 0.080\text{ s}$ (80 ms).
- Intervallo temporale del frame $t$: $I_t = [t \cdot 0.080, (t+1) \cdot 0.080]$.
- Utterance utente: $\mathcal{U}_{\text{user}} = \{(s_i, e_i, \text{txt}_i)\}$.
- Utterance agente: $\mathcal{U}_{\text{agent}} = \{(s_j, e_j, \text{txt}_j)\}$.

### 2.1. Algoritmo di Etichettatura Deterministica (Frame-by-Frame)

Per ogni frame $t \in [0, T-1]$:
1. **Calcolo Attivita di Canale**:
   - $u_t = \mathbb{I}(\exists (s, e) \in \mathcal{U}_{\text{user}} \text{ t.c. } [s, e] \cap I_t \neq \emptyset) \lor (\text{UserEnergy}_t > \text{thr}_{\text{user}})$
   - $a_t = \mathbb{I}(\exists (s, e) \in \mathcal{U}_{\text{agent}} \text{ t.c. } [s, e] \cap I_t \neq \emptyset) \lor (\text{AgentEnergy}_t > \text{thr}_{\text{agent}})$
2. **Determinazione Maschera Valida**:
   - $\text{valid\_mask}_t = u_t \lor a_t$ (1 se almeno un canale e attivo; 0 se entrambi sono silenzio/padding).
3. **Assegnazione Classe $y_t$**:
   - Se $a_t = 1 \land u_t = 0$:
     - Se $t$ e all'interno degli ultimi 3 frame di un'emissione agente (ovvero $a_{t+1}=0 \lor a_{t+2}=0 \lor a_{t+3}=0$):
       $$y_t = 1 \quad (\text{STATE\_YIELD})$$
     - Altrimenti:
       $$y_t = 0 \quad (\text{STATE\_SPEAK})$$
   - Se $a_t = 1 \land u_t = 1$ (Sovrapposizione):
     - Sia $\Delta t_{\text{user}}$ la durata continua dell'intervento utente che interseca il frame $t$:
     - Se $\Delta t_{\text{user}} \ge 0.400\text{ s}$ (o parole $>2$):
       $$y_t = 2 \quad (\text{STATE\_BARGE\_IN})$$
     - Se $\Delta t_{\text{user}} < 0.400\text{ s}$:
       $$y_t = 3 \quad (\text{STATE\_USER\_BACKCHANNEL})$$
   - Se $a_t = 0 \land u_t = 1$ (Solo Ascolto Utente):
     - Se l'utente e in una pausa intra-discorso ($300-800\text{ms}$) o ha completato una proposizione ($\text{state}=\text{<\|complete\|>}$):
       $$y_t = 4 \quad (\text{STATE\_EMIT\_BACKCHANNEL})$$
     - Altrimenti:
       $$y_t = 0 \text{ (o frame di ascolto)}$$

---

## 3. Gestione delle Finestre di SOLO ASCOLTO e Dataloader Anti-Collasso

### 3.1. Il Problema del Filtro `_mel_active_mask`
Nello Stage 2 precedente, `_mel_active_mask` scartava il 100% delle finestre in cui l'agente era in silenzio. Se applicato alla TurnStateHead, il modello non vedrebbe MAI situazioni in cui l'utente parla e l'agente deve restare in ascolto, rendendo impossibile imparare a discriminare `SPEAK` vs `LISTEN` ed `EMIT_BACKCHANNEL`.

### 3.2. Campionamento Stratificato a 3 Bucket
Il `TurnHeadDataset` raggruppa le finestre in tre categorie bilanciate:
1. **Bucket 1: Monologo Agente (40%)**:
   - Finestre con $\text{AgentFrames} \ge 20 \land \text{UserFrames} \le 5$.
   - Fornisce esempi robusti di `STATE_SPEAK` (0) e transizioni di chiusura `STATE_YIELD` (1).
2. **Bucket 2: Solo Ascolto Utente (30%)**:
   - Finestre con $\text{UserFrames} \ge 20 \land \text{AgentFrames} \le 5$.
   - Insegna alla head a non eccitarsi su `SPEAK` e a identificare i punti di ancoraggio `STATE_EMIT_BACKCHANNEL` (4).
3. **Bucket 3: Sovrapposizioni e Barge-In (30%)**:
   - Finestre con $\text{AgentFrames} \ge 10 \land \text{UserFrames} \ge 10$.
   - Cruciale per discriminare `STATE_BARGE_IN` (2) da `STATE_USER_BACKCHANNEL` (3).

---

## 4. Input Features Vector $z_t \in \mathbb{R}^{2304}$

Per ogni frame $t \in [0, T-1]$:
$$z_t = \Big[ h_t^{\text{temp}} \,(2048\text{D}) \;\Big\|\; e_t^{\text{user}} \,(128\text{D}) \;\Big\|\; \Delta e_t^{\text{user}} \,(128\text{D}) \Big] \in \mathbb{R}^{2304}$$
- $h_t^{\text{temp}} \in \mathbb{R}^{2048}$: Rappresentazione latente contestualizzata estratta dal `TemporalResidualAdapter` congelato.
- $e_t^{\text{user}} \in \mathbb{R}^{128}$: Vettore di energia mel utente mediato sugli 8 frame mel corrispondenti al frame $t$:
  $$e_t^{\text{user}} = \frac{1}{8} \sum_{k=0}^7 \text{mel}_{\text{user}}[8t + k, :]$$
- $\Delta e_t^{\text{user}} \in \mathbb{R}^{128}$: Derivata temporale discreta del profilo spettrale utente:
  $$\Delta e_t^{\text{user}} = e_t^{\text{user}} - e_{t-1}^{\text{user}} \quad (\text{con } e_{-1}^{\text{user}} = e_0^{\text{user}})$$

---

## 5. Loss Function e Metriche di Validazione

### 5.1. Weighted Cross-Entropy Loss
$$\mathcal{L}_{\text{turn}} = - \frac{1}{\sum_{t=1}^T \text{valid\_mask}_t} \sum_{t=1}^T \text{valid\_mask}_t \sum_{k=0}^4 w_k \cdot \mathbb{I}(y_t = k) \log \sigma(\mathbf{s}_t)_k$$
con $\mathbf{w} = [1.0, 8.0, 6.0, 3.0, 4.0]$.

### 5.2. Metriche Monitorate in Report (`stage3turnhead_report.json`)
- **Overall Accuracy**: Frazione di predizioni corrette sui frame validi ($\ge 0.70$).
- **Per-Class F1-Score**: $F_1^{(0)}$ (SPEAK), $F_1^{(1)}$ (YIELD), $F_1^{(2)}$ (BARGE_IN $\ge 0.30$), $F_1^{(3)}$ (USER_BC), $F_1^{(4)}$ (EMIT_BC).
- **Confusion Matrix**: Matrice $5 \times 5$ per rilevare eventuali confusioni tra Barge-In e Backchannel.
- **Gradient Probes**: Verifica che $\text{grad}(\text{turnhead}) > 0$ e $\text{grad}(\text{backbone}) = 0$, $\text{grad}(\text{adapter}) = 0$.
