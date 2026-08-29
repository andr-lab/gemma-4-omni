# Matrice di Addestramento — Componenti, Metodo, Dati (Aggiornata V6 / ADR-012)

**Data Aggiornamento**: 2026-08-26 · **Base**: docs/decisions/ADR-012-text-anchored-projection-v6.md, docs/VERDETTO_TASK12_ARCHITETTURA_V5.md
**Principio guida**: Ogni componente si allena SOLO con i dati che possono insegnargli il suo ruolo; la voce italiana NON si addestra (Qwen3-TTS congelato è già pronto).

---

## 1. LA MATRICE DEI COMPONENTI (V6 / STRADA B)

| # | Componente | Stato | COSA impara | COME (ricetta) | CON COSA (dati) | Evidenza |
|---|---|---|---|---|---|---|
| 1 | **Gemma-4-E2B** | 🔒 CONGELATA | — (preserva ragionamento, tool calling, testo) | Nessun gradiente; estrattore {raw}$ (audio) e {tx}$ (testo) | — | Vincolo di sistema; Thinker congelato 4-bit |
| 2 | **TextAnchoredProjection** | 🎯 **IN TRAINING V6** (25.19M params) | Mappare {raw} + h_{tx}$ nello spazio {cond}$ del talker in loop | Loss congiunta: CE($) + 0.5·CE({1..15}$) con warm-start audio e ReZero text | **CV26 Italiano 40.8h** (	rain_tokens_cv26_it_40.8h_31916s.parquet) + cache v2 | ADR-012; sblocca $ sotto entropia uniforme (.2 	o < 6.8$) |
| 3 | **Backbone Qwen3-TTS 1.7B** | 🔒 CONGELATO | Predire $ nativamente tramite codec_head | Ingestion autoregressiva di  = codec\_agg_{t-1} + h_{cond}[t]$ | Pesi pre-addestrati | Qwen3-TTS report; Loop A (25 ago) |
| 4 | **Sub-Talker Qwen3-TTS (MTP)** | 🔒 CONGELATO | Generare  \dots CB_{15}$ dal \_hidden$ del loop | Protocollo ufficiale orward_sub_talker_finetune | Pesi pre-addestrati | J-Moshi; Qwen3-TTS report |
| 5 | **Code2Wav Decoder** | 🔒 CONGELATO | Sintesi audio 24kHz PCM | Decodifica RVQ ufficiale in-process | Pesi ufficiali | Qwen3TTSTokenizerV2Model (ADR-011) |
| 6 | **Decision Maker (DM) Head** | ⏳ PIANIFICATO | Classificare eventi full-duplex (<speak>, <yield>, <backchannel>, <barge_in>) | Classificatore cross-entropy pesato sugli eventi rari da {raw}$ | **16 dialoghi sintetici stereo IT** (GT nativo) | DuplexCascade; survey full-duplex |

---

## 2. FLUSSO DI CONDIZIONAMENTO (V6)

`
Utente audio ──► Gemma 4 E2B (congelata) ──► h_raw (T, 1536) [Prosodia, stile]
                                          │
Risposta agente (testo) ──────────────────┴──► h_tx  (1, 1536) [Contenuto, fonetica]
                                                    │
                                                    ▼
                                    TextAnchoredProjection (25.19M)
                                                    │
                                                    ▼ h_cond (T, 2048)
      ┌─────────────────────────────────────────────┘
      ▼
Qwen3-TTS 1.7B Talker Loop:
   Passo t: inputs_t = codec_agg_{t-1} + h_cond[t]
      ├─ codec_head(last_hidden) ──> CB0 (Semantico)
      └─ sub_talker([past_hidden, CB0]) ──> CB1..15 (Acustici)
            │
            ▼
   Code2Wav Decoder ──► Audio 24kHz PCM
`

---

## 3. DATI E RUOLI

| Dataset | Contenuto | Ruolo | Usato da |
|---|---|---|---|
| **MDC26 / CV26 (40.8h)** | Italiano mono, clip + trascrizioni | Supervisione congiunta semantica e acustica | TextAnchoredProjection (V6) |
| **16 Dialoghi Sintetici IT** | Stereo interattivo con overlap / barge-in | Supervisione eventi di turno (GT nativo) | Turn-Taking DM Head |
| **Qwen3-TTS Base 1.7B** | Pesi pre-addestrati multilingue | Rendering vocale in loop e decoding Code2Wav | Talker & Vocoder (congelati) |
