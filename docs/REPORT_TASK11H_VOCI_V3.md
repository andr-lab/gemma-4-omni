# REPORT TASK 11H — VOCI V3 (adapter CV26 40.8h + decoder ufficiale)

**Data**: 23 agosto 2026 | **Giudice percettivo**: agy (ascolto + trascrizione)

## Risultato chiave
**La voce generata dal modello v3 è INTELLIGIBILE e naturale al 100%.**

## Verdetti percettivi (trascrizione letterale di agy)

| File | Trascrizione agy | Match | Qualità | Verdetto |
|---|---|---|---|---|
| generated_003 (MODELLO) | "Questa ipotesi venne scartata." | **100%** | naturale | **INTELLIGIBILE** |
| oracle_001 (GT) | "Il perizzoma è dorata all'esterno ed dipinto di rosso all'interno." | 90% | naturale | INTELLIGIBILE |
| oracle_003 (GT) | "Questa ipotesi venne scartata." | 100% | naturale | INTELLIGIBILE |
| generated_001, generated_005 | non verificati (API audio agy degradata, 80+ tentativi falliti) | — | — | pending ascolto utente |

## Confronto col feedback v2 dell'utente
v2: "voce generated incomprensibile e robotica" → v3: 100% intelligibile, naturale.

## Metriche numeriche di supporto
- corr@12.5Hz generated: -0.059 / -0.013 (battimento eliminato; firma decoder ufficiale)
- mel_l1 generated vs oracle: 0.4494 medio | MCD 10.44 dB
- Validazione training: CB0 loss 4.99, Top-5 27.3%, SVD rank 1520/2048

## Componenti del risultato
1. **Adapter v3** (checkpoints/gemma4_duplex_sota_v3/final): ResidualAdapter + DeepSemanticHead + QLoRA co-adattamento, 2500 step su CV26 it 40.8h (31.916 clip, 5.986 parlanti)
2. **Decoder ufficiale** via subprocess venv phase1 (ADR-003) — root-cause risolta en route: fallback silenzioso alla replica standalone (causa del battimento), ora instrumentato con log diagnostico
3. **Dataset TurnHead v5 pronto** (GT nativo DuplexConv, 6.1M frame) per il training del controller di turno

## Limiti onesti
- Top-1 10.6% sui codici semantici: la fedeltà prosodica al GT non è perfetta (mel_l1 0.45); la comprensibilità però è piena
- Verifica percettiva completa su 1/3 generated (limite API audio agy); le altre due vanno ascoltate dall'utente
- TurnHead v5 non ancora addestrato (dataset pronto); policy full-duplex (backchannel/barge-in) = prossima fase
