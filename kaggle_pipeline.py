# %% [markdown]
# # PIPELINE FULL-DUPLEX DUAL-CHANNEL (GEMMA 4 E2B + QWEN3-TTS)
#
# Questa pipeline è strutturata in **2 FASI DISACCOPPIATE** (ADR-003) per eliminare qualsiasi conflitto tra `qwen-tts` e `transformers>=5.x`:
#
# - **FASE 1 (Celle 1 - 3)**: Setup ambiente `qwen-tts`, generazione di dialoghi stereo dual-channel con **Barge-in / Sovrapposizioni reali** (DuplexConv + MDC ITA), tokenizzazione RVQ con **barre di avanzamento tqdm** e salvataggio dei file Parquet su disco.
# - **FASE 2 (Celle 4 - 7)**: Upgrade dell'ambiente a `transformers>=5.x` e `unsloth`, caricamento immediato dei Parquet (<0.05s), sintesi con **Code2Wav Decoder** standalone, fine-tuning QLoRA di **Gemma 4 E2B** Dual-Channel Full-Duplex (usando l'encoder audio Conformer nativo per l'input utente secondo ADR-004) e salvataggio del checkpoint unificato.

# %% [code]
import os

# ============================================================================
# SINGLE-GPU (default): Gemma 4 E2B in 4-bit occupa ~3-4GB e sta su una sola
# GPU; il multi-GPU con device_map="balanced" ha dato errori CUDA illegal memory
# access in passato (dispatch + quantizzazione su più device). Forziamo una
# singola GPU PRIMA di qualsiasi import torch.
# ============================================================================
# ============================================================================
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Default single-GPU
import sys

if "." not in sys.path:
    sys.path.insert(0, ".")
import json
import time
import gc
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import concurrent.futures
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Tuple, Optional
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# DISCO GRANDE per i file temporanei pesanti (tar dataset, jsons, wav temporanei):
# env BIG_DISK_DIR se impostato, altrimenti ./data in locale. I PARQUET finali
# restano in data/ per la Fase 2.
# ---------------------------------------------------------------------------
_BIG_DISK_CACHE: Optional[str] = None


def big_disk_base() -> str:
    """Percorso del disco grande per i file temporanei pesanti (env > locale)."""
    global _BIG_DISK_CACHE
    if _BIG_DISK_CACHE is not None:
        return _BIG_DISK_CACHE
    try:
        import paths
        _BIG_DISK_CACHE = paths.get_big_disk_dir()
        return _BIG_DISK_CACHE
    except Exception:
        pass
    env = os.environ.get("BIG_DISK_DIR", "").strip()
    candidates = [env] if env else []
    candidates.append(os.path.join(os.getcwd(), "data"))  # fallback locale
    for c in candidates:
        try:
            os.makedirs(c, exist_ok=True)
            probe = os.path.join(c, ".write_test")
            with open(probe, "w") as f:
                f.write("1")
            os.remove(probe)
            _BIG_DISK_CACHE = c
            print(f"[Disco] File temporanei pesanti su: {c}")
            sys.stdout.flush()
            return c
        except Exception:
            continue
    _BIG_DISK_CACHE = os.path.join(os.getcwd(), "data")
    return _BIG_DISK_CACHE


# Costanti Architetturali Globali
MODEL_NAME = "unsloth/gemma-4-E2B-it"  # Backbone Gemma 4 E2B con USM-Conformer Audio Encoder nativo
MAX_SEQ_LENGTH = 2048
FRAME_RATE_HZ = 12.5  # 12.5 Hz = 80ms per frame acustico

# ---------------------------------------------------------------------------
# Iperparametri tokenizer Qwen3-TTS-12Hz — VERIFICATI sul config HF reale (2026-08-05):
#   encoder_valid_num_quantizers = 16   (1 layer SEMANTICO + 15 RVQ acustici)
#   codebook_size             = 2048    (vocabolario dei layer RVQ acustici)
#   semantic_codebook_size    = 4096    (vocabolario del layer semantico cb0, WavLM-guided)
#   num_semantic_quantizers   = 1
# (Fonte: Qwen3-TTS Technical Report arXiv:2601.15621 + config.json del repo HF.
#  In passato la pipeline usava 15x1024: il guard di shape scartava i codici reali a
#  16 righe e cadeva nel fallback FFT -> target degenerati -> loss collassata a ~0.002.)
# ---------------------------------------------------------------------------
N_CODEBOOKS = 16
SEMANTIC_VOCAB_SIZE = 4096  # vocabolario del primo codebook (semantico)
ACOUSTIC_VOCAB_SIZE = 2048  # vocabolario dei codebook RVQ acustici 1..15
CODEBOOK_VOCAB_SIZE = ACOUSTIC_VOCAB_SIZE  # alias di compatibilità (codebook acustici)
SILENCE_TOKEN_ID = 0  # Token ID di default (sostituito in Fase 1 dai codici di silenzio EMPIRICI estratti)
SILENCE_LOSS_WEIGHT = 0.2  # Peso per la loss sul silenzio nei codebook acustici 1..15
USER_AUDIO_TOKEN_OFFSET = (
    200000  # Offset legacy per il loopback discreto (solo inference generate/stream)
)

# ---------------------------------------------------------------------------
# SINTESI DI ESEMPIO A FINE TRAINING (solo diagnosi/ascolto, non partecipa al training)
# ---------------------------------------------------------------------------
SYNTH_OUTPUT_DIR = (
    "samples"  # wav ascoltabili (generati dal modello + riferimenti oracle)
)
SYNTH_SAMPLE_WINDOWS = 8  # quante finestre di validazione decodificare
SYNTH_TEMPERATURE = 0.7  # sampling temperature del generate (0 = argmax)
SYNTH_SAMPLE_RATE = 24000  # sample rate di uscita del decoder Code2Wav (24kHz)

# ---------------------------------------------------------------------------
# COUPLING QWEN3-TTS (sub-talker congelato) + DECODER STANDALONE Code2Wav
# ---------------------------------------------------------------------------
# Il coupling sostituisce le 15 teste MTP (approssimazione da zero dei residui RVQ)
# con il sub-talker di Qwen3-TTS-12Hz-1.7B-Base CONGELATO (5M ore di training):
# Gemma produce h_t (adapter trainabile 3072->2048) + cb0 (semantic_head) e il
# sub-talker genera i codebook acustici 1..15 a qualità nativa Qwen (ADR-005).
# Attivo automaticamente se la dir QWEN_SUBTALKER_DIR esiste (esegui in Fase 1:
#   !python extract_qwen_subtalker.py --out-dir qwen_subtalker
# con secret Kaggle HF_TOKEN — repo Base gated).
QWEN_SUBTALKER_DIR = (
    "qwen_subtalker"  # dir con qwen_subtalker.safetensors + config.json
)
CODE2WAV_WEIGHTS_DIR = (
    "code2wav_weights"  # decoder Code2Wav standalone (estratto in Fase 1)
)
COUPLING_LOSS_WEIGHT = 0.6  # scala della loss coupling rispetto a cb0
COUPLING_WARMUP_STEPS = 200  # primi N step: h_t.detach() (impara solo l'adapter)
COUPLING_TALKER_HIDDEN = 2048  # hidden del talker Qwen (input del sub-talker)
COUPLING_ACOUSTIC_VOCAB = 2048  # vocabolario dei codebook acustici del sub-talker

# ---------------------------------------------------------------------------
# PERCORSO AUDIO NATIVO Gemma 4 E2B (design ufficiale: Report sez. 2.2 + config HF).
# Il tokenizzatore audio del modello produce MEL log-spettrali a 100 fps (16kHz,
# frame 20ms, hop 10ms, 128 bin, nessuna normalizzazione per-bin: per_bin_mean=null).
# L'audio tower USM-Conformer (305M, 12 layer, congelato) fa subsampling 4x ->
# 25 fps (= audio_ms_per_token=40 del processor_config.json). Il LLM ingerisce le
# feature CONTINUE proiettate (NO VQ): i placeholder audio_token_id negli input_ids
# vengono sostituiti dalle feature dell'audio tower (masked_scatter nel forward).
# Token speciali dal config HF: audio_token_id=258881, boa_token_id=256000,
# eoa_token_id=258883, bos_token_id=2.
# Allineamento temporale con i codici agente Qwen3-TTS (12.5 fps / 80ms):
#   1 frame agente = 8 frame mel (100fps) = 2 feature audio (25fps).
# ---------------------------------------------------------------------------
MEL_SAMPLING_RATE = 16000
MEL_FRAME_MS = 20.0
MEL_HOP_MS = 10.0
MEL_BINS = 128
MEL_FRAME_RATE_HZ = 1000.0 / MEL_HOP_MS  # 100 fps
AUDIO_FEATURE_RATE_HZ = 25.0  # output dell'audio tower (subsampling 4x)
MEL_PER_AGENT_FRAME = int(round(MEL_FRAME_RATE_HZ / FRAME_RATE_HZ))  # 8
AGENT_FEATURES_PER_FRAME = int(round(AUDIO_FEATURE_RATE_HZ / FRAME_RATE_HZ))  # 2
AUDIO_TOKEN_ID = 258881
BOA_TOKEN_ID = 256000
EOA_TOKEN_ID = 258883
BOS_TOKEN_ID = 2
PAD_TARGET_ID = -100  # ignore_index per i frame agente di padding nella loss

# ---------------------------------------------------------------------------
# SUPERVISIONE TESTUALE (testo nei target): la sequenza LLM diventa
#   [bos, (TEXT_BLOCK) testo risposta agente, boa, feature audio x n_feat, eoa]
# Il blocco testo (token BPE Gemma 4, left-padding a TEXT_BLOCK_LEN con
# pad_token_id) PRECEDE le feature audio: causalmente le feature e i codici
# audio possono ATTENDERE al testo -> il modello impara "dato il testo che
# dirò, produrne l'audio nei tempi del duplex" (direzione testo->audio nativa
# di E2B). Il testo viene predetto alla posizioni del blocco con la lm_head
# condivisa (tied embeddings). Le frasi vengono tokenizzate in FASE 1 (niente
# dipendenza da transformers>=5: tokenizers lib sul tokenizer.json del repo).
# NOTA (verificata sul tokenizer.json): gli id 256000..258883 sono token
# SPECIALI riservati (<|audio|>, <unused*>) -> nessuna collisione col BPE testo.
TEXT_BLOCK_LEN = 48  # token BPE max per il blocco testo (truncate, left-pad)
TEXT_LOSS_WEIGHT = 0.5  # scala della loss testo rispetto a quella audio
E2B_PAD_TOKEN_ID = 0  # pad_token_id del tokenizer testo Gemma 4 (config)
E2B_BOS_TOKEN_ID = 2
E2B_EOS_TOKEN_ID = 1
# ---------------------------------------------------------------------------
# VALIDATION SPLIT: l'ultimo VAL_FRACTION di frame viene tenuto fuori dal train
# a un confine di SEGMENTO (conversazione/turno) annotato in Fase 1
# (segment_ends nel sidecar) -> le finestre di validazione non tagliano a metà
# una conversazione.
VAL_FRACTION = 0.10

# ---------------------------------------------------------------------------
# SELF-LISTENING (auto-ascolto): il modello sente il PROPRIO parlato in input.
# La sequenza mel in ingresso interleava per ogni frame agente (80ms):
#   [mel UTENTE x8, mel AGENTE x8, mel UTENTE x8, mel AGENTE x8, ...]
# Il tower processa il flusso unico (subsampling 4x -> 4 feature per frame:
# 2 canale utente + 2 canale agente) e l'attenzione causale del LLM dà a ogni
# frame agente il contesto [utente fino a t, proprio parlato fino a t-delay]
# (il mel agente del blocco t è il parlato del frame t-SELF_LISTENING_DELAY_FRAMES,
# zero-pad per i primi blocchi). Coerente con il design E2B e con l'evidenza
# community (dockhardman/gemma-4-E2B-duplex: "barge-in handled natively by the
# model"): il modello deve "sentirsi parlare" per sapere di essere stato
# interrotto e per cedere il turno in modo nativo.
SELF_LISTENING = True
MEL_CHANNELS = 2  # [utente, agente] per frame (1 = solo utente, legacy)
FEATURES_PER_AGENT_FRAME = (
    AGENT_FEATURES_PER_FRAME * MEL_CHANNELS
)  # 4 (2 utente + 2 agente)
SELF_LISTENING_DELAY_FRAMES = (
    1  # il mel agente del blocco t è il parlato del frame t-delay
)

# ---------------------------------------------------------------------------
# FRAME-LEVEL IMPORTANCE WEIGHTING (ricetta dockhardman): il segnale "act now"
# (apertura turno, barge-in, backchannel) è una frazione minuscola dei frame
# supervisionati e viene sommerso dai frame di silenzio (<wait>). Amplificando
# la loss sulle posizioni decisive, turn-taking e barge-in emergono con la sola
# SFT (finding verificato: "amplifying the training loss on those few decisive
# positions cures all three at once").
USE_FRAME_WEIGHTING = True
FRAME_WEIGHT_OPEN = 5.0  # frame di APERTURA turno (agente: silenzio -> parlato)
FRAME_WEIGHT_YIELD = (
    3.0  # frame di CESSIONE turno (agente smette mentre l'utente parla)
)
FRAME_WEIGHT_CO_TALK = (
    3.0  # frame di co-talk (parlato agente E utente: barge-in/backchannel)
)
AGENT_SILENCE_MATCH_FRAC = 0.90  # frazione di codebook al codice di silenzio per definire un frame agente "silenzio"
USER_ENERGY_BIAS = 2.0  # soglia di attività utente: p10(mel) + max(bias, 0.5*(p90-p10))

# Codici di silenzio EMPIRICI per codebook: il tokenizer 12Hz NON ha un token di
# silenzio esplicito (code 0 è un centroide RVQ reale; il padding è -1, verificato
# nel decode() di modeling_qwen3_tts_tokenizer_v2.py). Vengono estratti in Fase 1
# codificando una waveform di puro silenzio (moda per codebook) e persistiti nel
# sidecar JSON.
SILENCE_TOKEN_IDS: List[int] = [0] * N_CODEBOOKS
# Confini di segmento (fine conversazioni/turni in frame agente) ripristinati dal
# sidecar in Fase 2: usati per lo split di validazione a confine di conversazione.
SEGMENT_ENDS: List[int] = []

# Configurazione Subsetting Dati
DUPLEXCONV_MAX_HOURS = 50.0  # Ore di audio da caricare in streaming da DuplexConv
MDC_MAX_SAMPLES = 2500  # Campioni per il dataset MDC Italiano
BARGE_IN_PROBABILITY = 0.20  # 20% di probabilità di sovrapposizione reale (barge-in) per simulare l'interruzione

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Pipeline Inizializzata] Modello Backbone: {MODEL_NAME}")
print(f"[Pipeline Inizializzata] Device attivo: {DEVICE}")
sys.stdout.flush()

# Secrets (HF_TOKEN, MDC_API_KEY) passati via variabili d'ambiente — mai nel codice/git.
sys.stdout.flush()

# %% [markdown]
# ## ====================================================================
# ## FASE 1: PREPROCESSING & TOKENIZZAZIONE PARQUET (Celle 1 - 3)
# ## ====================================================================

# %% [code]
# Cell 1: Setup Ambiente Fase 1 (Tokenizzatore Qwen3-TTS + Mel numpy)
# Esegui questa cella su Kaggle se stai avviando la Fase 1 per la prima volta:
# !apt-get update -y && apt-get install -y sox libsox-fmt-all
# !pip install -q qwen-tts
# !pip install -q sox librosa datacollective soundfile pandas numpy pyarrow datasets tqdm transformers
#
# NOTA: qwen-tts (0.1.1) porta da sé la sua transformers compatibile (==4.57.3,
# verificato sul pyproject del pacchetto): niente pin manuale e niente --no-deps,
# pip risolve da solo la versione giusta. È SUFFICIENTE per la Fase 1: l'estrattore
# mel Gemma4 è replicato in numpy puro (load_mel_extractor -> _NumpyGemma4MelExtractor,
# bit-pari con il Gemma4AudioFeatureExtractor di transformers>=5). L'audio tower
# (Fase 2) usa transformers>=5.0 in kernel separato (ADR-003).
# L'encoder Qwen3-TTS richiede sr esplicito: encode_audio_with_qwen_tts passa
# sr=sampling_rate del file/dataset (fix "must provide sr").
# librosa è già una dipendenza di qwen-tts, ma viene installato esplicitamente
# per sicurezza (serve per resample 48k/24k -> 16k del mel).


def load_duplexconv_dataset(
    max_hours: float = 1.0, max_samples: Optional[int] = None
) -> Any:
    """Carica DuplexConv da Hugging Face in streaming.

    NOTA (schema verificato sul Hub): gli esempi del webdataset sono SEGMENTI
    stereo (campo "wav", sr=48000, ~5-30s l'uno, canale 0 = utente / canale 1 =
    agente), non dialoghi interi da 120s. `max_samples` è quindi il numero di
    segmenti: ~1500 segmenti (default 50h con euristica 120s) equivalgono a
    ~4-8h di audio reale.
    """
    try:
        from datasets import load_dataset

        print(
            f"[Fase 1 - Dataset] Connessione streaming a qualialabsAI/DuplexConv (target: {max_hours} ore)..."
        )
        sys.stdout.flush()
        ds = load_dataset("qualialabsAI/DuplexConv", split="train", streaming=True)
        if max_samples is None:
            max_samples = max(10, int((max_hours * 3600) / 120))
        ds_subset = ds.take(max_samples)
        print(
            f"[Fase 1 - Dataset] Subset DuplexConv configurato (~{max_samples} segmenti stereo in streaming)."
        )
        sys.stdout.flush()
        return ds_subset
    except Exception as e:
        print(
            f"[Fase 1 - Dataset] Avviso DuplexConv ({e}). Modalità mock attiva per test locali."
        )
        sys.stdout.flush()
        return None


def sync_tokenizer_hyperparams(tokenizer: Any) -> None:
    """Sovrascrive le costanti globali con gli iperparametri REALI del tokenizer Qwen3-TTS-12Hz.

    Fonte: config.json del repo HF (encoder_valid_num_quantizers / codebook_size /
    semantic_codebook_size). Se la sincronizzazione fallisce restano i default verificati.
    """
    global N_CODEBOOKS, SEMANTIC_VOCAB_SIZE, ACOUSTIC_VOCAB_SIZE
    try:
        cfg = getattr(tokenizer, "config", None)
        enc = getattr(cfg, "encoder_config", None)
        dec = getattr(cfg, "decoder_config", None)
        n_valid = getattr(cfg, "encoder_valid_num_quantizers", None)
        if n_valid is not None:
            N_CODEBOOKS = int(n_valid)
        if enc is not None and getattr(enc, "num_semantic_quantizers", 1) >= 1:
            sem_vocab = getattr(dec, "semantic_codebook_size", None)
            if sem_vocab is not None:
                SEMANTIC_VOCAB_SIZE = int(sem_vocab)
            acu_vocab = getattr(enc, "codebook_size", None)
            if acu_vocab is not None:
                ACOUSTIC_VOCAB_SIZE = int(acu_vocab)
        print(
            f"[Fase 1 - Tokenizer] Iperparametri sincronizzati dal config reale: "
            f"N_CODEBOOKS={N_CODEBOOKS}, SEMANTIC_VOCAB_SIZE={SEMANTIC_VOCAB_SIZE}, "
            f"ACOUSTIC_VOCAB_SIZE={ACOUSTIC_VOCAB_SIZE}."
        )
    except Exception as e:
        print(
            f"[Fase 1 - Tokenizer] Sincronizzazione iperparametri non riuscita ({e}). Default usati."
        )
    sys.stdout.flush()


def save_tokenizer_hyperparams(json_path: str, extra: Optional[dict] = None) -> None:
    """Persiste gli iperparametri del tokenizer accanto al parquet per la Fase 2 (kernel riavviato)."""
    import json

    payload = {
        "N_CODEBOOKS": N_CODEBOOKS,
        "SEMANTIC_VOCAB_SIZE": SEMANTIC_VOCAB_SIZE,
        "ACOUSTIC_VOCAB_SIZE": ACOUSTIC_VOCAB_SIZE,
        "REAL_ENCODING": TOKENIZER_REAL_ENCODING_ACTIVE,
        "SILENCE_TOKEN_IDS": [int(x) for x in SILENCE_TOKEN_IDS],
    }
    if extra:
        payload.update(extra)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Fase 1 - Parquet Export] Iperparametri tokenizer salvati in: {json_path}")
    sys.stdout.flush()


def load_tokenizer_hyperparams(json_path: str) -> dict:
    """Ripristina in Fase 2 gli iperparametri salvati in Fase 1. Ritorna il payload completo."""
    global \
        N_CODEBOOKS, \
        SEMANTIC_VOCAB_SIZE, \
        ACOUSTIC_VOCAB_SIZE, \
        TOKENIZER_REAL_ENCODING_ACTIVE, \
        SILENCE_TOKEN_IDS
    global \
        TEXT_BLOCK_LEN, \
        TEXT_LOSS_WEIGHT, \
        E2B_PAD_TOKEN_ID, \
        E2B_BOS_TOKEN_ID, \
        E2B_EOS_TOKEN_ID, \
        VAL_FRACTION, \
        SEGMENT_ENDS
    SEGMENT_ENDS = []
    try:
        import json

        with open(json_path) as f:
            d = json.load(f)
        N_CODEBOOKS = int(d["N_CODEBOOKS"])
        SEMANTIC_VOCAB_SIZE = int(d["SEMANTIC_VOCAB_SIZE"])
        ACOUSTIC_VOCAB_SIZE = int(d["ACOUSTIC_VOCAB_SIZE"])
        TOKENIZER_REAL_ENCODING_ACTIVE = bool(d.get("REAL_ENCODING", False))
        sids = d.get("SILENCE_TOKEN_IDS")
        if isinstance(sids, list) and len(sids) == N_CODEBOOKS:
            SILENCE_TOKEN_IDS = [int(x) for x in sids]
            print(
                f"[Fase 2 - DataLoad] Codici di silenzio empirici ripristinati: {SILENCE_TOKEN_IDS}"
            )
        # Super-supervisione testuale + validazione
        if "TEXT_BLOCK_LEN" in d:
            TEXT_BLOCK_LEN = int(d["TEXT_BLOCK_LEN"])
        if "TEXT_LOSS_WEIGHT" in d:
            TEXT_LOSS_WEIGHT = float(d["TEXT_LOSS_WEIGHT"])
        if "E2B_PAD_TOKEN_ID" in d:
            E2B_PAD_TOKEN_ID = int(d["E2B_PAD_TOKEN_ID"])
        if "E2B_BOS_TOKEN_ID" in d:
            E2B_BOS_TOKEN_ID = int(d["E2B_BOS_TOKEN_ID"])
        if "E2B_EOS_TOKEN_ID" in d:
            E2B_EOS_TOKEN_ID = int(d["E2B_EOS_TOKEN_ID"])
        if "VAL_FRACTION" in d:
            VAL_FRACTION = float(d["VAL_FRACTION"])
        se = d.get("SEGMENT_ENDS")
        if isinstance(se, list):
            SEGMENT_ENDS = [int(x) for x in se]
        print(
            f"[Fase 2 - DataLoad] Iperparametri tokenizer ripristinati: "
            f"N_CODEBOOKS={N_CODEBOOKS}, SEMANTIC_VOCAB_SIZE={SEMANTIC_VOCAB_SIZE}, "
            f"ACOUSTIC_VOCAB_SIZE={ACOUSTIC_VOCAB_SIZE}, TEXT_BLOCK_LEN={TEXT_BLOCK_LEN}, "
            f"SEGMENT_ENDS={len(SEGMENT_ENDS)}."
        )
        return d
    except Exception:
        print(
            f"[Fase 2 - DataLoad] Sidecar iperparametri non trovato ({json_path}); uso i default del modulo."
        )
        return {}
    finally:
        sys.stdout.flush()


def load_qwen_tts_tokenizer(model_name: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz") -> Any:
    """Carica il tokenizer acustico Qwen3-TTS neurale a 12.5 Hz (15 RVQ Codebooks)."""
    import shutil

    if shutil.which("sox") is None and sys.platform.startswith("linux"):
        print(
            "[Fase 1 - Tokenizer] SoX non trovato. Avvio installazione SoX su ambiente Linux..."
        )
        sys.stdout.flush()
        os.system(
            "apt-get update -y && apt-get install -y sox libsox-fmt-all > /dev/null 2>&1"
        )

    print(
        f"[Fase 1 - Tokenizer] Caricamento Qwen3-TTS Tokenizer neurale ({model_name})..."
    )
    sys.stdout.flush()
    try:
        from qwen_tts import Qwen3TTSTokenizer

        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            model_name, attn_implementation="eager"
        )
        if (
            torch.cuda.is_available()
            and hasattr(tokenizer, "model")
            and tokenizer.model is not None
        ):
            tokenizer.model = tokenizer.model.to("cuda")
            tokenizer.device = torch.device("cuda")
            print(
                "[Fase 1 - Tokenizer] Qwen3-TTS Tokenizer allocato con successo su GPU CUDA."
            )
        else:
            print("[Fase 1 - Tokenizer] Qwen3-TTS Tokenizer caricato su CPU.")
        sync_tokenizer_hyperparams(tokenizer)
        sys.stdout.flush()
        return tokenizer
    except Exception as e:
        print(
            f"[Fase 1 - Tokenizer] Avviso caricamento Qwen3-TTS Tokenizer ({e}). Attivo fallback spettrale FFT."
        )
        sys.stdout.flush()
        return None


def extract_code2wav_decoder_weights(
    qwen_tokenizer, out_dir: Optional[str] = None
) -> Optional[str]:
    """Estrae i pesi del decoder Code2Wav (prefissi `decoder.*`) dal tokenizer 12Hz
    e li salva standalone per la Fase 2 (dove transformers 5.x NON carica il decoder:
    `qwen3_tts_tokenizer_12hz` è registrato solo nella transformers 4.57.3 portata da
    qwen-tts, presente in Fase 1 e assente dopo il riavvio del kernel).

    Output: `code2wav_weights/code2wav_decoder.safetensors` + `config.json`, letti in
    Fase 2 da `Code2WavDecoder._init_decoder` via `code2wav_decoder.load_code2wav_decoder`
    (replica torch pura, bit-exact verificata vs il decoder ufficiale).
    """
    out_dir = out_dir or CODE2WAV_WEIGHTS_DIR
    model = getattr(qwen_tokenizer, "model", None)
    if model is None:
        raise RuntimeError(
            "tokenizer Qwen3-TTS senza `.model` (fallback FFT): impossibile estrarre il decoder."
        )
    sd = model.state_dict()
    dec = {
        k[len("decoder.") :]: v.detach().cpu()
        for k, v in sd.items()
        if k.startswith("decoder.")
    }
    if not dec:
        raise RuntimeError("nessun tensore `decoder.*` nello state_dict del tokenizer.")
    os.makedirs(out_dir, exist_ok=True)
    import safetensors.torch as st

    st_path = os.path.join(out_dir, "code2wav_decoder.safetensors")
    st.save_file(dec, st_path)

    # Config del decoder (replica pura: vedi DEFAULT_DECODER_CONFIG in code2wav_decoder.py)
    cfg = {}
    mcfg = getattr(model, "config", None)
    src = getattr(mcfg, "decoder_config", None) if mcfg is not None else None
    if src is None and mcfg is not None:
        src = getattr(mcfg, "decoder", None)
    if src is not None:
        if isinstance(src, dict):
            cfg.update(
                {
                    k: src.get(k)
                    for k in (
                        "num_quantizers",
                        "codebook_dim",
                        "codebook_size",
                        "latent_dim",
                        "hidden_size",
                        "num_hidden_layers",
                        "num_attention_heads",
                        "num_key_value_heads",
                        "head_dim",
                        "intermediate_size",
                        "hidden_act",
                        "rms_norm_eps",
                        "layer_scale_initial_scale",
                        "attention_bias",
                        "attention_dropout",
                        "sliding_window",
                        "max_position_embeddings",
                        "rope_theta",
                        "decoder_dim",
                        "upsample_rates",
                        "upsampling_ratios",
                    )
                    if src.get(k) is not None
                }
            )
        else:
            for k in (
                "num_quantizers",
                "codebook_dim",
                "codebook_size",
                "latent_dim",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "intermediate_size",
                "hidden_act",
                "rms_norm_eps",
                "layer_scale_initial_scale",
                "attention_bias",
                "attention_dropout",
                "sliding_window",
                "max_position_embeddings",
                "rope_theta",
                "decoder_dim",
                "upsample_rates",
                "upsampling_ratios",
            ):
                v = getattr(src, k, None)
                if v is not None:
                    cfg[k] = v
    cfg_path = os.path.join(out_dir, "code2wav_decoder_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(
        f"[Fase 1 - Code2Wav] ✅ {len(dec)} tensori decoder estratti in {st_path} "
        f"({os.path.getsize(st_path) / 1e6:.0f} MB) + config ({len(cfg)} chiavi)."
    )
    sys.stdout.flush()
    return out_dir


def _ensure_subtalker() -> bool:
    """Estrae il sub-talker Qwen3-TTS Base (ADR-005) in QWEN_SUBTALKER_DIR se assente.

    Obbligatorio per il coupling in Fase 2 (fail-fast in load_gemma4_duplex_system):
    scarica SOLO i ~363MB (range HTTP parziale) dalla repo GATED — serve il secret
    Kaggle HF_TOKEN. Ritorna True se la dir è pronta; non blocca mai la Fase 1.
    """
    marker = os.path.join(QWEN_SUBTALKER_DIR, "qwen_subtalker.safetensors")
    if os.path.isfile(marker):
        print(
            f"[Fase 1 - Coupling] Sub-talker già presente in {QWEN_SUBTALKER_DIR}: estrazione saltata."
        )
        sys.stdout.flush()
        return True
    print(
        "[Fase 1 - Coupling] Estrazione automatica sub-talker Qwen3-TTS Base (ADR-005) "
        f"in {QWEN_SUBTALKER_DIR} (~363MB, range parziale)..."
    )
    sys.stdout.flush()
    try:
        from extract_qwen_subtalker import extract_subtalker

        extract_subtalker(QWEN_SUBTALKER_DIR, big_disk_base())
        ok = os.path.isfile(marker)
        print(
            f"[Fase 1 - Coupling] {'✅ Sub-talker salvato' if ok else '⚠️ marker mancante'} "
            f"in {QWEN_SUBTALKER_DIR}: in Fase 2 il coupling è ATTIVO (obbligatorio)."
        )
        sys.stdout.flush()
        return ok
    except Exception as e:
        print(
            f"[Fase 1 - Coupling] ⚠️ Estrazione sub-talker fallita ({e}). "
            "Imposta il secret Kaggle HF_TOKEN (repo GATED) e riprova, oppure esegui "
            "`!python extract_qwen_subtalker.py --out-dir qwen_subtalker`. "
            "Senza di esso, in Fase 2 `load_gemma4_duplex_system` farà fail-fast."
        )
        sys.stdout.flush()
        return False


def load_e2b_text_tokenizer() -> Any:
    """Carica il tokenizer TESTO di Gemma 4 E2B (BPE, vocab 262144) senza dipendere da transformers>=5.

    In Fase 1 (transformers<5, portato da qwen-tts) il modello Gemma4 non è
    registrato: il tokenizer.json viene caricato direttamente con la libreria
    `tokenizers` (32MB, verificato sul repo google/gemma-4-E2B-it). In Fase 2 si
    potrebbe usare AutoTokenizer, ma i token vengono già salvati nel parquet in
    Fase 1: la Fase 2 non deve tokenizzare nulla.

    Ritorna un oggetto con `.encode(text) -> ids` (lista int) e `.tokenizer_id`.
    """
    tokenizer_json = os.path.join(os.getcwd(), "data", "e2b_tokenizer.json")
    try:
        from tokenizers import Tokenizer as FastTokenizer

        if not os.path.exists(tokenizer_json):
            print(f"[Fase 1 - Testo] Scarico tokenizer.json di Gemma 4 E2B (~32MB)...")
            sys.stdout.flush()
            try:
                from huggingface_hub import hf_hub_download

                tokenizer_json = hf_hub_download(
                    "google/gemma-4-E2B-it",
                    "tokenizer.json",
                    repo_type="model",
                    local_dir=os.path.dirname(tokenizer_json),
                )
            except Exception:
                import urllib.request

                os.makedirs(os.path.dirname(tokenizer_json) or ".", exist_ok=True)
                urllib.request.urlretrieve(
                    "https://huggingface.co/google/gemma-4-E2B-it/resolve/main/tokenizer.json",
                    tokenizer_json,
                )
        tok = FastTokenizer.from_file(tokenizer_json)

        def encode_ids(text: str, max_len: int = TEXT_BLOCK_LEN) -> List[int]:
            ids = tok.encode(str(text)).ids
            # Il post-processor può aggiungere bos/eos: il bos globale è già a
            # posizione 0 della sequenza LLM, l'eos non serve nei target.
            if ids and ids[0] == E2B_BOS_TOKEN_ID:
                ids = ids[1:]
            if ids and ids[-1] == E2B_EOS_TOKEN_ID:
                ids = ids[:-1]
            return ids[:max_len]

        encode_ids.tokenizer_id = "google/gemma-4-E2B-it"
        print(
            "[Fase 1 - Testo] Tokenizer testo Gemma 4 E2B caricato (vocab 262144, BPE)."
        )
        sys.stdout.flush()
        return encode_ids
    except Exception as e:
        print(
            f"[Fase 1 - Testo] ⚠️ Tokenizer testo non disponibile ({e}): training SOLO audio (testo disattivato)."
        )
        sys.stdout.flush()
        return None


def encode_audio_with_qwen_tts(
    tokenizer: Any, waveform: np.ndarray, sr: int = 24000
) -> Optional[np.ndarray]:
    """Encoder neurale che converte una forma d'onda audio nei N_CODEBOOKS RVQ.

    Ritorna None (e NON un fallback silenzioso) quando il tokenizer reale funziona ma la
    shape dei codici non combacia con N_CODEBOOKS: in quel caso gli iperparametri sono
    sbagliati e il training su dati FFT sarebbe degenerato (loss collassata ~0).
    """
    global TOKENIZER_REAL_ENCODING_ACTIVE
    if tokenizer is not None and len(waveform) > 0:
        try:
            with torch.no_grad():
                if hasattr(tokenizer, "encode"):
                    # qwen_tts.Qwen3TTSTokenizer.encode(audios, sr=...): per input numpy
                    # (list/array) sr è OBBLIGATORIO, altrimenti ValueError "For numpy
                    # waveform input, you must provide sr". Ritorna
                    # Qwen3TTSTokenizerV2EncoderOutput con .audio_codes =
                    # List[torch.LongTensor (T, num_quantizers)] per ogni audio.
                    enc = tokenizer.encode(waveform, sr=sr)
                    if hasattr(enc, "audio_codes"):
                        codes = enc.audio_codes
                        if isinstance(codes, (list, tuple)) and len(codes) == 1:
                            codes = codes[0]
                    else:
                        codes = enc
                else:
                    audio_tensor = torch.tensor(
                        waveform, dtype=torch.float32
                    ).unsqueeze(0)
                    if torch.cuda.is_available():
                        audio_tensor = audio_tensor.to("cuda")
                    codes = tokenizer(audio_tensor)

                if isinstance(codes, torch.Tensor):
                    codes = codes.cpu().numpy()
                elif (
                    isinstance(codes, (tuple, list))
                    and codes
                    and isinstance(codes[0], torch.Tensor)
                ):
                    # output raw (return_dict=False): primo elemento = codici
                    codes = codes[0].cpu().numpy()
                if isinstance(codes, np.ndarray):
                    if codes.ndim == 3:
                        codes = codes.squeeze(0)
                    if (
                        codes.ndim == 2
                        and codes.shape[0] != N_CODEBOOKS
                        and codes.shape[1] == N_CODEBOOKS
                    ):
                        codes = codes.T
                    if codes.ndim == 2 and codes.shape[0] == N_CODEBOOKS:
                        TOKENIZER_REAL_ENCODING_ACTIVE = True
                        return codes.astype(np.int16)
                    print(
                        f"[Fase 1 - Encoding] FORMATO CODICI INATTESO: shape={codes.shape}, "
                        f"N_CODEBOOKS={N_CODEBOOKS}. Controlla gli iperparametri del tokenizer "
                        f"(config HF: 16 codebook, semantico 4096, acustici 2048). Nessun fallback FFT."
                    )
                    sys.stdout.flush()
                    return None
        except Exception as e:
            print(
                f"[Fase 1 - Encoding] Avviso encoding Qwen3-TTS: {e}. Fallback su FFT."
            )
            sys.stdout.flush()

    # Fallback FFT spettrale SOLO se l'encoder neurale è davvero indisponibile.
    if not TOKENIZER_REAL_ENCODING_ACTIVE:
        print(
            "[Fase 1 - Encoding] ⚠️ ATTENZIONE: encoder neurale non disponibile. "
            "Fallback FFT attivo: i dati saranno DEGENERATI (non adatti a training reale)."
        )
        sys.stdout.flush()
    samples_per_frame = int(sr / FRAME_RATE_HZ)
    num_frames = max(1, len(waveform) // samples_per_frame)
    tokens = np.zeros((N_CODEBOOKS, num_frames), dtype=np.int16)
    for f in range(num_frames):
        chunk = waveform[f * samples_per_frame : (f + 1) * samples_per_frame]
        rms = np.sqrt(np.mean(chunk**2) + 1e-9)
        if rms < 0.005:
            tokens[:, f] = SILENCE_TOKEN_ID
        else:
            fft_mag = np.abs(np.fft.rfft(chunk))
            bands = np.array_split(fft_mag, N_CODEBOOKS)
            for k in range(N_CODEBOOKS):
                band_energy = float(np.mean(bands[k])) if len(bands[k]) > 0 else rms
                tokens[k, f] = (
                    int((np.log1p(band_energy * 1000) * 137 + (k + 1) * 31) % 1023) + 1
                )
    return tokens


def extract_silence_codes(
    qwen_tokenizer: Any, sr: int = 24000, seconds: float = 2.0
) -> np.ndarray:
    """Estrae automaticamente i codici di silenzio per codebook (moda per codebook).

    Il tokenizer Qwen3-TTS 12Hz NON definisce un token di silenzio esplicito:
    code 0 è un centroide reale del RVQ e il padding è -1 (verificato nel decode()
    di modeling_qwen3_tts_tokenizer_v2.py: clamp(min=0) + audio_codes[...,0] > -1).
    Codifichiamo quindi una waveform di puro silenzio e prendiamo il codice DOMINANTE
    emesso per ogni codebook: è la rappresentazione empirica del silenzio di questo codec.
    """
    silence = np.zeros(int(sr * seconds), dtype=np.float32)
    codes = encode_audio_with_qwen_tts(qwen_tokenizer, silence, sr=sr)
    if codes is None:
        print(
            "[Fase 1 - Silenzio] Encoder non disponibile: uso il codice 0 come silenzio (fallback)."
        )
        sys.stdout.flush()
        return np.zeros(N_CODEBOOKS, dtype=np.int16)
    per_cb = []
    for k in range(N_CODEBOOKS):
        row = codes[k]
        vals, counts = np.unique(row, return_counts=True)
        per_cb.append(int(vals[counts.argmax()]))
    print(
        f"[Fase 1 - Silenzio] Codici di silenzio EMPIRICI estratti per {N_CODEBOOKS} codebook: {per_cb}"
    )
    sys.stdout.flush()
    return np.array(per_cb, dtype=np.int16)


class _NumpyGemma4MelExtractor:
    """Replica pura numpy del Gemma4AudioFeatureExtractor ufficiale (transformers 5.x).

    Config esatta letta da processor_config.json di google/gemma-4-E2B-it:
      sampling_rate=16000, fft_length=512, frame_length=320, hop_length=160,
      feature_size=128, min_frequency=0, max_frequency=8000, mel_floor=0.001,
      nessuna normalizzazione (per_bin_mean/stddev=null), preemphasis=0, dither=0.

    Algoritmo (feature_extraction_gemma4.py, Apache-2.0):
      1. padding semicaUsale: prepend frame_length//2 = 160 zeri (primo frame centrato a t=0)
      2. unfold finestre di 321 campioni con passo 160; si scarta l'ultimo campione -> 320
      3. window Hann periodica 320; rfft con n=512
      4. bank mel HTK (257 -> 128); log(mel + mel_floor)

    Funziona identico con transformers 4.49 (Fase 1 su Kaggle, dove il Gemma4
    feature extractor non esiste) e con qualsiasi versione >= 5: stesso output bit-pari.
    """

    def __init__(self):
        self.sampling_rate = MEL_SAMPLING_RATE
        self.feature_size = MEL_BINS
        self.frame_length = 320
        self.hop_length = 160
        self.fft_length = 512
        self.mel_floor = 1e-3
        # Window Hann periodica (come window_function(hann, periodic=True))
        self.window = (
            0.5
            - 0.5
            * np.cos(2.0 * np.pi * np.arange(self.frame_length) / self.frame_length)
        ).astype(np.float32)
        self.mel_filters = self._mel_filter_bank(
            num_frequency_bins=self.fft_length // 2 + 1,
            num_mel_filters=self.feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=self.sampling_rate,
        ).astype(np.float32)

    # --- repliche di transformers.audio_utils (HTK mel scale) ---
    @staticmethod
    def _hertz_to_mel(freq: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + (freq / 700.0))

    @staticmethod
    def _mel_to_hertz(mels: np.ndarray) -> np.ndarray:
        return 700.0 * (np.power(10.0, mels / 2595.0) - 1.0)

    @staticmethod
    def _triangular_filter_bank(
        fft_freqs: np.ndarray, filter_freqs: np.ndarray
    ) -> np.ndarray:
        filter_diff = np.diff(filter_freqs)
        slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
        down_slopes = -slopes[:, :-2] / filter_diff[:-1]
        up_slopes = slopes[:, 2:] / filter_diff[1:]
        return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))

    @classmethod
    def _mel_filter_bank(
        cls,
        num_frequency_bins: int,
        num_mel_filters: int,
        min_frequency: float,
        max_frequency: float,
        sampling_rate: int,
    ) -> np.ndarray:
        mel_min = cls._hertz_to_mel(min_frequency)
        mel_max = cls._hertz_to_mel(max_frequency)
        mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
        filter_freqs = cls._mel_to_hertz(mel_freqs)
        fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)
        return cls._triangular_filter_bank(fft_freqs, filter_freqs)

    @staticmethod
    def _unfold(array: np.ndarray, size: int, step: int) -> np.ndarray:
        """Equivalente numpy di torch unfold sull'ultima dimensione (2D)."""
        batch_size, original_length = array.shape
        num_frames = (original_length - size) // step + 1
        if num_frames <= 0:
            return np.zeros((batch_size, 0, size), dtype=array.dtype)
        output_shape = (batch_size, num_frames, size)
        output_strides = (array.strides[0], array.strides[1] * step, array.strides[1])
        return np.lib.stride_tricks.as_strided(
            array, shape=output_shape, strides=output_strides
        )

    def _extract_spectrogram(
        self, waveform: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """waveform: (1, T) mono -> mel (nf, MEL_BINS) float32 + mask bool (nf,)."""
        pad_left = self.frame_length // 2
        waveform = np.pad(waveform, ((0, 0), (pad_left, 0)), mode="constant")
        frames = self._unfold(
            waveform, size=self.frame_length + 1, step=self.hop_length
        )[..., :-1]
        frames = frames * self.window
        stft = np.fft.rfft(frames, n=self.fft_length, axis=-1)
        magnitude = np.abs(stft)
        log_mel = np.log(magnitude @ self.mel_filters + self.mel_floor)  # (1, nf, 128)
        mel = log_mel.squeeze(0).astype(np.float32)  # (nf, 128)
        # Mask frame-aware: ogni frame è valido se la sua finestra ricade nell'audio reale
        # (con il solo padding semicaUsale iniziale tutti i frame sono validi).
        frame_end = (
            np.arange(mel.shape[0]) * self.hop_length + self.frame_length + 1 - 1
        )
        mask = frame_end < waveform.shape[1]
        return mel, mask

    def __call__(
        self,
        raw_speech,
        padding=False,
        truncation=False,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors=None,
        **kwargs,
    ):
        if isinstance(raw_speech, np.ndarray) and raw_speech.ndim == 1:
            waves = [raw_speech]
        elif isinstance(raw_speech, np.ndarray) and raw_speech.ndim == 2:
            waves = [raw_speech[i] for i in range(raw_speech.shape[0])]
        else:
            waves = [np.asarray(w) for w in raw_speech]
        mels, masks = [], []
        for w in waves:
            if w.ndim > 1:
                w = w.mean(axis=-1)
            mel, mask = self._extract_spectrogram(w[None, :])
            mels.append(mel)
            masks.append(mask)
        return {"input_features": mels, "input_features_mask": masks}


def load_mel_extractor(model_name: str = MODEL_NAME) -> Any:
    """Carica l'estrattore mel Gemma4: prima il path transformers (>= 5.x), poi numpy puro.

    Il Gemma4AudioFeatureExtractor (16kHz, 128 mel bin HTK, 100fps, nessuna
    normalizzazione) non esiste in transformers < 5.x (Fase 1 su Kaggle usa la
    transformers legacy portata da qwen-tts, 4.x): in quel caso si usa la replica
    numpy auto-contenuta, bit-pari con l'originale.
    """
    try:
        from transformers import AutoFeatureExtractor

        fe = AutoFeatureExtractor.from_pretrained(model_name)
        print(
            f"[Fase 1 - Mel] FeatureExtractor Gemma4 caricato: sampling_rate={getattr(fe, 'sampling_rate', '?')}, "
            f"feature_size={getattr(fe, 'feature_size', '?')}, hop={getattr(fe, 'hop_length', '?')}."
        )
        sys.stdout.flush()
        return fe
    except Exception as e:
        print(
            f"[Fase 1 - Mel] FeatureExtractor transformers non disponibile ({str(e)[:80]})."
        )
        sys.stdout.flush()

    print(
        "[Fase 1 - Mel] Uso l'estrattore mel NUMPY auto-contenuto (replica esatta "
        "Gemma4AudioFeatureExtractor: 16k, fft 512, window 320, hop 160, 128 mel HTK, "
        "mel_floor 1e-3, nessuna normalizzazione). Funziona anche con transformers 4.49."
    )
    sys.stdout.flush()
    return _NumpyGemma4MelExtractor()


def extract_mel_features(
    mel_extractor: Any, waveform: np.ndarray, sr: int
) -> Optional[np.ndarray]:
    """Converte la waveform del canale in mel log-spettrale (T, MEL_BINS) float16 @100fps."""
    if mel_extractor is None or len(waveform) == 0:
        return None
    try:
        if sr != MEL_SAMPLING_RATE:
            from librosa import resample

            waveform = resample(
                waveform.astype(np.float32),
                orig_sr=int(sr),
                target_sr=MEL_SAMPLING_RATE,
            )
        # Input SEMPRE batched ([w]): il Gemma4AudioFeatureExtractor ufficiale richiede
        # input batched (con 1D lancia "cannot select an axis to squeeze out").
        feats = mel_extractor(
            [waveform.astype(np.float32)],
            padding=False,
            truncation=False,
            max_length=None,
            pad_to_multiple_of=None,
            return_tensors="np",
        )
        mel = np.asarray(feats["input_features"][0], dtype=np.float16)  # (T, MEL_BINS)
        return mel
    except Exception as e:
        print(f"[Fase 1 - Mel] Avviso estrazione mel: {e}")
        sys.stdout.flush()
        return None


def align_mel_to_codes(mel: np.ndarray, n_agent: int) -> np.ndarray:
    """Allinea il mel (100fps) ai frame agente (12.5fps): 8 frame mel per frame agente.

    Tronca o zero-pada (silenzio) in coda per garantire l'allineamento esatto 8:1.
    """
    need = n_agent * MEL_PER_AGENT_FRAME
    if mel.shape[0] >= need:
        return mel[:need]
    pad = np.zeros((need - mel.shape[0], mel.shape[1]), dtype=mel.dtype)
    return np.concatenate([mel, pad], axis=0)


# Diventa True non appena l'encoder neurale reale produce codici validi (persistito per Fase 2).
TOKENIZER_REAL_ENCODING_ACTIVE = False


def _reduce_channels(arr_tc: np.ndarray) -> Optional[np.ndarray]:
    """Riduce una waveform (T, C) alla coppia stereo (T, 2) o None.

    Alcuni segmenti DuplexConv hanno 3 canali (probabile canale duplicato o layout
    esteso): teniamo la coppia MENO correlata (i veri canali utente/agente), o le
    prime due se tutti i canali sono quasi identici (mono duplicato). Mono reale o
    >3 canali distinti -> None (non mappabile al duplex stereo).
    """
    if arr_tc is None or arr_tc.ndim != 2:
        return None
    c = arr_tc.shape[1]
    if c == 2:
        return arr_tc
    if c == 1:
        return None
    if c == 3:
        seg = arr_tc[: min(200000, arr_tc.shape[0])]
        if seg.shape[0] < 4:
            return None
        ch0, ch1, ch2 = seg[:, 0], seg[:, 1], seg[:, 2]
        r01 = abs(float(np.corrcoef(ch0, ch1)[0, 1]))
        r02 = abs(float(np.corrcoef(ch0, ch2)[0, 1]))
        r12 = abs(float(np.corrcoef(ch1, ch2)[0, 1]))
        if r01 > 0.99 and r02 > 0.99 and r12 > 0.99:
            return arr_tc[:, [0, 1]]  # mono duplicato su tutti i canali
        _, i, j = min([(r01, 0, 1), (r02, 0, 2), (r12, 1, 2)])
        return arr_tc[:, [i, j]]
    return None


def _decode_duplex_audio(raw_audio: Any) -> Optional[Tuple[np.ndarray, int]]:
    """Decodifica il campo audio del webdataset DuplexConv in (waveform (T,2) float32, sr).

    Il formato effettivo del campo `wav` DIPENDE dalla versione di `datasets` su Kaggle
    (verificato sul campo reale: esempi grezzi = {'path': ..., 'bytes': b'RIFF...'}, ma
    con datasets >= 5.x l'iterazione yielda oggetti `torchcodec.decoders.AudioDecoder`):
      - torchcodec AudioDecoder -> `get_all_samples()`: tensor (T, C) + sample_rate
        (NON usare la __getitem__ del wrapper datasets: fa np.mean sugli assi != ultimo,
        cioè MISCELA i canali stereo in un vettore (2,)!)
      - feature Audio già decodificata -> {'array': np.ndarray (T,2), 'sampling_rate': 48000}
      - esempio grezzo -> {'path': ..., 'bytes': b'RIFF...'} (decodifica manuale via soundfile)
      - solo path -> tentativo di lettura diretta
    Gestiamo tutte le forme per non dipendere dalla versione di datasets su Kaggle.
    """
    if isinstance(raw_audio, (list, tuple)) and raw_audio:
        raw_audio = raw_audio[0]

    # 0) torchcodec / datasets AudioDecoder: get_all_samples() -> AudioSamples(data, sample_rate)
    #    ATTENZIONE alla convenzione di shape: torchcodec restituisce data con i CANALI
    #    per primi, (C, T) (num_samples = data.shape[1], duration = data.shape[1]/sr).
    #    NIENTE __getitem__ del wrapper datasets (fa np.mean sugli assi != ultimo,
    #    cioè downmix mono dei canali).
    getter = getattr(raw_audio, "get_all_samples", None)
    if callable(getter):
        try:
            s = getter()
            sr = getattr(s, "sample_rate", None)
            if sr is None:
                md = getattr(raw_audio, "metadata", None) or {}
                sr = md.get("sampling_rate")
            arr = np.asarray(s.data.cpu().numpy())
            if arr.ndim == 2 and arr.shape[0] in (1, 2, 3) and sr:
                arr = _reduce_channels(arr.T)  # (C, T) -> (T, C') con C'==2 se valido
                if arr is not None:
                    return arr.astype(np.float32), int(sr)
                print(
                    f"[Fase 1 - Decode] AudioDecoder: canali non riducibili a stereo "
                    f"(data={tuple(np.asarray(s.data.cpu().numpy()).shape)}, sr={sr})."
                )
                sys.stdout.flush()
            if arr.ndim > 0:
                print(
                    f"[Fase 1 - Decode] AudioDecoder: shape inattesa data={arr.shape}, "
                    f"sr={sr}. Attesa (C, T) con C in (1, 2, 3)."
                )
                sys.stdout.flush()
        except Exception as e:
            print(f"[Fase 1 - Decode] Avviso AudioDecoder: {e}")
            sys.stdout.flush()

    if isinstance(raw_audio, dict):
        try:
            arr = raw_audio.get("array")
            sr = raw_audio.get("sampling_rate")
            if arr is not None:
                arr = _reduce_channels(np.asarray(arr))
                if arr is not None:
                    return arr.astype(np.float32), int(sr or 48000)
                return None
            b = raw_audio.get("bytes")
            if b is not None:
                import io

                import soundfile as sf

                data, fs = sf.read(io.BytesIO(b))
                data = _reduce_channels(data)
                if data is not None:
                    return data.astype(np.float32), int(fs)
                return None
            p = raw_audio.get("path")
            if p:
                import soundfile as sf

                data, fs = sf.read(p)
                data = _reduce_channels(data)
                if data is not None:
                    return data.astype(np.float32), int(fs)
        except Exception as e:
            print(f"[Fase 1 - Decode] Avviso decodifica audio DuplexConv: {e}")
            sys.stdout.flush()
        return None

    # 1) attributi bytes/path su oggetti generici (altri decoder)
    for attr in ("bytes", "path"):
        v = getattr(raw_audio, attr, None)
        if v is not None:
            try:
                import io

                import soundfile as sf

                data, fs = sf.read(io.BytesIO(v)) if attr == "bytes" else sf.read(v)
                data = _reduce_channels(data)
                if data is not None:
                    return data.astype(np.float32), int(fs)
            except Exception as e:
                print(f"[Fase 1 - Decode] Avviso attributo {attr}: {e}")
                sys.stdout.flush()
    return None


def load_duplex_annotations(download_dir: Optional[str] = None) -> Dict[str, str]:
    """Scarica ed estrae gli archivi JSON di annotazioni DuplexConv (asr per canale).

    Le trascrizioni NON sono nel webdataset audio (campo wav/__key__/__url__): stanno
    in archivi separati `{Edu,edu,none_Edu}/jsons.tar.gz` (~57+24+150 MB). Ogni JSON
    (es. `Edu--000001.json`) ha `asr[track][utterance]` con `labels.txt` (trascrizione),
    `startInMs`/`endInMs`, `state` (<|complete|>/<|incomplete|>). L'estrazione va sul
    DISCO GRANDE: ~230MB tar + ~1GB estratti.

    Ritorna la mappa key -> path del file JSON estratto (i file restano su disco;
    l'archivio .tar.gz viene rimosso per liberare spazio).
    """
    import json as _json
    import tarfile
    import urllib.request

    if download_dir is None:
        download_dir = os.path.join(big_disk_base(), "duplex_jsons")
    os.makedirs(download_dir, exist_ok=True)
    index: Dict[str, str] = {}
    for domain in ("Edu", "edu", "none_Edu"):
        url = f"https://huggingface.co/datasets/qualialabsAI/DuplexConv/resolve/main/{domain}/jsons.tar.gz?download=true"
        tarpath = os.path.join(download_dir, f"{domain}_jsons.tar.gz")
        outdir = os.path.join(download_dir, domain)
        try:
            if not os.path.exists(tarpath):
                print(f"[Fase 1 - Annotazioni] Scarico {domain}/jsons.tar.gz ...")
                sys.stdout.flush()
                urllib.request.urlretrieve(url, tarpath)
            if not os.path.exists(outdir):
                print(f"[Fase 1 - Annotazioni] Estrazione {domain}/jsons.tar.gz ...")
                sys.stdout.flush()
                os.makedirs(outdir, exist_ok=True)
                with tarfile.open(tarpath, "r:gz") as tar:
                    tar.extractall(path=outdir)
                os.remove(tarpath)  # libera disco
            for fn in os.listdir(outdir):
                if fn.endswith(".json"):
                    index[fn[:-5]] = os.path.join(outdir, fn)
        except Exception as e:
            print(f"[Fase 1 - Annotazioni] Avviso dominio {domain}: {e}")
            sys.stdout.flush()
    print(f"[Fase 1 - Annotazioni] Indice annotazioni pronto: {len(index)} segmenti.")
    sys.stdout.flush()
    return index


def _align_duplex_agent_text(
    json_path: str, n_agent_frames: int, seq_len: int, text_tokenizer: Any
) -> Optional[list]:
    """Allinea la trascrizione del canale AGENTE (asr[1]) alle finestre di seq_len frame.

    Per finestra [w*5.12s, (w+1)*5.12s) prende tutte le utterance del canale agente
    che la intersecano (labels.txt) e le tokenizza in BPE (truncate TEXT_BLOCK_LEN).
    Ritorna list len=n_windows di list[int] o None.
    """
    import json as _json

    try:
        with open(json_path, encoding="utf-8") as f:
            d = _json.load(f)
        asr = d.get("asr")
        if not isinstance(asr, list) or len(asr) < 2:
            return None
        utts = []
        for u in asr[1]:
            txt = ((u.get("labels") or {}).get("txt") or "").strip()
            if not txt:
                continue
            s = int(u.get("startInMs", 0)) / 1000.0
            e = int(u.get("endInMs", 0)) / 1000.0
            utts.append((s, e, txt))
        if not utts:
            return None
        n_windows = n_agent_frames // seq_len + (1 if n_agent_frames % seq_len else 0)
        res: list = []
        frame_s = 1.0 / FRAME_RATE_HZ
        for w in range(n_windows):
            ws = w * seq_len * frame_s
            we = (w + 1) * seq_len * frame_s
            parts = [txt for (s, e, txt) in utts if not (e <= ws or s >= we)]
            if not parts:
                res.append(None)
                continue
            joined = " ".join(parts)
            ids = text_tokenizer(joined) if text_tokenizer is not None else None
            res.append(ids)
        return res
    except Exception as e:
        print(f"[Fase 1 - Annotazioni] Avviso allineamento testo {json_path}: {e}")
        sys.stdout.flush()
        return None


def tokenize_duplexconv_subset(
    duplexconv_subset,
    seq_len: int = 64,
    qwen_tokenizer: Any = None,
    mel_extractor: Any = None,
    annotations: Optional[Dict[str, str]] = None,
    text_tokenizer: Any = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    """Tokenizza i dialoghi DuplexConv in (user_mel, agent_codes, text_by_window, segment_ends).

    Ritorna:
      - user_mel: mel log-spettrale del canale UTENTE (T_mel, MEL_BINS) float16 @100fps
                  (input del percorso NATIVO: audio tower Conformer di Gemma 4 E2B)
      - agent_mel: mel log-spettrale del canale AGENTE (T_mel, MEL_BINS) float16 @100fps
                   (auto-ascolto: il modello sente il proprio parlato in input)
      - agent_codes: codici RVQ Qwen3-TTS del canale AGENTE (N_CODEBOOKS, T_agent) int16 @12.5fps
      - text_by_window: list per finestra di seq_len frame con i token BPE della
                        trascrizione AGENTE (asr[1].labels.txt, allineata ai timestamp
                        millisecondo) oppure None. Le trascrizioni stanno negli archivi
                        jsons.tar.gz separati (vedi load_duplex_annotations).
      - segment_ends: lista di indici di frame agente della FINE di ogni segmento accettato
                      (confini di conversazione per lo split di validazione)
    """
    user_mel_blocks, agent_mel_blocks, agent_blocks = [], [], []
    segment_ends: list = []
    text_by_window_all: list = []
    if duplexconv_subset is not None:
        print(
            "[Fase 1 - Tokenizzazione] Avvio tokenizzazione RVQ agente + mel utente/agente del dataset DuplexConv..."
        )
        sys.stdout.flush()

        pbar = tqdm(duplexconv_subset, desc="Tokenizzazione DuplexConv", unit="dialogo")
        count = 0
        debug_discard_shown = False
        for item in pbar:
            count += 1
            u_mel, a_mel, a_codes = None, None, None
            # Lo schema del webdataset DuplexConv (verificato sul Hub): campo "wav"
            # stereo (T, 2) 48000 Hz. Canale 0 = utente (input mel), canale 1 =
            # agente (target codici + mel auto-ascolto). Il formato esatto del campo
            # dipende dalla versione di datasets (feature decodificata o bytes grezzi)
            # -> decode robusto multi-forma (vedi _decode_duplex_audio).
            raw_audio = item.get("wav") or item.get("audio") or item.get("user_audio")
            decoded = _decode_duplex_audio(raw_audio)
            if decoded is not None:
                arr, sr = decoded
                # soundfile -> (T, canali): il canale è sull'asse 1
                u_mel = extract_mel_features(mel_extractor, arr[:, 0], sr=sr)
                a_raw = encode_audio_with_qwen_tts(qwen_tokenizer, arr[:, 1], sr=sr)
                a_mel = extract_mel_features(mel_extractor, arr[:, 1], sr=sr)
                if u_mel is not None and a_mel is not None and a_raw is not None:
                    n_agent = a_raw.shape[1]
                    u_mel = align_mel_to_codes(u_mel, n_agent)
                    a_mel = align_mel_to_codes(a_mel, n_agent)
                    a_codes = a_raw[:, :n_agent]

            if a_codes is None:
                if not debug_discard_shown:
                    # Diagnostica del primo scarto: mostra la struttura REALE del campo audio
                    dbg = {}
                    for k in ("wav", "audio", "user_audio"):
                        v = item.get(k)
                        dbg[k] = (
                            type(v).__name__,
                            (list(v.keys()) if isinstance(v, dict) else ""),
                        )
                    print(
                        f"[Fase 1 - Tokenizzazione] Struttura audio reale primo scarto: {dbg}"
                    )
                    sys.stdout.flush()
                    debug_discard_shown = True
                print(
                    f"[Fase 1 - Tokenizzazione] Dialogo {count} scartato (mel o codici non validi)."
                )
                sys.stdout.flush()
                continue

            user_mel_blocks.append(u_mel)
            agent_mel_blocks.append(a_mel)
            agent_blocks.append(a_codes)
            segment_ends.append(
                int(a_codes.shape[1])
            )  # fine del segmento in frame agente

            # ---- Testo risposta agente (trascrizioni asr[1] dagli archivi JSON) ----
            seg_texts = None
            if annotations:
                key = item.get("__key__")
                jpath = annotations.get(str(key)) if key is not None else None
                if jpath is not None:
                    seg_texts = _align_duplex_agent_text(
                        jpath, int(a_codes.shape[1]), seq_len, text_tokenizer
                    )
            n_win = int(a_codes.shape[1]) // seq_len + (
                1 if int(a_codes.shape[1]) % seq_len else 0
            )
            if seg_texts is not None and len(seg_texts) == n_win:
                text_by_window_all.extend(seg_texts)
            else:
                text_by_window_all.extend([None] * n_win)
            pbar.set_postfix(
                {"Frame Mel": u_mel.shape[0], "Frame Agente": a_codes.shape[1]}
            )

        print(
            f"[Fase 1 - Tokenizzazione] DuplexConv tokenizzato con successo: {count} conversazioni."
        )
        sys.stdout.flush()

    if not user_mel_blocks:
        print(
            "[Fase 1 - Tokenizzazione] DuplexConv assente. Generazione dataset sintetico di test..."
        )
        sys.stdout.flush()
        total_frames = 1500 * seq_len
        a_stacked = np.tile(
            (np.arange(total_frames) % 15 + 1).astype(np.int16), (N_CODEBOOKS, 1)
        )
        # mel sintetico: zeri (silenzio) allineato 8:1 ai codici
        u_stacked = np.zeros(
            (total_frames * MEL_PER_AGENT_FRAME, MEL_BINS), dtype=np.float16
        )
        a_mel_stacked = np.zeros(
            (total_frames * MEL_PER_AGENT_FRAME, MEL_BINS), dtype=np.float16
        )
        return u_stacked, a_mel_stacked, a_stacked, [None] * 1500, []

    # Concatenazione: mel per-frame (T, MEL_BINS) vstack, codici per-frame (N_CB, T) hstack.
    # I segment_ends sono CUMULATI sull'unione (indici di frame assoluti nel parquet finale).
    cum = 0
    cum_ends = []
    for e in segment_ends:
        cum += int(e)
        cum_ends.append(cum)
    return (
        np.vstack(user_mel_blocks).astype(np.float16),
        np.vstack(agent_mel_blocks).astype(np.float16),
        np.hstack(agent_blocks).astype(np.int16),
        text_by_window_all,
        cum_ends,
    )


def load_mozilla_italian_dataset(
    dataset_id: str = "cmqini14100vmnq07309ocknr", max_samples: int = 2500
) -> pd.DataFrame:
    """Carica Common Voice Scripted Speech 26.0 - Italiano (Mozilla Data Collective).

    Schema reale (verificato dal registry MDC + API): dataset tar.gz di ~10GB con
    clip mp3 MONO a 32kHz in cv-corpus-26.0-2026-06-12/it/clips/. Il loader
    rinomina le colonne: path -> audio_path (path ASSOLUTO del clip), sentence ->
    transcription, client_id -> speaker_id, e aggiunge la colonna split
    (dev/invalidated/other/test/train/validated).

    Dopo il load l'archivio tar.gz (~10GB) viene ELIMINATO per non saturare il
    disco di Kaggle: servono solo i clip già estratti. Gli split di bassa qualità
    (invalidated/other) vengono esclusi.
    """
    try:
        from datacollective import load_dataset

        print(
            f"[Fase 1 - MDC] Caricamento dataset Mozilla Data Collective Italiano ({dataset_id})..."
        )
        sys.stdout.flush()
        # Scarica sul DISCO GRANDE, NON nella working dir del progetto:
        # tar ~10GB + clip estratti ~10GB non stanno nella quota working.
        mdc_dir = os.path.join(big_disk_base(), "mdc_download")
        df = load_dataset(dataset_id, download_directory=mdc_dir, show_progress=True)
        # Pulisce l'archivio (10GB) dopo l'estrazione: restano solo i clip mp3 estratti
        try:
            import glob as _glob

            for _f in _glob.glob(os.path.join(mdc_dir, "*.tar.gz")) + _glob.glob(
                os.path.join(mdc_dir, "*.zip")
            ):
                _sz = os.path.getsize(_f) / 1e9
                os.remove(_f)
                print(
                    f"[Fase 1 - MDC] Archivio rimosso per liberare spazio: {os.path.basename(_f)} ({_sz:.1f} GB)"
                )
        except Exception as e:
            print(f"[Fase 1 - MDC] Avviso pulizia archivio ({e}).")
        # Normalizzazione colonne: lo schema MDC rinomina (audio_path/transcription/speaker_id)
        df = df.rename(
            columns={
                "audio_path": "path",
                "transcription": "sentence",
                "speaker_id": "client_id",
            }
        )
        # Esclude gli split di bassa qualità (clip scartati dalla community)
        if "split" in df.columns:
            _before = len(df)
            df = df[~df["split"].isin(["invalidated", "other"])].copy()
            print(
                f"[Fase 1 - MDC] Split di bassa qualità esclusi: {_before} -> {len(df)} clip validi."
            )
        if len(df) > max_samples:
            df = df.head(max_samples)
        print(
            f"[Fase 1 - MDC] Caricati correttamente {len(df)} clip dal dataset MDC Italiano (mp3 mono 32kHz)."
        )
        sys.stdout.flush()
        return df
    except Exception as e:
        print(
            f"[Fase 1 - MDC] Avviso caricamento MDC ({e}). Modalità mock attiva per test locali."
        )
        sys.stdout.flush()
        return pd.DataFrame(
            {
                "client_id": ["user_1", "user_2"],
                "path": ["audio_sample_1.wav", "audio_sample_2.wav"],
                "sentence": [
                    "Buongiorno, vorrei informazioni.",
                    "Certamente! Quale argomento interessa?",
                ],
            }
        )


def create_dual_channel_dialogue(
    df: pd.DataFrame,
    output_wav_path: str,
    sample_rate: int = 24000,
    silence_sec: float = 0.4,
    barge_in_prob: float = BARGE_IN_PROBABILITY,
    max_samples: int = 500,
) -> str:
    """Crea una traccia stereo dual-channel (Canale 0: Utente, Canale 1: Agente) dai CLIP REALI.

    Legge i clip mp3 dal path assoluto in `row['path']` (colonna `audio_path` del
    loader MDC, rinominata in `load_mozilla_italian_dataset`). I clip sono mono a
    32kHz: vengono ricampionati a `sample_rate` (default 24kHz: FIX PERF — prima
    48kHz; il tokenizer Qwen3-TTS è nativo 24k e il mel extractor resample a 16k,
    quindi 48k era lavoro CPU/IO sprecato; 24k dimezza costo di resample e dimensione
    wav), normalizzati a RMS 0.05 (coerente con i livelli di DuplexConv) e alternati
    tra i due canali con pause. Mantiene la simulazione di BARGE-IN / OVERLAP (20%):
    l'utente interrompe l'agente, che si ferma dopo il primo 50% del turno.

    NIENTE più rumore bianco sintetico: se nessun clip reale è caricabile, il
    contributo MDC diventa un breve silenzio (mai dati degenerati).
    """
    print(
        f"[Fase 1 - Stereo Dual-Channel] Creazione traccia stereo dual-channel (Barge-In simulated @ {int(barge_in_prob * 100)}%)..."
    )
    sys.stdout.flush()
    try:
        import librosa
    except ImportError:
        raise RuntimeError(
            "librosa è necessario per leggere i clip mp3 MDC (installare: pip install librosa)."
        )

    silence_samples = int(silence_sec * sample_rate)
    silence_arr = np.zeros(silence_samples, dtype=np.float32)
    df_sub = df.head(min(len(df), max_samples)).reset_index(drop=True)

    # FIX PERF (DDD): il load MP3 seriale può essere il collo di bottiglia (fino a ~900ms
    # /clip a freddo). Carichiamo i clip in PARALLELO con ThreadPoolExecutor: con la page
    # cache calda è ~3-5ms/clip (2500 clip = ~15s). Uso i THREAD (non multiprocessing.Pool):
    # la closure _load_clip non è picklable per il Pool e i thread danno lo stesso speedup
    # per load I/O-bound. Il dialogo resta in streaming dal risultato.
    def _load_clip(row):
        clip_path = str(row.get("path") or "").strip()
        if not clip_path or not os.path.isfile(clip_path):
            return None
        try:
            y, sr_src = librosa.load(clip_path, sr=None, mono=True)
            if len(y) == 0:
                return None
            y = y.astype(np.float32)
            if sr_src != sample_rate:
                y = librosa.resample(y, orig_sr=sr_src, target_sr=sample_rate)
            rms = float(np.sqrt(np.mean(y**2))) + 1e-8
            y = y * (0.05 / rms)
            return y, sr_src
        except Exception:
            return None

    _rows = df_sub.to_dict("records")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(4, (os.cpu_count() or 1))
    ) as _pool:
        _loaded = list(
            tqdm(
                _pool.map(_load_clip, _rows, chunksize=8),
                total=len(_rows),
                desc="Load clip MDC (threads)",
                unit="clip",
            )
        )

    pbar = tqdm(
        df_sub.iterrows(),
        total=len(df_sub),
        desc="Generazione Stereo Dual-Channel",
        unit="clip",
    )
    barge_in_count, loaded = 0, 0
    prev_chunk: Optional[np.ndarray] = (
        None  # clip precedente per la voce dell'agente durante il barge-in
    )
    prev_text: str = ""
    agent_turn_info: list = []  # {start_sample, end_sample, text} per ogni turno AGENTE (per il testo nei target)
    offset = 0  # campioni già emessi nei canali (per l'allineamento testo->frame)

    # FIX ANTI-OOM (DDD): prima la traccia veniva accumulata in liste Python
    # (ch_user/ch_agent) e poi np.column_stack -> con 2500 clip (~5.2GB float32 stereo
    # 48kHz) la RAM esplodeva. Ora scriviamo in STREAMING blocco-per-blocco su
    # sf.SoundFile (modalità write): in RAM resta solo il blocco corrente (~30s).
    os.makedirs(os.path.dirname(output_wav_path) or ".", exist_ok=True)
    sfw = sf.SoundFile(
        output_wav_path, mode="w", samplerate=sample_rate, channels=2, subtype="PCM_16"
    )
    stereo_buf = []  # blocchi (N,2) float32 pronti per la scrittura
    _buf_samples = 0

    def _flush_stereo() -> None:
        nonlocal stereo_buf, _buf_samples
        if stereo_buf:
            sfw.write(np.concatenate(stereo_buf, axis=0))
            stereo_buf = []
            _buf_samples = 0

    for idx, row in pbar:
        # FIX PERF: usa il clip già caricato dal Pool (niente reload MP3 nel loop)
        _pre = _loaded[idx] if idx < len(_loaded) else None
        if _pre is None:
            continue
        y, sr_src = _pre
        loaded += 1
        audio_chunk = y
        row_text = str(row.get("sentence") or row.get("transcription") or "").strip()

        # Simulazione Sovrapposizione / Barge-In (Interruzione): l'utente interrompe l'agente
        is_barge_in = (
            (np.random.rand() < barge_in_prob)
            and (idx > 0)
            and (prev_chunk is not None)
        )

        if is_barge_in:
            barge_in_count += 1
            overlap_dur = int(len(audio_chunk) * 0.5)
            user_speech = audio_chunk
            agent_speech = np.zeros(len(audio_chunk), dtype=np.float32)
            # L'agente stava parlando (clip precedente) e si ferma appena l'utente interrompe
            prev_piece = (
                prev_chunk[:overlap_dur]
                if len(prev_chunk) >= overlap_dur
                else np.resize(prev_chunk, overlap_dur)
            )
            agent_speech[:overlap_dur] = prev_piece * 0.6

            n_turn = len(user_speech) + len(silence_arr)
            block = np.zeros((n_turn, 2), dtype=np.float32)
            block[: len(user_speech), 0] = user_speech
            block[len(user_speech) :, 0] = silence_arr
            block[: len(agent_speech), 1] = agent_speech
            block[len(agent_speech) :, 1] = silence_arr
            stereo_buf.append(block)
            _buf_samples += n_turn
            # Turno agente = audio del clip precedente (interrotto dopo il 50%)
            agent_turn_info.append(
                {
                    "start_sample": offset,
                    "end_sample": offset + overlap_dur,
                    "text": prev_text,
                }
            )
            offset += n_turn
        else:
            n_turn = len(audio_chunk) + len(silence_arr)
            block = np.zeros((n_turn, 2), dtype=np.float32)
            if idx % 2 == 0:
                block[: len(audio_chunk), 0] = audio_chunk
                block[len(audio_chunk) :, 0] = silence_arr
            else:
                block[: len(audio_chunk), 1] = audio_chunk
                block[len(audio_chunk) :, 1] = silence_arr
                agent_turn_info.append(
                    {
                        "start_sample": offset,
                        "end_sample": offset + len(audio_chunk),
                        "text": row_text,
                    }
                )
            stereo_buf.append(block)
            _buf_samples += n_turn

        # Flush periodico: mantiene la RAM ~costante (blocchi fino a ~30s di audio)
        if _buf_samples >= 30 * sample_rate:
            _flush_stereo()

        prev_chunk = audio_chunk
        prev_text = row_text
        pbar.set_postfix({"clip reali": loaded, "Barge-in": barge_in_count})

    if loaded == 0:
        print(
            "[Fase 1 - Stereo Dual-Channel] ⚠️ NESSUN clip reale caricato: contributo MDC = silenzio (2s, nessun dato degenerato)."
        )
        sys.stdout.flush()
        placeholder = np.zeros(int(2 * sample_rate), dtype=np.float32)
        stereo_audio = np.column_stack((placeholder, placeholder))
        sfw.write(stereo_audio)
        agent_turn_info = []
    else:
        _flush_stereo()

    sfw.close()
    total_dur = offset
    print(
        f"[Fase 1 - Stereo Dual-Channel] Traccia salvata in: {output_wav_path} ({loaded} clip reali, {barge_in_count} barge-in, durata {total_dur / sample_rate:.0f}s)"
    )
    sys.stdout.flush()
    return output_wav_path, agent_turn_info


def tokenize_dual_channel_audio(
    audio_path: str,
    qwen_tokenizer: Any = None,
    mel_extractor: Any = None,
    agent_turn_info: Optional[list] = None,
    text_tokenizer: Any = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    """Tokenizza l'audio stereo: mel del canale utente (NATIVO) + codici RVQ agente.

    Se `agent_turn_info` (lista di {start_sample, end_sample, text}, dal
    create_dual_channel_dialogue) e `text_tokenizer` sono forniti, produce anche
    `text_by_window` (testo della risposta agente per finestra di 64 frame,
    tokenizzato in BPE, None se assente) e `segment_ends` (indici di frame della
    fine di ogni turno agente, per lo split di validazione a confine di turno).
    """
    print(f"[Fase 1 - Tokenizzazione Stereo] Lettura file audio {audio_path}...")
    sys.stdout.flush()
    # FIX ANTI-OOM (DDD): prima sf.read caricava TUTTA la traccia in RAM (con MDC esteso
    # ~2500 clip = ~3.7h stereo 48kHz = ~2.6GB float32). Ora leggiamo in STREAMING a
    # blocchi di 120s e concateniamo SOLO gli output (mel float16 + codici int16),
    # che sono piccoli (~350MB totali anche con 50h di audio).
    sr = sf.info(audio_path).samplerate
    _total = sf.info(audio_path).frames
    print(
        f"[Fase 1 - Tokenizzazione Stereo] Traccia: {_total / sr:.0f}s, {sr} Hz, streaming a blocchi da 120s."
    )
    sys.stdout.flush()
    user_mel_parts, agent_mel_parts, agent_tokens_parts = [], [], []
    with sf.SoundFile(audio_path, "r") as sfr:
        _chunk = 120 * sr
        while True:
            block = sfr.read(_chunk, dtype="float32", always_2d=True)
            if block.shape[0] == 0:
                break
            user_wave = block[:, 0]
            agent_wave = block[:, 1]
            u_mel_b = extract_mel_features(mel_extractor, user_wave, sr=sr)
            a_tok_b = encode_audio_with_qwen_tts(qwen_tokenizer, agent_wave, sr=sr)
            a_mel_b = extract_mel_features(mel_extractor, agent_wave, sr=sr)
            if u_mel_b is not None and a_mel_b is not None and a_tok_b is not None:
                # Allinea il mel del BLOCCA al n_agent del blocco (tronca/pada 8:1)
                # PRIMA di concatenare: il merge a fine funzione usa i totali.
                n_b = a_tok_b.shape[1]
                u_mel_b = align_mel_to_codes(u_mel_b, n_b)
                a_mel_b = align_mel_to_codes(a_mel_b, n_b)
                a_tok_b = a_tok_b[:, :n_b]
                user_mel_parts.append(u_mel_b)
                agent_mel_parts.append(a_mel_b)
                agent_tokens_parts.append(a_tok_b)
    user_mel = (
        np.vstack(user_mel_parts).astype(np.float16) if user_mel_parts else None
    )
    agent_mel = (
        np.vstack(agent_mel_parts).astype(np.float16) if agent_mel_parts else None
    )
    agent_tokens = (
        np.hstack(agent_tokens_parts).astype(np.int16)
        if agent_tokens_parts
        else None
    )
    del user_mel_parts, agent_mel_parts, agent_tokens_parts
    gc.collect()

    if user_mel is None or agent_mel is None or agent_tokens is None:
        raise RuntimeError(
            "Tokenizzazione stereo fallita: mel utente/agente o codici agente non validi "
            f"(N_CODEBOOKS={N_CODEBOOKS}). Correggi gli iperparametri prima di procedere."
        )
    n_agent = agent_tokens.shape[1]
    user_mel = align_mel_to_codes(user_mel, n_agent)
    agent_mel = align_mel_to_codes(agent_mel, n_agent)
    agent_tokens = agent_tokens[:, :n_agent]
    print(
        f"[Fase 1 - Tokenizzazione Stereo] Tokenizzazione completata con successo "
        f"({n_agent} frame agente @ 12.5 Hz, mel utente/agente {user_mel.shape[0]} frame @ 100 Hz, "
        f"{agent_tokens.shape[0]} codebook)."
    )
    sys.stdout.flush()

    # ---- Testo risposta agente per finestra (solo MDC: i turni hanno trascrizioni) ----
    text_by_window: list = [None] * (n_agent // 64 + (1 if n_agent % 64 else 0))
    segment_ends: list = []
    if agent_turn_info and text_tokenizer is not None and n_agent > 0:
        hop = sr / FRAME_RATE_HZ  # campioni per frame agente
        # mappa frame -> testo del turno che lo copre (maggioranza per finestra)
        frame_text = [None] * n_agent
        for turn in agent_turn_info:
            s0 = int(turn["start_sample"] // hop)
            s1 = min(int((turn["end_sample"] - 1) // hop) + 1, n_agent)
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            for t in range(s0, s1):
                if t >= 0 and t < n_agent:
                    frame_text[t] = text
            segment_ends.append(s1)
        if segment_ends and segment_ends[-1] < n_agent:
            segment_ends.append(n_agent)
        for w in range(len(text_by_window)):
            f0, f1 = w * 64, min((w + 1) * 64, n_agent)
            from collections import Counter

            cnt = Counter(x for x in frame_text[f0:f1] if x is not None)
            if cnt:
                best = cnt.most_common(1)[0][0]
                ids = text_tokenizer(best)
                text_by_window[w] = ids if ids else None
        n_text = sum(1 for t in text_by_window if t is not None)
        print(
            f"[Fase 1 - Testo] {n_text}/{len(text_by_window)} finestre MDC con testo risposta agente "
            f"({n_agent} frame agente, {len(segment_ends)} turni)."
        )
        sys.stdout.flush()
    return user_mel, agent_mel, agent_tokens, text_by_window, segment_ends


def _write_mel_parquet(path: str, mel: np.ndarray) -> None:
    """Scrive una matrice mel per-frame (T, MEL_BINS) in Parquet (colonne mel_k float16)."""
    df = pd.DataFrame(
        {f"mel_{k}": mel[:, k].astype(np.float16) for k in range(mel.shape[1])}
    )
    df.to_parquet(path, compression="snappy", index=False)


def export_duplex_parquet(
    parquet_path: str,
    user_mel: np.ndarray,
    agent_mel: np.ndarray,
    agent_codes: np.ndarray,
    text_by_window: Optional[list] = None,
    segment_ends: Optional[list] = None,
) -> str:
    """Esporta mel utente + mel agente (100fps) e codici agente (12.5fps) in Parquet compressi.

    Formati:
      - *_mel.parquet: colonne mel_k (T_mel x MEL_BINS) float16     (canale UTENTE)
      - *_ameld.parquet: colonne mel_k (T_mel x MEL_BINS) float16   (canale AGENTE, auto-ascolto)
      - *_codes.parquet: colonne a_cb_k (N_CODEBOOKS x T_agent) int16
      - *_text.parquet: testo risposta agente per finestra (window_idx, text=list[int])
    Allineamento: T_mel == 8 * T_agent esattamente (garantito da align_mel_to_codes).
    I confini di segmento (per la validazione) vanno nel sidecar _hyperparams.json.
    """
    print(
        f"[Fase 1 - Parquet Export] Preparazione mel utente/agente ({user_mel.shape}) e codici ({agent_codes.shape})..."
    )
    sys.stdout.flush()
    os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)
    mel_path = parquet_path.replace(".parquet", "_mel.parquet")
    ameld_path = parquet_path.replace(".parquet", "_ameld.parquet")
    codes_path = parquet_path.replace(".parquet", "_codes.parquet")

    _write_mel_parquet(mel_path, user_mel)
    _write_mel_parquet(ameld_path, agent_mel)

    df_codes = pd.DataFrame(
        {
            f"a_cb_{k}": agent_codes[k].astype(np.int16)
            for k in range(agent_codes.shape[0])
        }
    )
    df_codes.to_parquet(codes_path, compression="snappy", index=False)

    # Testo risposta agente per finestra (solo finestre MDC; vuote = nessun testo)
    if text_by_window is not None and any(t is not None for t in text_by_window):
        text_path = parquet_path.replace(".parquet", "_text.parquet")
        rows = [
            (i, list(t) if t is not None else []) for i, t in enumerate(text_by_window)
        ]
        df_text = pd.DataFrame(rows, columns=["window_idx", "text"])
        df_text.to_parquet(text_path, compression="snappy", index=False)
        n_nonempty = int((df_text["text"].str.len() > 0).sum())
        print(
            f"[Fase 1 - Parquet Export] Testo salvato: {text_path} ({n_nonempty}/{len(df_text)} finestre con testo)."
        )
        sys.stdout.flush()
    elif segment_ends:
        print(
            "[Fase 1 - Parquet Export] Nessuna finestra con testo: training SOLO audio (testo disattivato)."
        )
        sys.stdout.flush()

    # I confini di segmento finiscono nel sidecar JSON (letto dalla Fase 2 per lo split di validazione)
    if segment_ends:
        import json

        sidecar = parquet_path.replace(".parquet", "_hyperparams.json")
        try:
            with open(sidecar) as f:
                d = json.load(f)
            d["SEGMENT_ENDS"] = [int(x) for x in segment_ends]
            with open(sidecar, "w") as f:
                json.dump(d, f, indent=2)
        except Exception:
            pass

    mel_mb = os.path.getsize(mel_path) / (1024 * 1024)
    ameld_mb = os.path.getsize(ameld_path) / (1024 * 1024)
    codes_mb = os.path.getsize(codes_path) / (1024 * 1024)
    print(
        f"[Fase 1 - Parquet Export] Cache salvata: mel utente {mel_path} ({mel_mb:.2f} MB), "
        f"mel agente {ameld_path} ({ameld_mb:.2f} MB), codici {codes_path} ({codes_mb:.2f} MB)."
    )
    sys.stdout.flush()
    return parquet_path


def execute_phase_1():
    """Esegue la Fase 1: estrazione mel utente + codici RVQ agente e salvataggio Parquet (DuplexConv + MDC Italiano)."""
    global SILENCE_TOKEN_IDS
    start_time = time.time()
    print("\n====================================================================")
    print(
        "=== AVVIO FASE 1: PREPROCESSING & TOKENIZZAZIONE PARQUET (DUPLEXCONV + MDC ITA) ==="
    )
    print("====================================================================\n")
    sys.stdout.flush()
    parquet_cache = (
        f"data/train_tokens_{DUPLEXCONV_MAX_HOURS}h_{MDC_MAX_SAMPLES}s.parquet"
    )
    qwen_tokenizer = load_qwen_tts_tokenizer()
    # Estrazione automatica dei pesi del decoder Code2Wav: in Fase 2 (kernel riavviato,
    # transformers 5.x) il decoder non è caricabile -> serve la replica standalone
    # (code2wav_weights/), bit-exact verificata. Costo quasi zero: state_dict in RAM.
    if qwen_tokenizer is not None:
        try:
            extract_code2wav_decoder_weights(qwen_tokenizer)
        except Exception as e:
            print(
                f"[Fase 1 - Code2Wav] ⚠️ Estrazione decoder fallita ({e}): in Fase 2 userà il fallback mock (beep)."
            )
            sys.stdout.flush()
    # Estrazione automatica del sub-talker Qwen3-TTS (ADR-005): OBBLIGATORIO in
    # Fase 2 (fail-fast). Come il decoder, non blocca la Fase 1 se fallisce
    # (es. secret HF_TOKEN mancante): stampa istruzioni per il retry.
    _ensure_subtalker()
    mel_extractor = load_mel_extractor()
    text_tokenizer = (
        load_e2b_text_tokenizer()
    )  # BPE Gemma 4 E2B per il testo nei target (solo MDC)

    # 0. Estrazione automatica dei codici di silenzio (il tokenizer 12Hz NON ha un token di silenzio esplicito)
    SILENCE_TOKEN_IDS = [int(x) for x in extract_silence_codes(qwen_tokenizer)]

    # 1. Caricamento e tokenizzazione DuplexConv (trascrizioni ASR dagli archivi jsons.tar.gz)
    duplexconv_subset = load_duplexconv_dataset(max_hours=DUPLEXCONV_MAX_HOURS)
    duplex_annotations = (
        load_duplex_annotations() if duplexconv_subset is not None else {}
    )
    (
        u_mel_duplex,
        a_mel_duplex,
        a_codes_duplex,
        text_by_window_duplex,
        segment_ends_duplex,
    ) = tokenize_duplexconv_subset(
        duplexconv_subset,
        seq_len=64,
        qwen_tokenizer=qwen_tokenizer,
        mel_extractor=mel_extractor,
        annotations=duplex_annotations,
        text_tokenizer=text_tokenizer,
    )
    # Le annotazioni non servono più (il testo è in memoria): libera ~1GB PRIMA del
    # download MDC per non stringere il budget disco totale.
    del duplex_annotations
    gc.collect()
    _jsons_dir = os.path.join(big_disk_base(), "duplex_jsons")
    if os.path.isdir(_jsons_dir):
        shutil.rmtree(_jsons_dir, ignore_errors=True)
        print(f"[Fase 1 - Pulizia] Annotazioni DuplexConv rimosse ({_jsons_dir}).")
        sys.stdout.flush()

    # 2. Caricamento e tokenizzazione MDC Italiano (clip REALI + trascrizioni -> testo nei target)
    df_mdc = load_mozilla_italian_dataset(max_samples=MDC_MAX_SAMPLES)
    mdc_wav_path = os.path.join(big_disk_base(), "train_italiano_dual.wav")
    # FIX (DDD): create_dual_channel_dialogue ha default max_samples=500; senza passare
    # MDC_MAX_SAMPLES il dialogo MDC usava SOLO 500 clip anche con 2500 richiesti.
    wav_path, mdc_turn_info = create_dual_channel_dialogue(
        df_mdc, mdc_wav_path, max_samples=MDC_MAX_SAMPLES
    )
    u_mel_mdc, a_mel_mdc, a_codes_mdc, text_by_window_mdc, segment_ends_mdc = (
        tokenize_dual_channel_audio(
            wav_path,
            qwen_tokenizer=qwen_tokenizer,
            mel_extractor=mel_extractor,
            agent_turn_info=mdc_turn_info,
            text_tokenizer=text_tokenizer,
        )
    )

    del df_mdc
    gc.collect()

    # 3. Concatenazione mel utente + mel agente + codici agente (DuplexConv + MDC Italiano)
    print(
        "[Fase 1 - Merging] Concatenazione mel utente/agente + codici agente (DuplexConv + MDC Italiano)..."
    )
    sys.stdout.flush()
    user_mel = np.vstack([u_mel_duplex, u_mel_mdc]).astype(np.float16)
    agent_mel = np.vstack([a_mel_duplex, a_mel_mdc]).astype(np.float16)
    agent_codes = np.hstack([a_codes_duplex, a_codes_mdc]).astype(np.int16)
    # FIX ANTI-OOM: libera i buffer sorgente dopo la concatenazione (con 50h+ il vstack
    # crea una copia da ~3.5GB mentre i sorgenti sono ancora in RAM).
    del u_mel_duplex, a_mel_duplex, a_codes_duplex
    del u_mel_mdc, a_mel_mdc, a_codes_mdc
    gc.collect()

    # ---- Testo risposta agente per finestra (allineato ai frame concatenati) ----
    # finestre = blocchi di 64 frame agente; DuplexConv porta la trascrizione ASR del
    # canale agente (asr[1].labels.txt), MDC il testo del turno agente. Le finestre
    # MDC partono dopo quelle Duplex.
    n_windows = agent_codes.shape[1] // 64 + (1 if agent_codes.shape[1] % 64 else 0)
    # FIX (DDD): il testo DuplexConv è allineato PER-SEGMENTO (ceil per segmento) ma la
    # Fase 2 lo indicizza PER FINESTRE GLOBALI di 64 frame contigui -> se un segmento non
    # ha lunghezza multipla di 64 il testo si sfasa e len(text_by_window_duplex) può
    # superare n_windows, facendo saltare il blocco MDC (bug: guard n_dc_windows<=n_windows
    # falso, testo MDC mai aggiunto, e ultime finestre text oltre i frame reali).
    # Ricostruiamo text_by_window troncando il duplex alle finestre REALI (n_windows)
    # e aggiungendo MDC SOLO per le finestre residue, senza disallineamento.
    text_by_window: list = list(text_by_window_duplex)[:n_windows]
    if len(text_by_window) < n_windows:
        text_by_window.extend([None] * (n_windows - len(text_by_window)))
    n_dc_windows = len(text_by_window_duplex)
    mdc_free = max(0, n_windows - n_dc_windows)
    if mdc_free > 0:
        mdc_len = min(len(text_by_window_mdc), mdc_free)
        text_by_window[n_dc_windows : n_dc_windows + mdc_len] = text_by_window_mdc[
            :mdc_len
        ]

    # ---- Confini di segmento cumulati (validazione a confine di conversazione) ----
    segment_ends = list(segment_ends_duplex)
    mdc_offset = int(a_codes_duplex.shape[1])
    segment_ends.extend([mdc_offset + int(e) for e in segment_ends_mdc])
    segment_ends.sort()
    n_text = sum(1 for t in text_by_window if t is not None)
    print(
        f"[Fase 1 - Testo] Finestre totali: {n_windows}, con testo risposta: {n_text} "
        f"({100.0 * n_text / max(1, n_windows):.1f}%), confini di segmento: {len(segment_ends)}."
    )
    sys.stdout.flush()

    # Validazione struttura
    if agent_codes.shape[0] != N_CODEBOOKS:
        raise RuntimeError(
            f"Struttura codici non valida: attesi {N_CODEBOOKS} codebook, ottenuti "
            f"{agent_codes.shape[0]}. Rivedi la tokenizzazione."
        )
    for name, mel in (("utente", user_mel), ("agente", agent_mel)):
        if mel.shape[0] != agent_codes.shape[1] * MEL_PER_AGENT_FRAME:
            raise RuntimeError(
                f"Disallineamento mel {name}/codici: mel={mel.shape[0]} frame, codici={agent_codes.shape[1]} "
                f"(attesi {agent_codes.shape[1] * MEL_PER_AGENT_FRAME} frame mel)."
            )

    a0 = agent_codes[0]
    vals, counts = np.unique(a0, return_counts=True)
    top_share = float(counts.max() / counts.sum())
    print(
        f"[Fase 1 - Validazione] Target cb0 agente: {len(vals)} token unici su {a0.shape[0]} frame, "
        f"token dominante {int(vals[counts.argmax()])} al {top_share:.1%}."
    )
    if top_share > 0.95 or len(vals) < 8 or not TOKENIZER_REAL_ENCODING_ACTIVE:
        print(
            "[Fase 1 - Validazione] ⚠️ DATI DEGENERATI: il training collasserà a loss ~0 (pattern osservato). "
            "Verifica che l'encoder Qwen3-TTS reale sia attivo (flag REAL_ENCODING)."
        )
    else:
        print(
            "[Fase 1 - Validazione] ✅ Distribuzione target accettabile per il training."
        )
    sys.stdout.flush()

    # FIX (DDD): save_tokenizer_hyperparams RISCUIVE il sidecar senza SEGMENT_ENDS ->
    # se chiamato DOPO export cancella i confini. Ordine corretto: save PRIMA, export
    # DOPO (export aggiunge SEGMENT_ENDS al JSON esistente).
    save_tokenizer_hyperparams(
        parquet_cache.replace(".parquet", "_hyperparams.json"),
        extra={
            "TEXT_BLOCK_LEN": TEXT_BLOCK_LEN,
            "TEXT_LOSS_WEIGHT": TEXT_LOSS_WEIGHT,
            "E2B_PAD_TOKEN_ID": E2B_PAD_TOKEN_ID,
            "E2B_BOS_TOKEN_ID": E2B_BOS_TOKEN_ID,
            "E2B_EOS_TOKEN_ID": E2B_EOS_TOKEN_ID,
            "VAL_FRACTION": VAL_FRACTION,
            "text_tokenizer": "google/gemma-4-E2B-it",
        },
    )
    export_duplex_parquet(
        parquet_cache,
        user_mel,
        agent_mel,
        agent_codes,
        text_by_window=text_by_window,
        segment_ends=segment_ends,
    )

    # Pulizia file temporanei (disco grande): wav stereo MDC ~1.3GB + clip estratti
    # ~10GB + annotazioni DuplexConv ~1GB. Dopo il parquet non servono più.
    # NB: i binari .dat (conversione memmap) NON vengono toccati: servono alla Fase 2.
    try:
        _big = big_disk_base()
        for _tmp in (
            os.path.join(_big, "train_italiano_dual.wav"),
            os.path.join(_big, "mdc_download"),
            os.path.join(_big, "duplex_jsons"),
        ):
            _p = _tmp
            if os.path.isfile(_p):
                os.remove(_p)
                print(f"[Fase 1 - Pulizia] Rimosso {_p}")
            elif os.path.isdir(_p):
                shutil.rmtree(_p, ignore_errors=True)
                print(f"[Fase 1 - Pulizia] Rimossa directory {_p}")
    except Exception as e:
        print(f"[Fase 1 - Pulizia] Avviso pulizia file temporanei ({e}).")
    # FIX ANTI-OOM: converte i parquet in binari .dat memmap (RAM ~0 in Fase 2).
    convert_parquet_to_memmap(parquet_cache)
    elapsed = time.time() - start_time
    print(f"\n====================================================================")
    print(
        f"[FASE 1 COMPLETATA IN {elapsed:.2f}s] File Parquet salvati su: {parquet_cache} (mel utente + mel agente + codes)"
    )
    print("[FASE 1] Ora puoi riavviare il kernel Kaggle ed eseguire la FASE 2.")
    print("====================================================================\n")
    sys.stdout.flush()


# %% [markdown]
# ## ====================================================================
# ## FASE 2: UPGRADE TRANSFORMERS, TRAINING GEMMA 4 & SINTESI CODE2WAV (Celle 4 - 7)
# ## ====================================================================

# %% [code]
# Cell 4: Setup Ambiente Fase 2 (Upgrade a Transformers 5.x & Unsloth)
# Esegui questa cella all'inizio della Fase 2:
# !pip install -q --upgrade unsloth unsloth_zoo transformers accelerate bitsandbytes peft trl tqdm


def load_tokens_from_parquet(
    parquet_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Carica i dati dal Parquet cache: mel utente + mel agente (T, MEL_BINS) + codici agente (N_CB, T)
    + testo risposta agente per finestra (list per finestra, None se assente).

    FIX ANTI-OOM (DDD): con dataset estesi (50h+MDC = 13.6M mel frame/canale) il load
    parquet materializza ~7.3GB per file (decompressione + layout colonnare) -> picco
    ~21GB per mel+ameld+codes -> OOM. Se i file binari grezzi .dat (float16 row-major,
    generati da convert_parquet_to_memmap) esistono, si usano np.memmap: zero copie in
    RAM, accesso on-demand via page cache del SO. Il parquet resta come formato di
    riferimento (condivisione/backup), il .dat come formato di training efficiente.
    """
    print(f"[Fase 2 - DataLoad] Caricamento dati da cache Parquet: {parquet_path}...")
    sys.stdout.flush()
    load_tokenizer_hyperparams(parquet_path.replace(".parquet", "_hyperparams.json"))
    t0 = time.time()
    mel_path = parquet_path.replace(".parquet", "_mel.parquet")
    ameld_path = parquet_path.replace(".parquet", "_ameld.parquet")
    codes_path = parquet_path.replace(".parquet", "_codes.parquet")
    # Percorsi dei binari memmap (su disco grande per non toccare la quota working)
    _big = big_disk_base()
    dat_user = os.path.join(_big, "mel_user.dat")
    dat_agent = os.path.join(_big, "mel_agent.dat")
    dat_codes = os.path.join(_big, "codes.dat")
    if os.path.exists(dat_user) and os.path.exists(dat_agent) and os.path.exists(dat_codes):
        _shp_user = (os.path.getsize(dat_user) // (MEL_BINS * 2), MEL_BINS)
        _shp_agent = (os.path.getsize(dat_agent) // (MEL_BINS * 2), MEL_BINS)
        _n_cb = int(
            np.fromfile(dat_codes, dtype=np.int16, count=1)[0]
            if os.path.getsize(dat_codes) >= 2
            else N_CODEBOOKS
        )
        # Formato codes.dat: header int16 N_CB (2B) + dati piatti N_CB*T int16.
        # Il load è un array 1D (header + dati) reshapato a (N_CB, T): così le righe
        # restano allineate (l'header di 2B romperebbe una vista 2D diretta).
        _n_frame = (os.path.getsize(dat_codes) - 2) // (2 * _n_cb)
        user_mel = np.memmap(dat_user, dtype=np.float16, mode="r", shape=_shp_user)
        agent_mel = np.memmap(dat_agent, dtype=np.float16, mode="r", shape=_shp_agent)
        _codes_flat = np.memmap(dat_codes, dtype=np.int16, mode="r")
        agent_codes = _codes_flat[1 : 1 + _n_cb * _n_frame].reshape(
            _n_cb, _n_frame
        )
        t_el = time.time() - t0
        print(
            f"[Fase 2 - DataLoad] MEMORY-MAPPED: {agent_codes.shape[1]} frame agente "
            f"(mel utente {user_mel.shape[0]}x{user_mel.shape[1]} float16 mmap, "
            f"mel agente {agent_mel.shape[0]}x{agent_mel.shape[1]} mmap) in {t_el:.2f}s "
            f"(0 copie in RAM)."
        )
        sys.stdout.flush()
    else:
        # Fallback legacy (parquet in RAM) — lento/non-OOM-safe per dataset grandi
        df_mel = pd.read_parquet(mel_path, memory_map=True)
        df_ameld = pd.read_parquet(ameld_path, memory_map=True)
        df_codes = pd.read_parquet(codes_path, memory_map=True)
        mel_cols = [c for c in df_mel.columns if c.startswith("mel_")]
        code_cols = [c for c in df_codes.columns if c.startswith("a_cb_")]
        user_mel = df_mel[mel_cols].values.astype(np.float16)  # (T_mel, MEL_BINS)
        agent_mel = df_ameld[mel_cols].values.astype(np.float16)  # (T_mel, MEL_BINS)
        agent_codes = df_codes[code_cols].values.T.astype(
            np.int16
        )  # (N_CODEBOOKS, T_agent)
        del df_mel, df_ameld, df_codes
        gc.collect()
        t_el = time.time() - t0
        print(
            f"[Fase 2 - DataLoad] Caricati {agent_codes.shape[1]} frame agente "
            f"(mel utente {user_mel.shape[0]}x{user_mel.shape[1]}, mel agente {agent_mel.shape[0]}x{agent_mel.shape[1]}) "
            f"da Parquet in {t_el:.4f}s."
        )
        sys.stdout.flush()
    # Testo risposta agente per finestra (colonna `text`: lista di id BPE)
    text_by_window: list = [None] * (
        agent_codes.shape[1] // 64 + (1 if agent_codes.shape[1] % 64 else 0)
    )
    text_path = parquet_path.replace(".parquet", "_text.parquet")
    if os.path.exists(text_path):
        try:
            df_text = pd.read_parquet(text_path)
            for _, r in df_text.iterrows():
                i = int(r["window_idx"])
                ids = list(r["text"]) if r["text"] is not None else []
                if i < len(text_by_window):
                    text_by_window[i] = [int(x) for x in ids] if ids else None
            n_text = sum(1 for t in text_by_window if t is not None)
            print(
                f"[Fase 2 - DataLoad] Testo risposta agente: {n_text}/{len(text_by_window)} finestre."
            )
        except Exception as e:
            print(
                f"[Fase 2 - DataLoad] Avviso caricamento testo ({e}): training SOLO audio."
            )

    t_el = time.time() - t0
    print(
        f"[Fase 2 - DataLoad] Caricati {agent_codes.shape[1]} frame agente "
        f"(mel utente {user_mel.shape[0]}x{user_mel.shape[1]}, mel agente {agent_mel.shape[0]}x{agent_mel.shape[1]}) "
        f"da Parquet in {t_el:.4f}s."
    )
    sys.stdout.flush()
    return user_mel, agent_mel, agent_codes, text_by_window


def convert_parquet_to_memmap(parquet_path: str) -> bool:
    """Converte i parquet mel/codes in binari grezzi .dat (float16 row-major) per il load
    memmap a RAM ~0 in Fase 2 (FIX ANTI-OOM). I file .dat finiscono sul disco grande.

    Formato:
      mel_user.dat / mel_agent.dat: (T_mel, MEL_BINS) float16 raw (T_mel * MEL_BINS * 2 byte)
      codes.dat: header int16 N_CODEBOOKS + (N_CODEBOOKS, T_agent) int16 raw
    Ritorna True se la conversione è andata a buon fine o i .dat erano già presenti.
    """
    _big = big_disk_base()
    dat_user = os.path.join(_big, "mel_user.dat")
    dat_agent = os.path.join(_big, "mel_agent.dat")
    dat_codes = os.path.join(_big, "codes.dat")
    if os.path.exists(dat_user) and os.path.exists(dat_agent) and os.path.exists(dat_codes):
        print("[Fase 1/2 - Memmap] File .dat già presenti: nessuna conversione necessaria.")
        sys.stdout.flush()
        return True
    mel_path = parquet_path.replace(".parquet", "_mel.parquet")
    ameld_path = parquet_path.replace(".parquet", "_ameld.parquet")
    codes_path = parquet_path.replace(".parquet", "_codes.parquet")
    if not (os.path.exists(mel_path) and os.path.exists(codes_path)):
        return False
    print(
        "[Fase 1/2 - Memmap] Conversione parquet -> binari .dat (RAM peak ~3.3GB/file, una tantum)..."
    )
    sys.stdout.flush()
    try:
        for src, dst in ((mel_path, dat_user), (ameld_path, dat_agent)):
            if os.path.exists(dst):
                continue
            df = pd.read_parquet(src)
            cols = [c for c in df.columns if c.startswith("mel_")]
            arr = df[cols].values.astype(np.float16)
            arr.tofile(dst)
            del df, arr
            gc.collect()
            print(f"  {os.path.basename(dst)} scritto ({os.path.getsize(dst)/1e9:.2f} GB).")
            sys.stdout.flush()
        if not os.path.exists(dat_codes):
            dfc = pd.read_parquet(codes_path)
            ccols = [c for c in dfc.columns if c.startswith("a_cb_")]
            arrc = dfc[ccols].values.T.astype(np.int16)  # (N_CB, T)
            with open(dat_codes, "wb") as f:
                f.write(np.int16(N_CODEBOOKS).tobytes())
                arrc.tofile(f)
            del dfc, arrc
            gc.collect()
            print(f"  codes.dat scritto ({os.path.getsize(dat_codes)/1e6:.1f} MB).")
            sys.stdout.flush()
        print("[Fase 1/2 - Memmap] Conversione completata.")
        sys.stdout.flush()
        return True
    except Exception as e:
        print(f"[Fase 1/2 - Memmap] Conversione fallita ({e}): si userà il fallback parquet.")
        sys.stdout.flush()
        return False


def extend_parquet_with_mdc(
    qwen_tokenizer: Any = None,
    mel_extractor: Any = None,
    text_tokenizer: Any = None,
    base_parquet: Optional[str] = None,
) -> bool:
    """Estende un parquet Fase 1 esistente con il contributo MDC Italiano.

    Utile quando la Fase 1 è stata eseguita SENZA datacollective (MDC = mock -> 0 clip):
    invece di rifare l'intera tokenizzazione DuplexConv (2h GPU), carica i parquet già
    esportati, tokenizza SOLO i clip MDC (scarica/reusa la cache sul disco grande) e
    concatena mel utente/agente + codici + testo, riesportando parquet e binari .dat.

    Usa il percorso streaming anti-OOM per la traccia MDC (create_dual_channel_dialogue
    e tokenize_dual_channel_audio con blocchi) e il merge testo corretto (FIX DDD).
    """
    global SILENCE_TOKEN_IDS
    if base_parquet is None:
        base_parquet = (
            f"data/train_tokens_{DUPLEXCONV_MAX_HOURS}h_{MDC_MAX_SAMPLES}s.parquet"
        )
    print(
        f"\n=== ESTENSIONE PARQUET CON MDC (base: {os.path.basename(base_parquet)}) ==="
    )
    sys.stdout.flush()
    # 1) Carica i parquet esistenti (SENZA MDC)
    user_mel, agent_mel, agent_codes, text_by_window = load_tokens_from_parquet(
        base_parquet
    )
    t_duplex = agent_codes.shape[1]
    n_windows_dc = len(text_by_window)
    print(f"  duplex già tokenizzato: {t_duplex} frame agente, {n_windows_dc} finestre.")
    sys.stdout.flush()
    # 2) Tokenizza MDC dai clip (cache su big disk se presente, altrimenti download)
    df_mdc = load_mozilla_italian_dataset(max_samples=MDC_MAX_SAMPLES)
    mdc_wav_path = os.path.join(big_disk_base(), "train_italiano_dual_ext.wav")
    wav_path, mdc_turn_info = create_dual_channel_dialogue(
        df_mdc, mdc_wav_path, max_samples=MDC_MAX_SAMPLES
    )
    (
        u_mel_mdc,
        a_mel_mdc,
        a_codes_mdc,
        text_by_window_mdc,
        segment_ends_mdc,
    ) = tokenize_dual_channel_audio(
        wav_path,
        qwen_tokenizer=qwen_tokenizer,
        mel_extractor=mel_extractor,
        agent_turn_info=mdc_turn_info,
        text_tokenizer=text_tokenizer,
    )
    del df_mdc
    gc.collect()
    if a_codes_mdc is None or a_codes_mdc.shape[1] == 0:
        print("[Extend MDC] ❌ Nessun clip MDC tokenizzato: il parquet resta invariato.")
        sys.stdout.flush()
        return False
    t_mdc = a_codes_mdc.shape[1]
    print(f"  MDC tokenizzato: {t_mdc} frame agente ({t_mdc / 12.5 / 3600:.2f}h).")
    sys.stdout.flush()
    # 3) Concatena (mel float16 nativo, niente copie float32)
    user_mel = np.vstack([np.asarray(user_mel, dtype=np.float16), u_mel_mdc]).astype(
        np.float16
    )
    agent_mel = np.vstack([np.asarray(agent_mel, dtype=np.float16), a_mel_mdc]).astype(
        np.float16
    )
    agent_codes = np.hstack([np.asarray(agent_codes, dtype=np.int16), a_codes_mdc])
    del u_mel_mdc, a_mel_mdc, a_codes_mdc
    gc.collect()
    # 4) Merge testo: finestre globali; le finestre MDC partono dopo quelle duplex
    n_windows = agent_codes.shape[1] // 64 + (1 if agent_codes.shape[1] % 64 else 0)
    text_by_window = list(text_by_window)[:n_windows]
    if len(text_by_window) < n_windows:
        text_by_window.extend([None] * (n_windows - len(text_by_window)))
    n_dc_windows = min(len(text_by_window_mdc), n_windows - n_windows_dc)
    if n_dc_windows > 0:
        text_by_window[n_windows_dc : n_windows_dc + n_dc_windows] = text_by_window_mdc[
            :n_dc_windows
        ]
    # 5) Segment ends: duplex (dai parquet non conservato) + MDC offsetato
    segment_ends = [t_duplex + int(e) for e in segment_ends_mdc]
    # 6) Riesporta
    # FIX (DDD): ordine save-PRIMA/export-DOPO (export aggiunge SEGMENT_ENDS al sidecar)
    save_tokenizer_hyperparams(
        base_parquet.replace(".parquet", "_hyperparams.json"),
        extra={
            "TEXT_BLOCK_LEN": TEXT_BLOCK_LEN,
            "TEXT_LOSS_WEIGHT": TEXT_LOSS_WEIGHT,
            "E2B_PAD_TOKEN_ID": E2B_PAD_TOKEN_ID,
            "E2B_BOS_TOKEN_ID": E2B_BOS_TOKEN_ID,
            "E2B_EOS_TOKEN_ID": E2B_EOS_TOKEN_ID,
            "VAL_FRACTION": VAL_FRACTION,
            "text_tokenizer": "google/gemma-4-E2B-it",
        },
    )
    export_duplex_parquet(
        base_parquet,
        user_mel,
        agent_mel,
        agent_codes,
        text_by_window=text_by_window,
        segment_ends=segment_ends,
    )
    convert_parquet_to_memmap(base_parquet)
    print(
        f"\n=== EXTEND MDC COMPLETATO: {agent_codes.shape[1]} frame agente "
        f"({t_duplex} duplex + {t_mdc} MDC), {n_windows} finestre ==="
    )
    sys.stdout.flush()
    return True


class Code2WavDecoder(nn.Module):
    """
    Decoder Audio Standalone Code2Wav (da Qwen3-TTS Tokenizer 12Hz):
    Converte matrici di 16 codebook RVQ in audio PCM waveform a 24kHz.
    Funziona in Fase 2 SENZA richiedere la libreria qwen-tts o versioni vecchie di transformers nel processo principale.

    Percorsi di caricamento (in ordine):
      1) In-process: Decoder Ufficiale Qwen3TTSTokenizerV2Model (se qwen_tts è importabile direttamente);
      2) Subprocess: Decoder Ufficiale via /opt/venv_phase1/bin/python + decode_official_worker.py (ADR-003);
      3) In-process: AutoModel.from_pretrained (fallback standard transformers);
      4) Replica torch pura da `code2wav_weights/`;
      5) Fallback mock (sinusoide) — solo se nessuno dei precedenti è disponibile.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz"):
        super().__init__()
        self.model_name = model_name
        self.decoder_module = None
        self._worker_proc = None
        self._venv_py = None
        self._worker_script = None
        self._wrapper_mode = "unknown"
        import atexit
        atexit.register(self._cleanup_worker)
        self._init_decoder()

    def _cleanup_worker(self):
        """Termina e libera esplicitamente il processo worker subprocess."""
        if self._worker_proc is not None:
            try:
                self._worker_proc.stdin.write(json.dumps({"command": "exit"}) + "\n")
                self._worker_proc.stdin.flush()
            except Exception:
                pass
            try:
                self._worker_proc.terminate()
                self._worker_proc.wait(timeout=1.0)
            except Exception:
                try:
                    self._worker_proc.kill()
                except Exception:
                    pass
            self._worker_proc = None

    def _start_worker(self) -> bool:
        """Avvia il worker subprocess nel venv phase1 in modalità daemon persistente."""
        import subprocess

        self._cleanup_worker()

        venv_candidates = [
            "/opt/venv_phase1/bin/python",
            os.environ.get("VENV_PHASE1_PYTHON", ""),
        ]
        venv_py = None
        for p in venv_candidates:
            if p and os.path.exists(p):
                venv_py = p
                break

        if venv_py is None:
            print(f"[Code2Wav][DEBUG] _start_worker FAIL: venv non trovato tra {venv_candidates}")
            sys.stdout.flush()
            return False

        worker_candidates = [
            "/workspace/decode_official_worker.py",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "decode_official_worker.py"),
            os.path.join(os.getcwd(), "decode_official_worker.py"),
        ]
        worker_script = None
        for w in worker_candidates:
            if w and os.path.exists(w):
                worker_script = w
                break

        if worker_script is None:
            print(f"[Code2Wav][DEBUG] _start_worker FAIL: worker script non trovato tra {worker_candidates}")
            sys.stdout.flush()
            return False

        try:
            cmd = [venv_py, worker_script, "--daemon", "--model_name", self.model_name]
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            # Legge risposta handshake READY ignorando eventuali banner/warning delle librerie
            start_t = time.time()
            ready = False
            while time.time() - start_t < 35.0:
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if line == "READY":
                    ready = True
                    break

            if not ready:
                try:
                    proc.kill()
                except Exception:
                    pass
                return False

            self._worker_proc = proc
            self._venv_py = venv_py
            self._worker_script = worker_script
            return True
        except Exception as e:
            print(f"[Code2Wav] Avviso avvio worker subprocess: {e}")
            sys.stdout.flush()
            return False

    def _read_worker_line(self, timeout: float = 60.0):
        """Legge una riga JSON dal worker subprocess."""
        if self._worker_proc is None or self._worker_proc.stdout is None:
            return None
        start_t = time.time()
        while time.time() - start_t < timeout:
            line = self._worker_proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except Exception:
                    pass
        return None

    def _init_decoder(self):
        # 1) Modello ufficiale in-process Qwen3TTSTokenizerV2Model
        try:
            import types
            import importlib.machinery
            if "torchaudio" not in sys.modules:
                mock_ta = types.ModuleType("torchaudio")
                mock_ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
                sys.modules["torchaudio"] = mock_ta
                sys.modules["torchaudio._extension"] = types.ModuleType("torchaudio._extension")
                sys.modules["torchaudio.compliance"] = types.ModuleType("torchaudio.compliance")
                sys.modules["torchaudio.compliance.kaldi"] = types.ModuleType("torchaudio.compliance.kaldi")

            Qwen3TTSTokenizerV2Model = None
            try:
                from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
            except Exception:
                pass

            if Qwen3TTSTokenizerV2Model is not None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tok_model = Qwen3TTSTokenizerV2Model.from_pretrained(
                    self.model_name,
                    token=os.environ.get("HF_TOKEN")
                )
                tok_model = tok_model.to(device).eval()
                self.decoder_module = tok_model
                self._wrapper_mode = "official_inprocess"
                print(f"[Code2Wav] ✅ Decoder Ufficiale Qwen3TTSTokenizerV2Model caricato in-process su {device}.")
                sys.stdout.flush()
                return
        except Exception as e:
            print(f"[Code2Wav] Info caricamento in-process ufficiale: {e}")
            sys.stdout.flush()

        # 2) Decoder Ufficiale via Subprocess nel venv Phase 1 (ADR-003)
        if self._start_worker():
            self._wrapper_mode = "official_subprocess"
            print(f"[Code2Wav] ✅ Decoder Ufficiale Qwen3TTSTokenizerV2Model attivo via Subprocess ({self._venv_py}).")
            sys.stdout.flush()
            return

        # 3) Wrapper transformers AutoModel
        try:
            from transformers import AutoModel

            tokenizer_model = AutoModel.from_pretrained(
                self.model_name,
                attn_implementation="eager",
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN"),
            )
            self.decoder_module = (
                tokenizer_model if hasattr(tokenizer_model, "decode") else None
            )
            if self.decoder_module is None and hasattr(tokenizer_model, "model"):
                self.decoder_module = getattr(tokenizer_model.model, "decoder", None)
            if self.decoder_module is not None:
                self._wrapper_mode = "transformers_automodel"
                print(
                    "[Code2Wav] ✅ Decoder Audio Standalone PyTorch (transformers) caricato."
                )
                sys.stdout.flush()
                return
        except Exception as e:
            print(
                f"[Code2Wav] transformers non carica il decoder ({e}): provo la replica standalone..."
            )
            sys.stdout.flush()

        # 4) Replica torch pura (Fase 2, zero transformers)
        try:
            from code2wav_decoder import load_code2wav_decoder

            decoder_module, cfg = load_code2wav_decoder(CODE2WAV_WEIGHTS_DIR)
            self.decoder_module = decoder_module
            self._wrapper_mode = "standalone_replica"
            print(
                f"[Code2Wav] ✅ Replica standalone torch pura caricata ({CODE2WAV_WEIGHTS_DIR}, "
                f"{len(decoder_module.state_dict())} tensori, upsample x{decoder_module.total_upsample})."
            )
            sys.stdout.flush()
            return
        except Exception as e:
            print(
                f"[Code2Wav] ⚠️ Replica standalone non disponibile ({e}): modalità sintetizzatore mock attiva."
            )
            sys.stdout.flush()
            self._wrapper_mode = "mock"

    def decode(self, audio_codes: torch.Tensor, normalize: bool = True) -> np.ndarray:
        """Decodifica i token audio in forma d'onda waveform float32 a 24kHz."""
        # Estrai array numpy per compatibilità uniforme
        if isinstance(audio_codes, torch.Tensor):
            codes_np = audio_codes.detach().cpu().numpy()
        else:
            codes_np = np.asarray(audio_codes)

        # 1. Percorso Subprocess Ufficiale (se attivo)
        if self._worker_proc is not None:
            import uuid
            uid = uuid.uuid4().hex[:8]
            in_f = f"/tmp/c2w_in_{os.getpid()}_{uid}.npy"
            out_f = f"/tmp/c2w_out_{os.getpid()}_{uid}.npy"
            try:
                np.save(in_f, codes_np)
                req = json.dumps({"input_file": in_f, "output_file": out_f, "normalize": normalize})
                self._worker_proc.stdin.write(req + "\n")
                self._worker_proc.stdin.flush()
                resp = self._read_worker_line(timeout=60.0)
                if resp is not None and resp.get("status") == "ok" and os.path.exists(out_f):
                    wav_arr = np.load(out_f)
                    return wav_arr.astype(np.float32)
            except Exception as e:
                print(f"[Code2Wav] Avviso decodifica worker: {e}, tento riavvio worker...")
                sys.stdout.flush()
                if self._start_worker():
                    try:
                        np.save(in_f, codes_np)
                        req = json.dumps({"input_file": in_f, "output_file": out_f, "normalize": normalize})
                        self._worker_proc.stdin.write(req + "\n")
                        self._worker_proc.stdin.flush()
                        resp = self._read_worker_line(timeout=60.0)
                        if resp is not None and resp.get("status") == "ok" and os.path.exists(out_f):
                            wav_arr = np.load(out_f)
                            return wav_arr.astype(np.float32)
                    except Exception:
                        pass
            finally:
                if os.path.exists(in_f):
                    try: os.remove(in_f)
                    except Exception: pass
                if os.path.exists(out_f):
                    try: os.remove(out_f)
                    except Exception: pass

        # 2. Percorso In-Process (ufficiale in-process, transformers o replica standalone)
        if self.decoder_module is not None:
            try:
                with torch.no_grad():
                    x = torch.from_numpy(codes_np).clone() if isinstance(codes_np, np.ndarray) else audio_codes.clone()
                    if x.dim() == 2:
                        if x.shape[0] == 16 and x.shape[1] != 16:
                            x = x.transpose(0, 1)  # (T, 16)
                        x = x.unsqueeze(0)  # (1, T, 16)
                    elif x.dim() == 3:
                        if x.shape[1] == 16 and x.shape[2] != 16:
                            x = x.transpose(1, 2)  # (B, T, 16)

                    x = torch.clamp(x, min=0, max=2047).long()

                    if self._wrapper_mode in ("official_inprocess", "transformers_automodel") or hasattr(self.decoder_module, "decode"):
                        dev = next(self.decoder_module.parameters()).device
                        out = self.decoder_module.decode(x.to(dev))
                        if hasattr(out, "audio_values"):
                            wav = out.audio_values[0]
                        elif hasattr(out, "audio"):
                            wav = out.audio[0]
                        elif isinstance(out, (list, tuple)):
                            wav = out[0]
                        else:
                            wav = out
                        if isinstance(wav, torch.Tensor):
                            wav_arr = wav.cpu().numpy().squeeze()
                        else:
                            wav_arr = np.array(wav).squeeze()
                    else:
                        wav = self.decoder_module(x.transpose(1, 2).cpu())
                        if isinstance(wav, torch.Tensor):
                            wav_arr = wav.cpu().numpy().squeeze()
                        else:
                            wav_arr = np.array(wav).squeeze()

                    wav_arr = wav_arr.astype(np.float32)
                    if normalize:
                        p_peak = float(np.percentile(np.abs(wav_arr), 99.7))
                        if p_peak > 1e-8:
                            wav_arr = np.clip(wav_arr / (p_peak * 1.05), -1.0, 1.0).astype(np.float32)
                        else:
                            max_val = float(np.max(np.abs(wav_arr)))
                            if max_val > 1e-8:
                                wav_arr = (wav_arr / max_val).astype(np.float32)
                    return wav_arr
            except Exception as e:
                print(f"[Code2Wav] Avviso decodifica: {e}")
                sys.stdout.flush()

        # 3. Fallback sintetico per test
        t_len = codes_np.shape[1] if codes_np.ndim == 3 else (codes_np.shape[0] if codes_np.shape[0] != 16 else codes_np.shape[1])
        num_samples = int(t_len * (24000 / FRAME_RATE_HZ))
        t = np.linspace(0, t_len / FRAME_RATE_HZ, num_samples)
        return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def __del__(self):
        """Cleanup worker process on garbage collection."""
        self._cleanup_worker()



class FullDuplexDataset(Dataset):
    """Dataset full-duplex con input audio NATIVO (mel -> Conformer Gemma 4 E2B) + auto-ascolto.

    Ogni campione è una finestra di `seq_len` frame agente (12.5 fps = 80ms):
      - input_ids: [bos] + [TESTO risposta agente (TEXT_BLOCK_LEN, left-pad)] + [boa] +
        [audio_token_id]*n_feat + [eoa], dove n_feat è il numero di feature
        dell'audio tower (25fps) per la finestra mel (4 per frame agente con
        SELF_LISTENING: 2 canale utente + 2 canale agente). Il blocco testo è pad
        mascherato di default (mai AUDIO_TOKEN_ID: il masked_scatter delle feature
        deve trovare ESATTAMENTE n_feat placeholder). I placeholder audio_token_id
        vengono SOSTITUITI dalle feature continue del Conformer (masked_scatter nel
        forward nativo di Gemma 4): niente codici VQ in ingresso, coerente con il
        design ufficiale E2B. Il testo PRECEDE le feature: causalmente i codici
        audio attendono alla risposta agente (direzione nativa testo->audio).
      - text_targets: next-token sul blocco testo (resp[i+1] a posizione i, l'ultimo
        token predice boa; PAD_TARGET_ID = niente loss) -> supervisione testuale
        tramite la lm_head condivisa del base (tied embeddings).
      - input_features: sequenza mel INTERLEAVATA (mel_window*MEL_CHANNELS, MEL_BINS):
        per ogni frame agente [mel UTENTE x8, mel AGENTE x8] (auto-ascolto: il modello
        sente il proprio parlato passato, con delay di SELF_LISTENING_DELAY_FRAMES).
      - audio_targets: codici RVQ agente (seq_len, N_CODEBOOKS), frame pad = PAD_TARGET_ID
      - frame_weights: pesi di importanza per frame (ricetta dockhardman: amplifica la
        loss su aperture turno, barge-in/co-talk e cessioni di turno).
    """

    def __init__(
        self,
        user_mel: np.ndarray,
        agent_mel: np.ndarray,
        agent_codes: np.ndarray,
        seq_len: int = 64,
        text_by_window: Optional[list] = None,
    ):
        # FIX ANTI-OOM (DDD): i mel restano in FLOAT16 nativo (precisione del dataset,
        # il modello gira comunque in fp16 su GPU). Il cast a float32 avveniva su TUTTO
        # l'array in memoria raddoppiando il footprint (con 50h+MDC: ~14GB solo mel).
        # Il cast float32 avviene ORA per-finestra in __getitem__ (512x128 = 256KB).
        self.user_mel = np.asarray(user_mel, dtype=np.float16)  # (T_mel, MEL_BINS)
        self.agent_mel = np.asarray(agent_mel, dtype=np.float16)  # (T_mel, MEL_BINS)
        self.agent_codes = np.asarray(agent_codes, dtype=np.int16)  # (N_CB, T_agent)
        self.n_code_frames = self.agent_codes.shape[1]
        self.seq_len = seq_len
        self.mel_window = (
            seq_len * MEL_PER_AGENT_FRAME
        )  # 512 frame mel per canale per finestra
        self.mel_channels = MEL_CHANNELS if SELF_LISTENING else 1
        self.input_mel_len = (
            self.mel_window * self.mel_channels
        )  # 1024 (interleaved) o 512
        # Output dell'audio tower: subsampling 4x sul mel (100fps -> 25fps)
        self.n_audio_feat = self.input_mel_len // 4  # 256 feature = 4 per frame agente
        # Sequenza LLM: [bos] + [blocco TESTO (TEXT_BLOCK_LEN)] + [boa] + [feat] + [eoa]
        self.input_len = 3 + self.n_audio_feat + TEXT_BLOCK_LEN
        self.text_by_window = text_by_window  # list per finestra: list[int] o None

        # Statistiche energia mel utente per il rilevamento di attivita' vocale
        # (barge-in/co-talk): soglia robusta p10 + max(bias, 0.5*(p90-p10)).
        # NOTA: con mel in memmap float16, astype(float32) dell'ARRAY INTERO creerebbe
        # una copia da ~6GB: calcoliamo in float16 nativo (precisione sufficiente per
        # una soglia di energia) senza copie.
        e = np.abs(self.user_mel).mean(axis=1)  # (T_mel,) float16
        p10, p90 = np.percentile(e.astype(np.float32), [10, 90])
        self.user_thresh = float(p10 + max(USER_ENERGY_BIAS, 0.5 * (p90 - p10)))

        # Frame agente di silenzio: frazione di codebook al codice di silenzio empirico
        sil = np.asarray(SILENCE_TOKEN_IDS, dtype=np.int64)[:, None]  # (N_CB, 1)
        self.agent_silent = (self.agent_codes == sil).sum(axis=0) >= int(
            AGENT_SILENCE_MATCH_FRAC * N_CODEBOOKS
        )  # (T_agent,) bool

    def __len__(self):
        return max(1, self.n_code_frames // self.seq_len)

    def __getitem__(self, idx):
        c0 = idx * self.seq_len
        c1 = min(c0 + self.seq_len, self.n_code_frames)
        m0 = c0 * MEL_PER_AGENT_FRAME
        m1 = min(m0 + self.mel_window, self.user_mel.shape[0])

        # Finestre mel per canale con zero-pad in coda (silenzio)
        u_win = np.zeros((self.mel_window, MEL_BINS), dtype=np.float32)
        u_win[: m1 - m0] = self.user_mel[m0:m1]
        if SELF_LISTENING:
            a_win = np.zeros((self.mel_window, MEL_BINS), dtype=np.float32)
            a_win[: m1 - m0] = self.agent_mel[m0:m1]
            # Auto-ascolto con delay: il mel agente del blocco t è il parlato del frame
            # t-SELF_LISTENING_DELAY_FRAMES (rotazione); i primi blocchi hanno zero-pad.
            d = SELF_LISTENING_DELAY_FRAMES
            if d > 0:
                a_win = np.concatenate(
                    [
                        np.zeros((d * MEL_PER_AGENT_FRAME, MEL_BINS), dtype=np.float32),
                        a_win[: -d * MEL_PER_AGENT_FRAME]
                        if d * MEL_PER_AGENT_FRAME < self.mel_window
                        else a_win,
                    ],
                    axis=0,
                )
            # Sequenza mel interleavata per frame: [u8, a8, u8, a8, ...]
            u4 = u_win.reshape(self.seq_len, MEL_PER_AGENT_FRAME, MEL_BINS)
            a4 = a_win.reshape(self.seq_len, MEL_PER_AGENT_FRAME, MEL_BINS)
            mel_seq = np.stack([u4, a4], axis=1).reshape(self.input_mel_len, MEL_BINS)
        else:
            mel_seq = u_win

        # Target agente: frame reali + PAD_TARGET_ID sui frame di padding
        targets = np.full((self.seq_len, N_CODEBOOKS), PAD_TARGET_ID, dtype=np.int64)
        n_real = c1 - c0
        if n_real > 0:
            targets[:n_real] = self.agent_codes[:, c0:c1].T

        # Sequenza LLM: [bos, (blocco TESTO risposta agente, left-pad), boa, audio_token_id x n_feat, eoa]
        input_ids = np.full(self.input_len, AUDIO_TOKEN_ID, dtype=np.int64)
        input_ids[0] = BOS_TOKEN_ID
        # IMPORTANTE: il blocco testo è PAD di default (mai AUDIO_TOKEN_ID, altrimenti
        # il masked_scatter delle feature tower troverebbe 48+256 placeholder e
        # fallirebbe il numel check nel forward nativo di Gemma 4).
        input_ids[1 : 1 + TEXT_BLOCK_LEN] = E2B_PAD_TOKEN_ID
        input_ids[1 + TEXT_BLOCK_LEN] = BOA_TOKEN_ID
        input_ids[-1] = EOA_TOKEN_ID
        attention_mask = np.ones(self.input_len, dtype=np.int64)
        attention_mask[1 : 1 + TEXT_BLOCK_LEN] = 0  # pad testo mascherati di default

        # ---- Blocco TESTO (supervisione testuale): il testo della risposta agente ----
        # precede le feature audio: causalmente i codici audio ATTENDONO al testo
        # (direzione nativa E2B testo->audio). Target: next-token sul blocco
        # (resp[i+1] predetto alla posizione del token i-esimo; l'ultimo token del
        # testo predice boa; PAD_TARGET_ID sui padding -> niente loss).
        text_targets = np.full(TEXT_BLOCK_LEN, PAD_TARGET_ID, dtype=np.int64)
        t_ids = self.text_by_window[idx] if self.text_by_window is not None else None
        if t_ids:
            K = min(len(t_ids), TEXT_BLOCK_LEN)
            start_b = (
                TEXT_BLOCK_LEN - K
            )  # il testo occupa [start_b, TEXT_BLOCK_LEN) del blocco
            input_ids[1 + start_b : 1 + TEXT_BLOCK_LEN] = t_ids[:K]
            attention_mask[1 + start_b : 1 + TEXT_BLOCK_LEN] = (
                1  # sblocca i token testo
            )
            resp = list(t_ids[:K]) + [BOA_TOKEN_ID]
            p0 = start_b - 1  # primo block pos con loss (l'ultimo pad predice text[0])
            for i in range(K + 1):
                text_targets[p0 + i] = resp[i]

        # Frame-level importance weighting (ricetta dockhardman):
        #  - OPEN: agente silenzio -> parlato (apertura turno)                 [peso alto]
        #  - CO_TALK: agente parlato E utente attivo (barge-in/backchannel)
        #  - YIELD: agente smette di parlare mentre l'utente è attivo (cessione)
        fw = np.ones(self.seq_len, dtype=np.float32)
        if USE_FRAME_WEIGHTING and n_real > 0:
            sil = self.agent_silent[c0:c1]
            ue = (
                np.abs(u_win)
                .mean(axis=1)
                .reshape(self.seq_len, MEL_PER_AGENT_FRAME)
                .max(axis=1)
                > self.user_thresh
            )
            for t in range(n_real):
                sp = not bool(sil[t])
                if sp:
                    if t == 0 or bool(sil[t - 1]):
                        fw[t] = max(fw[t], FRAME_WEIGHT_OPEN)
                    if bool(ue[t]):
                        fw[t] = max(fw[t], FRAME_WEIGHT_CO_TALK)
                elif t > 0 and not bool(sil[t - 1]) and bool(ue[t]):
                    fw[t] = max(fw[t], FRAME_WEIGHT_YIELD)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "input_features": torch.tensor(mel_seq, dtype=torch.float32),
            "input_features_mask": torch.ones(self.input_mel_len, dtype=torch.bool),
            "audio_targets": torch.tensor(targets, dtype=torch.long),
            "text_targets": torch.tensor(text_targets, dtype=torch.long),
            "frame_weights": torch.tensor(fw, dtype=torch.float32),
        }


# %% [code]
# Cell 6: Definizione Modello Gemma4FullDuplexModel


class ResidualAdapter(nn.Module):
    """Residual Latent Adapter for h_t prediction with skip connection and LayerNorm."""

    def __init__(self, in_features: int = 1536, out_features: int = 2048, hidden_dim: int = 3072):
        super().__init__()
        self.skip = nn.Linear(in_features, out_features)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, out_features),
        )
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.skip(x) + self.mlp(x))


class DeepSemanticHead(nn.Module):
    """Deep Semantic Head for CB0 codebook prediction (2048 classes) with LayerNorm and SiLU."""

    def __init__(self, in_features: int = 1536, hidden_dim: int = 2048, vocab_size: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Gemma4FullDuplexModel(nn.Module):
    """
    Architettura Full-Duplex Gemma-4 con COUPLING Qwen3-TTS (ADR-005) — unico percorso.
    Convoluzione Causale 1D + head semantica cb0 + adapter h_t + sub-talker congelato
    per i codebook acustici 1..15 (legacy MTPModule rimosso).
    """

    def __init__(self, base_model, hidden_size: int = 3072):
        super().__init__()
        self.base_model = base_model
        self.causal_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=0)
        # Head semantica sul codebook 0 (vocabolario SEMANTIC_VOCAB_SIZE, es. 4096)
        self.semantic_head = DeepSemanticHead(hidden_size, hidden_dim=2048, vocab_size=2048)

        # ---- Coupling Qwen3-TTS (ADR-005): UNICO percorso - MTP legacy RIMOSSO ----
        # Il sub-talker è OBBLIGATORIO: se la dir manca (Fase 1 senza
        # extract_qwen_subtalker.py) si fallisce con un errore chiaro.
        self.coupling_step = 0  # contatore step di training (warm-up h_t.detach())
        from qwen_subtalker import load_qwen_subtalker

        self.subtalker, _st_cfg = load_qwen_subtalker(QWEN_SUBTALKER_DIR)
        # Il sub-talker (congelato) va sul device del modello base: senza questo
        # resta su CPU mentre Gemma è su cuda (crash embedding device mismatch).
        st_dev = next(self.subtalker.parameters()).device
        base_dev = next(base_model.parameters()).device
        if st_dev != base_dev:
            self.subtalker.to(base_dev)
        self.ht_adapter = ResidualAdapter(
            hidden_size, out_features=COUPLING_TALKER_HIDDEN, hidden_dim=3072
        )
        print(
            f"[Coupling] ✅ SUB-TALKER Qwen3-TTS caricato e congelato ({QWEN_SUBTALKER_DIR}). "
            "I codebook acustici 1..15 sono generati dal sub-talker (qualità nativa Qwen); "
            "Gemma impara h_t (adapter) + cb0."
        )
        sys.stdout.flush()
        # lm_head condivisa del modello base (tied embeddings) per la loss TESTO:
        # viene cercata al primo forward (i path dipendono dall'impacchettamento
        # unsloth/PEFT); se assente la loss testo viene disattivata con un avviso.
        self._text_head = None
        self.text_head_checked = False

    def _apply_causal_conv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Helper privato per applicare la Convoluzione Causale 1D con Left-Padding (2,0) e allineamento Multi-GPU."""
        device = hidden_states.device
        if self.causal_conv.weight.device != device:
            self.causal_conv.to(device)

        hidden_conv = hidden_states.transpose(1, 2)
        hidden_padded = F.pad(hidden_conv, (2, 0))
        return self.causal_conv(
            hidden_padded.to(self.causal_conv.weight.dtype)
        ).transpose(1, 2)

    def _get_text_head(self) -> Optional[nn.Module]:
        """Trova la lm_head del modello base (tied embeddings) per la loss testo.
        Path dipendenti dal wrapping unsloth/PEFT: fallback a cascata."""
        if self.text_head_checked:
            return self._text_head
        self.text_head_checked = True
        base = self.base_model
        candidates = [
            ("language_model", "lm_head"),
            ("model", "language_model", "lm_head"),
            ("base_model", "language_model", "lm_head"),
            ("model", "lm_head"),
            ("lm_head",),
        ]
        for path in candidates:
            obj = base
            ok = True
            for a in path:
                if not hasattr(obj, a):
                    ok = False
                    break
                obj = getattr(obj, a)
            if ok and hasattr(obj, "weight") and hasattr(obj, "forward"):
                self._text_head = obj
                print(
                    f"[Fase 2 - Testo] lm_head trovata: {'.'.join(path)} (loss testo ATTIVA, "
                    f"vocab {obj.weight.shape[0]})."
                )
                sys.stdout.flush()
                return obj
        print(
            "[Fase 2 - Testo] ⚠️ lm_head non trovata: training SOLO audio (testo disattivato)."
        )
        sys.stdout.flush()
        return None

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        audio_targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        frame_weights: Optional[torch.Tensor] = None,
        text_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Allineamento difensivo degli input al device del modello base:
        # evita mismatch device CPU/GPU (es. "Unhandled FakeTensor Device
        # Propagation for aten.embedding" durante il tracing di torch.compile).
        base_device = next(self.base_model.parameters()).device
        if input_ids is not None and input_ids.device != base_device:
            input_ids = input_ids.to(base_device)
        if input_features is not None and input_features.device != base_device:
            input_features = input_features.to(base_device)
        if (
            input_features_mask is not None
            and input_features_mask.device != base_device
        ):
            input_features_mask = input_features_mask.to(base_device)
        if audio_targets is not None and audio_targets.device != base_device:
            audio_targets = audio_targets.to(base_device)
        if attention_mask is not None and attention_mask.device != base_device:
            attention_mask = attention_mask.to(base_device)
        if frame_weights is not None and frame_weights.device != base_device:
            frame_weights = frame_weights.to(base_device)
        if text_targets is not None and text_targets.device != base_device:
            text_targets = text_targets.to(base_device)

        kwargs = {
            "output_hidden_states": True,
            "return_dict": True,
            "logits_to_keep": 1,  # evitiamo il costo dell'lm_head su tutta la sequenza
        }
        if input_ids is not None:
            kwargs["input_ids"] = input_ids
        if input_features is not None and input_features_mask is not None:
            kwargs["input_features"] = input_features
            kwargs["input_features_mask"] = input_features_mask
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        outputs = self.base_model(**kwargs)

        hidden_states = outputs.hidden_states[-1]
        h_device = hidden_states.device

        # Allineamento temporale NATIVO + AUTO-ASCOLTO: l'audio tower emette feature a
        # 25fps (4 per frame agente a 12.5fps con SELF_LISTENING: 2 canale utente + 2
        # canale agente). Sequenza LLM: [bos, (TESTO), boa, feat0..feat_{F-1}, eoa] dove
        # il blocco del frame t è [u0,u1,a0,a1] -> il frame agente t viene predetto
        # dall'ULTIMA feature del blocco (posizione LLM AUDIO_FEAT_START+FPF*(t+1)-1),
        # causale: ha visto il mel utente fino a t, il proprio parlato fino a t-delay
        # (poi il conv 1D) e l'INTERO blocco TESTO (direzione testo->audio nativa E2B).
        fpf = FEATURES_PER_AGENT_FRAME if SELF_LISTENING else AGENT_FEATURES_PER_FRAME
        audio_feat_start = 2 + TEXT_BLOCK_LEN  # bos + blocco testo + boa
        if audio_targets is not None:
            n_agent = audio_targets.shape[1]
        else:
            n_agent = (hidden_states.shape[1] - (3 + TEXT_BLOCK_LEN)) // fpf
        agent_hidden = hidden_states[
            :, audio_feat_start + fpf - 1 : audio_feat_start + fpf * n_agent : fpf, :
        ]  # (B, n_agent, H)

        # Allineamento automatico delle teste al device di output di Gemma
        if next(self.semantic_head.parameters()).device != h_device:
            self.semantic_head.to(h_device)

        adapted_hidden = self._apply_causal_conv(agent_hidden)
        cb0_logits = self.semantic_head(adapted_hidden)

        # ---- Coupling (ADR-005): i codebook acustici 1..15 dal sub-talker Qwen congelato ----
        # h_t = ht_adapter(adapted_hidden). Durante il warm-up (COUPLING_WARMUP_STEPS)
        # h_t è detached a MONTE dell'adapter: il gradiente della coupling loss arriva
        # solo all'adapter (che impara a mappare gli hidden di Gemma congelata); dopo il
        # warm-up è end-to-end e il gradiente raggiunge anche la LoRA di Gemma.
        coupling_loss = None
        if audio_targets is not None:
            if next(self.ht_adapter.parameters()).device != h_device:
                self.ht_adapter.to(h_device)
            if next(self.subtalker.parameters()).device != h_device:
                self.subtalker.to(h_device)
            ad_dtype = next(self.ht_adapter.parameters()).dtype
            h_src = (
                adapted_hidden.detach()
                if (self.coupling_step < COUPLING_WARMUP_STEPS and self.training)
                else adapted_hidden
            )
            h_t = self.ht_adapter(h_src.to(ad_dtype))
            if h_t.dtype != next(self.subtalker.parameters()).dtype:
                h_t = h_t.to(next(self.subtalker.parameters()).dtype)
            sub_logits = self.subtalker.forward_sub_talker_finetune(h_t, audio_targets)
            coupling_loss = self.subtalker.coupling_loss(
                sub_logits, audio_targets, frame_weights
            )
            coupling_loss = coupling_loss * COUPLING_LOSS_WEIGHT
        # NOTA: cb0 usa SEMANTIC_VOCAB_SIZE (4096), i codebook 1..15 ACOUSTIC_VOCAB_SIZE
        # (2048): la loss li consuma separatamente (non concaténali sul vocabolario).

        # ---- Loss TESTO (supervisione testuale sui token BPE della risposta agente) ----
        # Usa la lm_head condivisa del base (tied embeddings): logits sul blocco testo
        # [1, 1+TEXT_BLOCK_LEN) della sequenza. PAD_TARGET_ID sui padding -> niente loss.
        text_loss = None
        if text_targets is not None:
            text_head = self._get_text_head()
            if text_head is not None:
                text_hidden = hidden_states[:, 1 : 1 + TEXT_BLOCK_LEN, :].to(
                    text_head.weight.dtype
                )
                text_logits = text_head(text_hidden)  # (B, TEXT_BLOCK_LEN, text_vocab)
                if text_logits.device != text_targets.device:
                    text_targets = text_targets.to(text_logits.device)
                t_flat = text_logits.reshape(-1, text_logits.shape[-1])
                t_tgt = text_targets.reshape(-1)
                valid_t = (t_tgt != PAD_TARGET_ID).float()
                if valid_t.sum() > 0:
                    per_tok_t = F.cross_entropy(
                        t_flat, t_tgt, reduction="none", ignore_index=PAD_TARGET_ID
                    )
                    text_loss = (
                        (per_tok_t * valid_t).sum()
                        / valid_t.sum().clamp(min=1)
                        * TEXT_LOSS_WEIGHT
                    )
                else:
                    text_loss = torch.zeros((), device=h_device, dtype=cb0_logits.dtype)

        loss = None
        if audio_targets is not None:
            if audio_targets.device != h_device:
                audio_targets = audio_targets.to(h_device)
            # Coupling: loss = cb0 (semantica) + coupling (sub-talker su cb1..cb15)
            loss = self.compute_discoupled_loss(
                cb0_logits, audio_targets, frame_weights
            )
            if coupling_loss is not None:
                loss = loss + coupling_loss
            if text_loss is not None:
                loss = loss + text_loss

        return {
            "loss": loss,
            "text_loss": text_loss,
            "coupling_loss": coupling_loss,
            "logits": getattr(outputs, "logits", None),
        }

    def compute_discoupled_loss(
        self,
        cb0_logits: torch.Tensor,
        audio_targets: torch.Tensor,
        frame_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Loss semantica sul codebook 0 (cb0) con pesatura per frame.

        `frame_weights` amplifica le posizioni decisive (apertura turno, barge-in,
        backchannel, cessione) — ricetta dockhardman: il segnale "act now" è una
        frazione minuscola dei frame supervisionati e va amplificato. I frame di
        padding (PAD_TARGET_ID) sono ignorati e non entrano nel denominatore.
        I codebook acustici 1..15 NON hanno loss qui: sono supervisionati dal
        sub-talker Qwen congelato (coupling_loss, ADR-005).
        """
        cb0_flat = cb0_logits.reshape(-1, SEMANTIC_VOCAB_SIZE)
        cb0_targets = audio_targets[:, :, 0].reshape(-1)
        per_frame_s = F.cross_entropy(
            cb0_flat, cb0_targets, reduction="none", ignore_index=PAD_TARGET_ID
        )  # (B*T,)
        if frame_weights is not None:
            fw = frame_weights.reshape(-1)
            if fw.device != per_frame_s.device:
                fw = fw.to(per_frame_s.device)
            per_frame_s = per_frame_s * fw
        valid_s = (cb0_targets != PAD_TARGET_ID).float()
        return (per_frame_s * valid_s).sum() / valid_s.sum().clamp(min=1)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        speaker_ref_codes: Optional[torch.Tensor] = None,
        max_new_tokens: int = 250,
        temperature: float = 0.7,
        top_p: float = 0.9,
        past_key_values: Optional[Any] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Genera token acustici (B, T, 15) supportando la KV-cache.

        Supporta il percorso audio NATIVO: passare input_ids già contenente i
        placeholder audio_token_id (vedi FullDuplexDataset) + input_features mel.
        Il loopback autoregressivo dell'agente resta sul percorso discreto legacy.
        """
        self.eval()
        device = next(self.base_model.parameters()).device
        if input_ids.device != device:
            input_ids = input_ids.to(device)

        kwargs = {
            "input_ids": input_ids,
            "output_hidden_states": True,
            "return_dict": True,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask.to(device)
        if input_features is not None:
            kwargs["input_features"] = input_features.to(device)
            if input_features_mask is not None:
                kwargs["input_features_mask"] = input_features_mask.to(device)

        outputs = self.base_model(**kwargs)

        hidden_states = outputs.hidden_states[-1]
        new_past_key_values = getattr(outputs, "past_key_values", None)

        if input_features is not None:
            # Percorso NATIVO (+ auto-ascolto): allinea le feature audio (25fps, 4 per
            # frame agente) ai frame agente (12.5fps) come nel forward di training.
            fpf = (
                FEATURES_PER_AGENT_FRAME if SELF_LISTENING else AGENT_FEATURES_PER_FRAME
            )
            n_agent = (hidden_states.shape[1] - (3 + TEXT_BLOCK_LEN)) // fpf
            audio_feat_start = 2 + TEXT_BLOCK_LEN
            hidden_states = hidden_states[
                :,
                audio_feat_start + fpf - 1 : audio_feat_start + fpf * n_agent : fpf,
                :,
            ]

        adapted_hidden = self._apply_causal_conv(hidden_states)
        # Device-check esplicito per le teste custom: il checkpoint le ricarica su CPU
        # e il forward() le sposta in base a h_device, ma generate() NON passava da
        # forward() -> RuntimeError mat2 on cpu / mat1 on cuda (bug diagnosi v6).
        if next(self.semantic_head.parameters()).device != adapted_hidden.device:
            self.semantic_head.to(adapted_hidden.device)
        cb0_logits = self.semantic_head(adapted_hidden)
        cb0_sampled = (
            torch.argmax(cb0_logits, dim=-1)
            if temperature == 0
            else torch.multinomial(
                F.softmax(cb0_logits / temperature, dim=-1).view(
                    -1, cb0_logits.shape[-1]
                ),
                1,
            ).view(cb0_logits.shape[:-1])
        )
        # Clamp nel range semantico GT [0, ACOUSTIC_VOCAB_SIZE): il semantic_head ha
        # vocab 4096 ma il decoder/quantizer (e il sub-talker) hanno codebook 2048.
        cb0_sampled = cb0_sampled.clamp(max=ACOUSTIC_VOCAB_SIZE - 1)

        # Coupling (ADR-005): i codebook acustici 1..15 dal sub-talker Qwen congelato
        # (catena RVQ a 15 passi; voce/qualità nativa del teacher).
        if next(self.ht_adapter.parameters()).device != adapted_hidden.device:
            self.ht_adapter.to(adapted_hidden.device)
        if next(self.subtalker.parameters()).device != adapted_hidden.device:
            self.subtalker.to(adapted_hidden.device)
        ad_dtype = next(self.ht_adapter.parameters()).dtype
        h_t = self.ht_adapter(adapted_hidden.to(ad_dtype))
        h_t = h_t.to(next(self.subtalker.parameters()).dtype)
        acust_sampled = self.subtalker.sample_acoustic_codes(
            h_t, cb0_sampled, temperature
        )

        generated_audio_codes = torch.cat(
            [cb0_sampled.unsqueeze(-1), acust_sampled], dim=-1
        )

        if speaker_ref_codes is not None:
            if speaker_ref_codes.dim() == 2:
                speaker_ref_codes = speaker_ref_codes.transpose(0, 1).unsqueeze(0)
            if speaker_ref_codes.device != generated_audio_codes.device:
                speaker_ref_codes = speaker_ref_codes.to(generated_audio_codes.device)
            if speaker_ref_codes.dtype != generated_audio_codes.dtype:
                speaker_ref_codes = speaker_ref_codes.to(generated_audio_codes.dtype)
            generated_audio_codes = torch.cat(
                [speaker_ref_codes, generated_audio_codes], dim=1
            )

        return {
            "audio_codes": generated_audio_codes,
            "past_key_values": new_past_key_values,
        }

    @torch.inference_mode()
    def stream_speech(
        self,
        input_ids: torch.Tensor,
        user_audio_stream: Optional[torch.Tensor] = None,
        speaker_ref_codes: Optional[torch.Tensor] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 250,
    ):
        """Generatore iterativo per lo streaming Full-Duplex Dual-Channel frame-by-frame."""
        self.eval()
        device = next(self.base_model.parameters()).device
        if input_ids.device != device:
            input_ids = input_ids.to(device)

        batch_size = input_ids.shape[0]
        hidden_size = self.causal_conv.in_channels
        conv_state = torch.zeros(
            batch_size,
            hidden_size,
            2,
            device=device,
            dtype=self.causal_conv.weight.dtype,
        )
        past_key_values = None

        outputs = self.base_model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
        )
        hidden_states = outputs.hidden_states[-1]
        past_key_values = getattr(outputs, "past_key_values", None)

        if hidden_states.shape[1] >= 2:
            conv_state = hidden_states[:, -2:, :].transpose(1, 2)
        elif hidden_states.shape[1] == 1:
            conv_state[:, :, -1:] = hidden_states[:, -1:, :].transpose(1, 2)

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)

        if speaker_ref_codes is not None:
            if speaker_ref_codes.dim() == 2:
                speaker_ref_codes = speaker_ref_codes.transpose(0, 1).unsqueeze(0)
            if speaker_ref_codes.device != device:
                speaker_ref_codes = speaker_ref_codes.to(device)
            for t in range(speaker_ref_codes.shape[1]):
                yield speaker_ref_codes[:, t : t + 1, :]

        for step in range(max_new_tokens):
            if user_audio_stream is not None and step < user_audio_stream.shape[1]:
                user_cb0 = user_audio_stream[:, step, 0].to(torch.long)
                next_token = (
                    (user_cb0 + USER_AUDIO_TOKEN_OFFSET).unsqueeze(-1).to(device)
                )

            outputs = self.base_model(
                input_ids=next_token,
                output_hidden_states=True,
                return_dict=True,
                past_key_values=past_key_values,
                use_cache=True,
            )
            hidden_frame = outputs.hidden_states[-1]
            past_key_values = getattr(outputs, "past_key_values", None)

            hidden_frame_t = hidden_frame.transpose(1, 2)
            conv_input = torch.cat([conv_state, hidden_frame_t], dim=2)
            conv_state = conv_input[:, :, 1:]

            adapted_hidden = self.causal_conv(conv_input).transpose(1, 2)
            cb0_logits = self.semantic_head(adapted_hidden)
            cb0_sampled = (
                torch.argmax(cb0_logits, dim=-1)
                if temperature == 0
                else torch.multinomial(
                    F.softmax(cb0_logits / temperature, dim=-1).view(
                        -1, cb0_logits.shape[-1]
                    ),
                    1,
                ).view(cb0_logits.shape[:-1])
            )
            # Clamp nel range semantico GT (codebook 2048) per la decodifica (vedi generate)
            cb0_sampled = cb0_sampled.clamp(max=ACOUSTIC_VOCAB_SIZE - 1)

            # Coupling (ADR-005): i codebook acustici 1..15 dal sub-talker Qwen congelato
            if next(self.ht_adapter.parameters()).device != adapted_hidden.device:
                self.ht_adapter.to(adapted_hidden.device)
            if next(self.subtalker.parameters()).device != adapted_hidden.device:
                self.subtalker.to(adapted_hidden.device)
            ad_dtype = next(self.ht_adapter.parameters()).dtype
            h_t = self.ht_adapter(adapted_hidden.to(ad_dtype))
            h_t = h_t.to(next(self.subtalker.parameters()).dtype)
            acust_sampled = self.subtalker.sample_acoustic_codes(
                h_t, cb0_sampled, temperature
            )

            if user_audio_stream is None or step >= user_audio_stream.shape[1]:
                next_token = (cb0_sampled[:, -1:] + USER_AUDIO_TOKEN_OFFSET).to(device)

            yield torch.cat([cb0_sampled.unsqueeze(-1), acust_sampled], dim=-1)


# %% [code]
# Cell 7: Caricamento Unsloth, Training Loop & Saving Checkpoint
def load_gemma4_duplex_system():
    """Carica Gemma 4 via Unsloth e inizializza il wrapper Full-Duplex (COUPLING, ADR-005).

    Il sub-talker Qwen3-TTS è OBBLIGATORIO: va estratto in Fase 1 con
    extract_qwen_subtalker.py (dir QWEN_SUBTALKER_DIR). Se manca -> errore chiaro
    (l'architettura legacy MTP è stata rimossa).
    """
    if not os.path.isdir(QWEN_SUBTALKER_DIR):
        raise RuntimeError(
            f"Sub-talker Qwen3-TTS non trovato in '{QWEN_SUBTALKER_DIR}'. "
            "L'architettura supportata è SOLO il coupling (ADR-005): esegui in Fase 1 "
            "`!python extract_qwen_subtalker.py --out-dir qwen_subtalker` (serve il secret HF_TOKEN)."
        )
    try:
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.set_device(0)

        from unsloth import FastLanguageModel

        print(f"[Fase 2 - Unsloth] Caricamento Unsloth Gemma 4 E2B ({MODEL_NAME})...")
        print(
            "[Fase 2 - Unsloth] NOTA: Assicurati che il kernel Kaggle sia stato RIAVVIATO dopo la Fase 1."
        )
        sys.stdout.flush()

        # FIX SINGLE-GPU (Source: issue unsloth #3389/#3434 + guida ufficiale Gemma 4):
        # Su Kaggle 2xT4 device_map="balanced" causa "CUDA error: an illegal memory access"
        # durante il dispatch multi-GPU e la quantizzazione 4-bit. Gemma 4 E2B 4-bit
        # occupa ~3-4GB di VRAM -> sta su una singola T4 (16GB): niente split necessario.
        # Con CUDA_VISIBLE_DEVICES="0" (vedi header del file) è visibile 1 sola GPU;
        # forziamo comunque "cuda:0" esplicito per robustezza.
        # Se in futuro servirà davvero il multi-GPU: usare torchrun/DDP (guida ufficiale)
        # e NON device_map="balanced" su modelli E2B/E4B.
        if DEVICE == "cuda":
            target_device_map = "cuda:0"
            print(
                f"[Fase 2 - Unsloth] Configurazione single-GPU: target_device_map='{target_device_map}' (multi-GPU Kaggle 2xT4 non stabile)."
            )
        else:
            target_device_map = None

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
            low_cpu_mem_usage=True,
            device_map=target_device_map,
        )

        print(
            "[Fase 2 - Unsloth] Inizializzazione adattatori QLoRA su target modules..."
        )
        sys.stdout.flush()
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
        )

        # FIX dtype audio tower (T4: bf16 non supportato -> unsloth carica gemma4 in
        # fp32 ma i LayerNorm del checkpoint restano fp16 => modello misto). In
        # eval/inference il layer_norm patched di unsloth chiama torch.layer_norm che
        # richiede dtype allineati: input fp32 (dal conv) + weight fp16 -> RuntimeError
        # "expected scalar type Half but found Float" (crash validazione). I LayerNorm
        # non sono MAI quantizzati: allinearli a fp32 è sicuro e risolve il mismatch.
        import torch.nn as _nn_mod

        for _name, _mod in model.named_modules():
            if (
                "norm" in type(_mod).__name__.lower()
                and getattr(_mod, "weight", None) is not None
                and _mod.weight.dtype != torch.float32
            ):
                _mod.to(torch.float32)

        # FIX layer_norm patched di unsloth (unsloth_zoo/patch_torch_functions.py:39-63):
        # il path COMPILED chiama torch.layer_norm SENZA castare l'input al dtype del
        # weight; in inference su modelli a dtype misto (conv fp32 + LayerNorm fp16)
        # -> RuntimeError "expected scalar type Half but found Float". Il path EAGER
        # di unsloth invece casta (unsloth_zoo/compiler.py:3866-3876: "Eager
        # F.layer_norm needs input dtype == weight dtype: cast in and out. Compiled
        # path left untouched (adding the cast there changes batched numerics)").
        # Replica quel comportamento col cast condizionale input->weight.dtype +
        # ripristino in uscita: no-op quando i dtype sono già allineati (training),
        # quindi numerics invariati; serve solo in inference con modello misto.
        import torch.nn.functional as _F
        import torch as _torch

        def _layer_norm_dtype_safe(
            _input,
            _normalized_shape,
            _weight=None,
            _bias=None,
            _eps=1e-5,
        ):
            _orig = _input.dtype
            if _weight is not None and _input.dtype != _weight.dtype:
                _input = _input.to(_weight.dtype)
            return _torch.layer_norm(
                _input,
                _normalized_shape,
                _weight,
                _bias,
                _eps,
                _torch.backends.cudnn.enabled,
            ).to(_orig)

        _F.layer_norm = _layer_norm_dtype_safe

        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None and hasattr(model.config, "text_config"):
            hidden_size = getattr(model.config.text_config, "hidden_size", 3072)
        if hidden_size is None:
            hidden_size = 3072

        # Token speciali audio dal config reale del modello (design nativo E2B)
        global AUDIO_TOKEN_ID, BOA_TOKEN_ID, EOA_TOKEN_ID, BOS_TOKEN_ID
        cfg = getattr(model, "config", None)
        if cfg is not None:
            AUDIO_TOKEN_ID = int(getattr(cfg, "audio_token_id", AUDIO_TOKEN_ID))
            BOA_TOKEN_ID = int(getattr(cfg, "boa_token_id", BOA_TOKEN_ID))
            EOA_TOKEN_ID = int(getattr(cfg, "eoa_token_id", EOA_TOKEN_ID))
            bos = getattr(cfg, "bos_token_id", None)
            if bos is None and getattr(cfg, "text_config", None) is not None:
                bos = getattr(cfg.text_config, "bos_token_id", None)
            if bos is not None:
                BOS_TOKEN_ID = int(bos)
            print(
                f"[Fase 2 - Unsloth] Token audio nativi dal config: audio={AUDIO_TOKEN_ID}, "
                f"boa={BOA_TOKEN_ID}, eoa={EOA_TOKEN_ID}, bos={BOS_TOKEN_ID}."
            )
            sys.stdout.flush()

        print(
            f"[Fase 2 - Unsloth] Inizializzazione Gemma4FullDuplexModel (Hidden Size: {hidden_size})..."
        )
        sys.stdout.flush()
        duplex_model = Gemma4FullDuplexModel(model, hidden_size=hidden_size)

        return duplex_model, tokenizer
    except Exception as e:
        print(f"[Fase 2 - Unsloth] ERRORE CARICAMENTO UNSLOTH ({e}).")
        if DEVICE == "cuda":
            print(
                "[Fase 2 - Unsloth] ERRORE CRITICO SU GPU: Il caricamento di Gemma 4 è fallito su CUDA."
            )
            print(
                "[Fase 2 - Unsloth] Controlla che: 1) Il Kernel Kaggle sia stato riavviato dopo la Fase 1, 2) device_map sia 'cuda:0', 3) unsloth sia aggiornato."
            )
        else:
            print("[Fase 2 - Unsloth] Modalità mock attiva per test locali su CPU.")
        sys.stdout.flush()
        return None, None


def save_gemma4_duplex_checkpoint(
    duplex_model, output_dir: str = "checkpoints/gemma4_duplex_sota"
):
    """Salva gli adattatori LoRA Unsloth e i pesi custom delle teste acustiche."""
    if duplex_model is None:
        print("[Fase 2 - Checkpoint] Nessun modello da salvare (Modalità mock).")
        sys.stdout.flush()
        return

    print(
        f"[Fase 2 - Checkpoint] Salvataggio adattatori LoRA e pesi custom su: {output_dir}..."
    )
    sys.stdout.flush()
    os.makedirs(output_dir, exist_ok=True)
    if hasattr(duplex_model, "base_model") and hasattr(
        duplex_model.base_model, "save_pretrained"
    ):
        duplex_model.base_model.save_pretrained(
            os.path.join(output_dir, "lora_adapters")
        )

    custom_heads_state = {
        "causal_conv": duplex_model.causal_conv.state_dict(),
        "semantic_head": duplex_model.semantic_head.state_dict(),
        "ht_adapter": duplex_model.ht_adapter.state_dict()
        if duplex_model.ht_adapter is not None
        else None,
        "coupling_step": getattr(duplex_model, "coupling_step", 0),
    }
    torch.save(custom_heads_state, os.path.join(output_dir, "custom_duplex_heads.pt"))
    print(
        f"[Fase 2 - Checkpoint] Coupling (ADR-005): adapter h_t incluso nel checkpoint "
        f"(sub-talker NON salvato: congelato, si ricarica da {QWEN_SUBTALKER_DIR})."
    )
    print(
        f"[Fase 2 - Checkpoint] Checkpoint unificato salvato con successo in: {output_dir}"
    )
    sys.stdout.flush()


def load_gemma4_duplex_checkpoint(
    duplex_model, checkpoint_dir: str = "checkpoints/gemma4_duplex_sota"
):
    """Ricarica LoRA + teste custom da un checkpoint salvato (per demo/ascolto senza riallenare)."""
    if duplex_model is None:
        print("[Fase 2 - Checkpoint] Nessun modello da ricaricare (Modalità mock).")
        sys.stdout.flush()
        return
    lora_dir = os.path.join(checkpoint_dir, "lora_adapters")
    if os.path.isdir(lora_dir):
        try:
            from unsloth import FastLanguageModel

            if hasattr(FastLanguageModel, "load_lora"):
                FastLanguageModel.load_lora(duplex_model.base_model, lora_dir)
            else:
                # unsloth 2026.8.x: API rimossa load_lora -> PEFT load_adapter diretto
                # (FIX DDD: verificato su unsloth 2026.8.9 + transformers 5.5.0, peft 0.19.1
                # firma: load_adapter(model_id, adapter_name)).
                if hasattr(duplex_model.base_model, "load_adapter"):
                    duplex_model.base_model.load_adapter(lora_dir, "default")
                else:
                    from peft import PeftModel, set_peft_model_state_dict

                    if isinstance(duplex_model.base_model, PeftModel):
                        duplex_model.base_model.load_adapter(lora_dir, "default")
                    else:
                        set_peft_model_state_dict(
                            duplex_model.base_model, lora_dir, adapter_name="default"
                        )
            print(f"[Fase 2 - Checkpoint] LoRA ricaricata da {lora_dir}.")
            sys.stdout.flush()
        except Exception as e:
            print(f"[Fase 2 - Checkpoint] ⚠️ Errore caricamento LoRA ({e}).")
            sys.stdout.flush()
    heads_path = os.path.join(checkpoint_dir, "custom_duplex_heads.pt")
    if os.path.isfile(heads_path):
        heads = torch.load(heads_path, map_location="cpu", weights_only=True)
        for name, mod in (
            ("causal_conv", duplex_model.causal_conv),
            ("semantic_head", duplex_model.semantic_head),
            ("ht_adapter", duplex_model.ht_adapter),
        ):
            if mod is None:
                continue
            if name in heads and heads[name] is not None:
                mod.load_state_dict(heads[name])
        duplex_model.coupling_step = int(heads.get("coupling_step", 0))
        print(f"[Fase 2 - Checkpoint] Teste custom ricaricate da {heads_path}.")
        sys.stdout.flush()
    else:
        print(
            f"[Fase 2 - Checkpoint] ⚠️ {heads_path} non trovato: teste custom ai valori iniziali."
        )
        sys.stdout.flush()


def _resample_np(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample numpy puro (FFT) senza dipendenze esterne.

    Necessario in Fase 2: dopo il riavvio del kernel Kaggle librosa NON è
    garantito (la Fase 2 installa solo transformers>=5 + unsloth). Per clip
    demo brevi il resample FFT è più che sufficiente e azzera le dipendenze.
    """
    if orig_sr == target_sr:
        return x.astype(np.float32)
    n_in = int(x.shape[0])
    n_out = max(1, int(round(n_in * target_sr / orig_sr)))
    X = np.fft.rfft(x.astype(np.float32))
    Y = np.zeros(n_out // 2 + 1, dtype=np.complex64)
    k = min(X.shape[0], Y.shape[0])
    Y[:k] = X[:k]
    y = np.fft.irfft(Y, n=n_out).real.astype(np.float32)
    return y * (
        float(n_out) / float(n_in)
    )  # irfft normalizza per n_out: scala per preservare l'ampiezza


def _wav_to_mel_l1(x: np.ndarray, sr: int, n_fft: int = 512, n_mels: int = 80) -> np.ndarray:
    """Mel-spettrogramma log (L1-friendly) da wav, numpy puro (senza librosa).

    Usato per la metrica di qualità sintesi: confronta generated vs oracle.
    Filtro mel approssimato (triangoli linear-spaced su mel) — sufficiente per
    una distanza relativa tra due versioni dello stesso segmento.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    hop = n_fft // 4
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop
    mel = np.zeros((n_frames, n_mels), dtype=np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mel_f = 2595.0 * np.log10(1.0 + np.linspace(0, sr / 2, n_mels + 2) / 700.0)
    mel_bins = 2595.0 * np.log10(1.0 + freqs / 700.0)
    filt = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for m in range(n_mels):
        lo, ctr, hi = mel_f[m], mel_f[m + 1], mel_f[m + 2]
        filt[m] = np.clip(
            np.minimum((mel_bins - lo) / (ctr - lo + 1e-9), (hi - mel_bins) / (hi - ctr + 1e-9)),
            0.0, 1.0,
        )
    win = np.hanning(n_fft).astype(np.float32)
    for t in range(n_frames):
        seg = xp[t * hop : t * hop + n_fft] * win
        spec = np.abs(np.fft.rfft(seg)) ** 2
        mel[t] = filt @ spec
    mel = np.log1p(mel)
    return mel


def _mel_dist_mel_l1(a: np.ndarray, b: np.ndarray) -> float:
    """L1 normalizzata per frame tra due mel-spettrogrammi (lunghezze allineate)."""
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return float("nan")
    return float(np.abs(a[:n] - b[:n]).mean())


def _dct2_ortho(x: np.ndarray) -> np.ndarray:
    """DCT di tipo II ortogonale (norm='ortho' di scipy.fft.dct) implementata in numpy puro.

    FIX (DDD, 2026-08-10): la versione precedente usava np.fft.dct, che NON esiste in
    NumPy (la DCT vive in scipy.fft): AttributeError a runtime, e la sintesi di esempio
    falliva silenziosamente (try/except) senza produrre samples/ né synthesis_metrics.
    Formula DCT-II ortogonale: y_k = sqrt(2/N) * sum_n x_n cos(pi/N (n + 1/2) k),
    con k=0 moltiplicata per 1/sqrt(2). Verificata contro scipy.fft.dct(norm='ortho').
    """
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[0]
    if N == 0:
        return x.copy()
    n = np.arange(N)
    k = np.arange(N)[:, None]
    _cos = np.cos((np.pi / N) * (n[None, :] + 0.5) * k)
    _y = _cos @ x
    _y[0] /= np.sqrt(2.0)  # k=0: la DCT-II ortogonale scala di 1/sqrt(2) (poi *sqrt(2/N))
    _y *= np.sqrt(2.0 / N)
    return _y


def _mel_dist_mcd(a: np.ndarray, b: np.ndarray) -> float:
    """Mel Cepstral Distortion (dB) approssimata: DCT del mel log + distanza euclidea."""
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return float("nan")
    # DCT-2 ortogonale (numpy puro) sui mel log -> cepstrum; confronto sui primi 13 coeff
    _c = np.zeros_like(a[:n])
    _d = np.zeros_like(b[:n])
    for i in range(a.shape[1]):
        _c[:, i] = _dct2_ortho(a[:n, i])
        _d[:, i] = _dct2_ortho(b[:n, i])
    return float(np.sqrt((( _c[:, :13] - _d[:, :13]) ** 2).mean()) * (10.0 / np.log(10.0)))


def _trim_trailing_silence(codes: np.ndarray, min_silent_frames: int = 4) -> np.ndarray:
    """Taglia il blocco di silenzio consecutivo in coda ai codici generati (per pulizia demo)."""
    if codes.shape[0] <= min_silent_frames:
        return codes
    sil = np.asarray(SILENCE_TOKEN_IDS, dtype=np.int64)[
        None, :
    ]  # (1, CB): codes è (T, CB)
    is_sil = (codes == sil).sum(axis=1) >= int(AGENT_SILENCE_MATCH_FRAC * N_CODEBOOKS)
    run = 0
    for t in range(codes.shape[0] - 1, -1, -1):
        if is_sil[t]:
            run += 1
        else:
            break
    if run >= min_silent_frames:
        return codes[: codes.shape[0] - run]
    return codes


def _build_demo_input(
    user_mel: np.ndarray, agent_mel: Optional[np.ndarray] = None, seq_len: int = 64
) -> Dict[str, torch.Tensor]:
    """Costruisce l'input LLM per la demo replicando ESATTAMENTE la logica di
    FullDuplexDataset.__getitem__ (percorso nativo + auto-ascolto, niente testo).

    user_mel / agent_mel: (T_mel, MEL_BINS) a 100fps; vengono allineati alla
    finestra (seq_len frame agente = seq_len*MEL_PER_AGENT_FRAME mel) e interleavati
    [u8, a8, u8, a8, ...]. Se agent_mel è None -> auto-ascolto a zero (primo turno).
    """
    mel_window = seq_len * MEL_PER_AGENT_FRAME
    u_win = np.zeros((mel_window, MEL_BINS), dtype=np.float32)
    k = min(user_mel.shape[0], mel_window)
    u_win[:k] = user_mel[:k]
    a_win = np.zeros((mel_window, MEL_BINS), dtype=np.float32)
    if agent_mel is not None:
        k = min(agent_mel.shape[0], mel_window)
        a_win[:k] = agent_mel[:k]
        d = SELF_LISTENING_DELAY_FRAMES
        if d > 0:
            a_win = np.concatenate(
                [
                    np.zeros((d * MEL_PER_AGENT_FRAME, MEL_BINS), dtype=np.float32),
                    a_win[: -d * MEL_PER_AGENT_FRAME]
                    if d * MEL_PER_AGENT_FRAME < mel_window
                    else a_win,
                ],
                axis=0,
            )
    u4 = u_win.reshape(seq_len, MEL_PER_AGENT_FRAME, MEL_BINS)
    a4 = a_win.reshape(seq_len, MEL_PER_AGENT_FRAME, MEL_BINS)
    mel_seq = np.stack([u4, a4], axis=1).reshape(mel_window * 2, MEL_BINS)

    input_len = 3 + TEXT_BLOCK_LEN + (mel_window * 2) // 4
    input_ids = np.full(input_len, AUDIO_TOKEN_ID, dtype=np.int64)
    input_ids[0] = BOS_TOKEN_ID
    input_ids[1 : 1 + TEXT_BLOCK_LEN] = E2B_PAD_TOKEN_ID
    input_ids[1 + TEXT_BLOCK_LEN] = BOA_TOKEN_ID
    input_ids[-1] = EOA_TOKEN_ID
    attention_mask = np.ones(input_len, dtype=np.int64)
    attention_mask[1 : 1 + TEXT_BLOCK_LEN] = 0
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0),
        "input_features": torch.tensor(mel_seq, dtype=torch.float32).unsqueeze(0),
        "input_features_mask": torch.ones(mel_window * 2, dtype=torch.bool).unsqueeze(
            0
        ),
    }


def _text_window_readable(text_by_window: Optional[list], idx: int) -> str:
    """Rende leggibile il testo atteso della finestra (decodifica tentata, fallback token)."""
    t_ids = None
    if text_by_window is not None and idx < len(text_by_window):
        t_ids = text_by_window[idx]
    if not t_ids:
        return "(nessun testo / finestra solo audio)"
    ids = [int(i) for i in t_ids]
    while ids and ids[0] == E2B_PAD_TOKEN_ID:
        ids.pop(0)
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "google/gemma-4-E2B-it", token=os.environ.get("HF_TOKEN")
        )
        txt = tok.decode(ids)
        if txt:
            return txt
    except Exception:
        pass
    return f"token={ids[:32]}"


def synthesize_examples(
    duplex_model,
    dataset,
    decoder: Optional[Code2WavDecoder] = None,
    n_windows: int = SYNTH_SAMPLE_WINDOWS,
    output_dir: str = SYNTH_OUTPUT_DIR,
) -> list:
    """Decodifica finestre del dataset con Code2Wav e salva wav ascoltabili.

    Per ogni finestra scrive DUE versioni:
      - generated_{i:03d}.wav : percorso del MODELLO (generate nativo -> 16 codebook)
      - oracle_{i:03d}.wav    : decodifica dei codici GT della finestra (tetto del decoder)
    Il confronto generated vs oracle isola la qualità: degradazione nel modello
    (h_t/codici predetti) vs nel decoder. Scrive anche README.txt con i testi
    attesi per finestra. Ritorna la lista dei percorsi wav generati.
    """
    os.makedirs(output_dir, exist_ok=True)
    if duplex_model is None or dataset is None or len(dataset) == 0:
        print("[Fase 2 - Sintesi] Nessun dataset/modello disponibile per la sintesi.")
        sys.stdout.flush()
        return []
    if decoder is None:
        decoder = Code2WavDecoder()
    device = next(duplex_model.base_model.parameters()).device
    duplex_model.eval()
    generated = []
    readme = []
    shown = 0
    synth_metrics = []  # (DDD) metrica di qualità sintesi generated vs oracle
    n_windows = max(1, int(n_windows))
    for i in range(min(n_windows * 3, len(dataset))):  # salta finestre di solo padding
        if shown >= n_windows:
            break
        item = dataset[i]
        n_real = int((item["audio_targets"][:, 0] != PAD_TARGET_ID).sum().item())
        if n_real <= 0:
            continue
        txt = _text_window_readable(
            dataset.text_by_window if hasattr(dataset, "text_by_window") else None, i
        )
        inp = {
            "input_ids": item["input_ids"].unsqueeze(0).to(device),
            "attention_mask": item["attention_mask"].unsqueeze(0).to(device),
            "input_features": item["input_features"].unsqueeze(0).to(device),
            "input_features_mask": item["input_features_mask"].unsqueeze(0).to(device),
        }
        with torch.inference_mode():
            out = duplex_model.generate(temperature=SYNTH_TEMPERATURE, **inp)
        gen_codes = out["audio_codes"][0][:n_real].cpu().numpy().astype(np.int64)
        gen_codes = _trim_trailing_silence(gen_codes)
        gt_codes = (
            item["audio_targets"][:n_real].numpy().astype(np.int64)
        )  # (n_real, CB), come gen_codes: decode() vuole (T, CB)
        wav_gen = decoder.decode(torch.from_numpy(gen_codes))
        wav_ora = decoder.decode(torch.from_numpy(gt_codes))
        # METRICA DI QUALITÀ SINTESI (DDD): mel-L1 normalizzata tra il wav del
        # modello (generated) e quello dei codici GT (oracle). Dato che sub-talker
        # + decoder sono gli STESSI, questa distanza isola quanto la predizione di
        # Gemma (cb0 + h_t) si avvicina al GT: è il proxy più vicino alla qualità
        # Qwen-TTS percepita. Più basso = generated più simile all'oracle.
        _mel_gen = _wav_to_mel_l1(wav_gen, SYNTH_SAMPLE_RATE)
        _mel_ora = _wav_to_mel_l1(wav_ora, SYNTH_SAMPLE_RATE)
        _mel_l1 = _mel_dist_mel_l1(_mel_gen, _mel_ora)
        _mcd = _mel_dist_mcd(_mel_gen, _mel_ora)
        synth_metrics.append(
            f"{i:03d}\t{_mel_l1:.4f}\t{_mcd:.3f}\t{n_real}\t{txt}"
        )
        p_gen = os.path.join(output_dir, f"generated_{i:03d}.wav")
        p_ora = os.path.join(output_dir, f"oracle_{i:03d}.wav")
        sf.write(p_gen, wav_gen, SYNTH_SAMPLE_RATE)
        sf.write(p_ora, wav_ora, SYNTH_SAMPLE_RATE)
        generated.append(p_gen)
        readme.append(
            f"finestra {i:03d} ({n_real} frame = {n_real * 80}ms)\n  testo atteso: {txt}\n  -> {p_gen}\n  -> oracle: {p_ora}"
        )
        shown += 1
    with open(os.path.join(output_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(readme))
    if synth_metrics:
        _m_path = os.path.join(output_dir, "synthesis_metrics.txt")
        with open(_m_path, "w", encoding="utf-8") as f:
            f.write(
                "# Metrica di qualità sintesi: generated (modello) vs oracle (GT)\n"
                "# sub-talker+decoder IDENTICI: la distanza isola la qualità della "
                "predizione Gemma (cb0+h_t)\n"
                "# mel_l1 = L1 normalizzata su mel (più basso = più vicino a Qwen)\n"
                "# mcd    = Mel Cepstral Distortion (dB, più basso = più simile)\n"
                "window\tmel_l1\tmcd\tn_frames\ttext\n"
            )
            f.write("\n".join(synth_metrics) + "\n")
            _l1s = [float(x.split("\t")[1]) for x in synth_metrics]
            _mcds = [float(x.split("\t")[2]) for x in synth_metrics]
            f.write(
                f"\nMEDIE\t{sum(_l1s)/len(_l1s):.4f}\t{sum(_mcds)/len(_mcds):.3f}\n"
            )
        print(
            f"[Fase 2 - Sintesi] 📊 Qualità sintesi (generated vs oracle): "
            f"mel_l1 medio={sum(_l1s)/len(_l1s):.4f}, MCD medio={sum(_mcds)/len(_mcds):.3f} dB"
        )
        sys.stdout.flush()
    print(
        f"[Fase 2 - Sintesi] {shown} esempi sintetizzati in '{output_dir}' "
        f"(generated_XXX.wav = modello, oracle_XXX.wav = codici GT)."
    )
    sys.stdout.flush()
    return generated


def demo_talk(
    duplex_model,
    decoder: Optional[Code2WavDecoder] = None,
    input_wav: Optional[str] = None,
    output_wav: str = os.path.join(SYNTH_OUTPUT_DIR, "agent_response.wav"),
):
    """🎤 DEMO "parla col modello": carica un wav utente, genera la risposta agente.

    In Kaggle il kernel NON ha microfono: registra il tuo parlato (telefono,
    registratore, ffmpeg) e caricalo in una di queste posizioni:
      1) input_wav=samples/user_input.wav (o percorso assoluto)
      2) samples/user_input.wav  (upload nel notebook via "Add data" o file upload)
      3) input_wav=<percorso qualsiasi> (es. campioni di prova in working)
    Il wav viene ri-campionato a 16kHz, trasformato in mel e processato dal
    modello in percorso NATIVO (come il training); la risposta agente (16 codebook)
    viene decodificata con Code2Wav e salvata in output_wav.
    """
    os.makedirs(SYNTH_OUTPUT_DIR, exist_ok=True)
    if duplex_model is None:
        print(
            "[Fase 2 - Demo] Nessun modello: carica prima il sistema (load_gemma4_duplex_system)."
        )
        sys.stdout.flush()
        return None
    if decoder is None:
        decoder = Code2WavDecoder()
    candidates = []
    if input_wav:
        candidates.append(input_wav)
    candidates.append(os.path.join(SYNTH_OUTPUT_DIR, "user_input.wav"))
    wav_path = next((c for c in candidates if os.path.isfile(c)), None)
    if wav_path is None:
        print(
            f"[Fase 2 - Demo] Nessun wav utente trovato in {candidates}. "
            f"Carica samples/user_input.wav o passa input_wav=<percorso>."
        )
        sys.stdout.flush()
        return None
    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    mono16 = _resample_np(mono, int(sr), MEL_SAMPLING_RATE)
    mel_ext = _NumpyGemma4MelExtractor()
    mel = extract_mel_features(mel_ext, mono16, MEL_SAMPLING_RATE)
    if mel is None or mel.shape[0] == 0:
        print(f"[Fase 2 - Demo] Estrazione mel fallita per {wav_path}.")
        sys.stdout.flush()
        return None
    inp = _build_demo_input(mel)
    device = next(duplex_model.base_model.parameters()).device
    inp = {k: v.to(device) for k, v in inp.items()}
    duplex_model.eval()
    with torch.inference_mode():
        out = duplex_model.generate(temperature=SYNTH_TEMPERATURE, **inp)
    codes = out["audio_codes"][0].cpu().numpy().astype(np.int64)
    codes = _trim_trailing_silence(codes)
    wav = decoder.decode(torch.from_numpy(codes))
    sf.write(output_wav, wav, SYNTH_SAMPLE_RATE)
    n_frames = codes.shape[0]
    print(
        f"[Fase 2 - Demo] ✅ Input: {wav_path} ({len(mono) / sr:.1f}s) -> "
        f"risposta agente {n_frames * 80}ms -> {output_wav}"
    )
    print("[Fase 2 - Demo] Scarica il wav da Kaggle e ascolta la risposta.")
    sys.stdout.flush()
    return output_wav


def demo_conversation(
    duplex_model,
    dataset,
    decoder: Optional[Code2WavDecoder] = None,
    n_turns: int = 3,
    output_dir: str = "samples/conv",
    self_listen_synthetic: bool = True,
) -> list:
    """💬 DEMO conversazione a catena: il modello risponde ai turni utente REALI
    della finestra di validazione ascoltando il PROPRIO parlato precedente.

    turno k: input = [mel utente GT del blocco k] + [mel agente del turno k-1
    (auto-ascolto: sintetizzato dal turno precedente se self_listen_synthetic,
    altrimenti mel agente GT)] -> il modello genera la risposta del turno k.
    Salva turn_0.wav .. turn_{k}.wav in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    if duplex_model is None or dataset is None or len(dataset) == 0:
        print("[Fase 2 - Demo] Nessun dataset/modello disponibile.")
        sys.stdout.flush()
        return []
    if decoder is None:
        decoder = Code2WavDecoder()
    device = next(duplex_model.base_model.parameters()).device
    duplex_model.eval()
    user_mel = np.asarray(dataset.user_mel, dtype=np.float32)  # (T_mel, BINS)
    agent_mel = np.asarray(dataset.agent_mel, dtype=np.float32)
    mel_window = 64 * MEL_PER_AGENT_FRAME
    agent_mel_synth = None
    paths = []
    for k in range(n_turns):
        m0 = k * mel_window
        u_blk = user_mel[m0 : m0 + mel_window]
        if u_blk.shape[0] == 0:
            break
        a_blk = (
            agent_mel_synth
            if (self_listen_synthetic and agent_mel_synth is not None)
            else agent_mel[m0 : m0 + mel_window]
        )
        inp = _build_demo_input(u_blk, a_blk)
        inp = {kk: vv.to(device) for kk, vv in inp.items()}
        with torch.inference_mode():
            out = duplex_model.generate(temperature=SYNTH_TEMPERATURE, **inp)
        codes = out["audio_codes"][0].cpu().numpy().astype(np.int64)
        codes = _trim_trailing_silence(codes)
        wav = decoder.decode(torch.from_numpy(codes))
        p = os.path.join(output_dir, f"turn_{k}.wav")
        sf.write(p, wav, SYNTH_SAMPLE_RATE)
        paths.append(p)
        # Auto-ascolto del turno successivo: mel del parlato SINTETIZZATO
        wav16 = _resample_np(
            np.asarray(wav, dtype=np.float32), SYNTH_SAMPLE_RATE, MEL_SAMPLING_RATE
        )
        mel_ext = _NumpyGemma4MelExtractor()
        m_s = extract_mel_features(mel_ext, wav16, MEL_SAMPLING_RATE)
        agent_mel_synth = m_s if (m_s is not None and m_s.shape[0] > 0) else None
        print(
            f"[Fase 2 - Demo] turno {k}: {codes.shape[0]} frame agente ({codes.shape[0] * 80}ms) -> {p}"
        )
        sys.stdout.flush()
    return paths


def run_training_pipeline(duplex_model=None, num_epochs: int = 1):
    """Carica i dati Parquet (Fase 2) ed esegue il training loop."""
    print("\n====================================================================")
    print("=== AVVIO FASE 2: TRAINING GEMMA 4 FULL-DUPLEX ===")
    print("====================================================================\n")
    sys.stdout.flush()
    parquet_cache = (
        f"data/train_tokens_{DUPLEXCONV_MAX_HOURS}h_{MDC_MAX_SAMPLES}s.parquet"
    )

    # La Fase 1 scrive SOLO file con suffisso (_mel/_ameld/_codes/_text/_hyperparams),
    # mai il file base: il check deve guardare i suffissi reali.
    cache_mel = parquet_cache.replace(".parquet", "_mel.parquet")
    cache_codes = parquet_cache.replace(".parquet", "_codes.parquet")
    if os.path.exists(cache_mel) and os.path.exists(cache_codes):
        # FIX ANTI-OOM: se i .dat memmap mancano, li genera una volta (peak ~3.3GB/file)
        convert_parquet_to_memmap(parquet_cache)
        user_mel, agent_mel, agent_codes, text_by_window = load_tokens_from_parquet(
            parquet_cache
        )
    elif os.environ.get("SMOKE") == "1" or os.environ.get("PHASE2_SMOKE_BATCHES"):
        print(
            "[Fase 2 - Training] SMOKE MODE ATTIVO: Nessun parquet in data/, generazione dati sintetici (4 finestre da 64 frame)..."
        )
        sys.stdout.flush()
        n_frames = 256  # 4 finestre da 64
        user_mel = np.random.randn(n_frames * MEL_PER_AGENT_FRAME, MEL_BINS).astype(np.float32)
        agent_mel = np.random.randn(n_frames * MEL_PER_AGENT_FRAME, MEL_BINS).astype(np.float32)
        agent_codes = np.random.randint(0, 2048, size=(N_CODEBOOKS, n_frames), dtype=np.int16)
        agent_codes[0] = np.random.randint(0, 2048, size=n_frames, dtype=np.int16)
        text_by_window = [[101, 102, 103]] * (n_frames // 64)
        global SEGMENT_ENDS, TOKENIZER_REAL_ENCODING_ACTIVE
        SEGMENT_ENDS = [n_frames]
        TOKENIZER_REAL_ENCODING_ACTIVE = True
    else:
        print(
            "[Fase 2 - Training] Cache Parquet non trovata: attesi "
            f"{os.path.basename(cache_mel)} e {os.path.basename(cache_codes)} in data/. "
            "Esegui prima la FASE 1 per generarla!"
        )
        sys.stdout.flush()
        return

    # Verifica struttura dati coerente con gli iperparametri del tokenizer
    if agent_codes.shape[0] != N_CODEBOOKS:
        raise RuntimeError(
            f"Codici con {agent_codes.shape[0]} codebook, attesi {N_CODEBOOKS}. "
            "Rigenera il parquet rieseguendo la FASE 1 (kernel riavviato)."
        )
    for name, mel in (("utente", user_mel), ("agente", agent_mel)):
        if mel.shape[0] != agent_codes.shape[1] * MEL_PER_AGENT_FRAME:
            raise RuntimeError(
                f"Disallineamento mel {name}/codici: mel={mel.shape[0]} frame, attesi "
                f"{agent_codes.shape[1] * MEL_PER_AGENT_FRAME}. Rigenera la FASE 1."
            )
    print(
        f"[Fase 2 - Training] SELF_LISTENING={SELF_LISTENING} (doppio canale mel interleavato "
        f"{MEL_CHANNELS}x{MEL_PER_AGENT_FRAME}x{MEL_BINS} per frame), "
        f"USE_FRAME_WEIGHTING={USE_FRAME_WEIGHTING} (open={FRAME_WEIGHT_OPEN}, "
        f"yield={FRAME_WEIGHT_YIELD}, co-talk={FRAME_WEIGHT_CO_TALK}), "
        f"TEXT_BLOCK_LEN={TEXT_BLOCK_LEN} (supervisione testuale, peso {TEXT_LOSS_WEIGHT})."
    )
    sys.stdout.flush()

    # ---- Validation split a confine di SEGMENTO (conversazione/turno) ----
    # SEGMENT_ENDS viene ripristinato dal sidecar; val_start = primo confine >= 90%
    # dei frame. Le finestre di validazione non tagliano a metà una conversazione.
    t_agent = agent_codes.shape[1]
    val_start = int((1.0 - VAL_FRACTION) * t_agent)
    for e in SEGMENT_ENDS:
        if e >= val_start:
            val_start = e
            break
    val_start = min(val_start, t_agent)
    n_train_w = val_start // 64
    n_val_w = max(0, (t_agent - val_start) // 64)
    print(
        f"[Fase 2 - Training] VALIDATION SPLIT: {n_train_w} finestre train (frame 0..{val_start}), "
        f"{n_val_w} finestre val (frame {val_start}..{t_agent}) "
        f"({100.0 * val_start / max(1, t_agent):.1f}% train, confine di segmento)."
    )
    sys.stdout.flush()

    # ---- Dataset con supervisione testuale (testo risposta agente per finestra) ----
    def _slice_text(text_list: list, f_start: int) -> list:
        if text_list is None:
            return None
        w_start = f_start // 64
        return text_list[w_start:]

    n_text = sum(1 for t in (text_by_window or []) if t is not None)
    if n_text:
        print(
            f"[Fase 2 - Training] Supervisione testuale ATTIVA: {n_text}/{len(text_by_window)} finestre "
            f"con testo risposta agente (LM head condivisa, tied embeddings)."
        )
    else:
        print("[Fase 2 - Training] Nessuna finestra con testo: training SOLO audio.")
    sys.stdout.flush()

    train_dataset = FullDuplexDataset(
        user_mel[: val_start * MEL_PER_AGENT_FRAME],
        agent_mel[: val_start * MEL_PER_AGENT_FRAME],
        agent_codes[:, :val_start],
        seq_len=64,
        text_by_window=_slice_text(text_by_window, 0),
    )
    if n_val_w > 0:
        val_dataset = FullDuplexDataset(
            user_mel[val_start * MEL_PER_AGENT_FRAME :],
            agent_mel[val_start * MEL_PER_AGENT_FRAME :],
            agent_codes[:, val_start:],
            seq_len=64,
            text_by_window=_slice_text(text_by_window, val_start),
        )
    else:
        val_dataset = None
    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Diagnostica distribuzione target: loss ~0.002 + spike = target degenerato (non bug di visualizzazione)
    a0 = agent_codes[0]
    vals, counts = np.unique(a0, return_counts=True)
    top_share = float(counts.max() / counts.sum())
    print(
        f"[Fase 2 - Training] Target cb0 agente: {len(vals)} token unici, "
        f"dominante {int(vals[counts.argmax()])} al {top_share:.1%}."
    )
    if not TOKENIZER_REAL_ENCODING_ACTIVE:
        print(
            "[Fase 2 - Training] ⚠️ Fase 1 NON ha usato l'encoder neurale reale (REAL_ENCODING assente/false): "
            "dati fallback FFT/sintetici -> la loss tenderà a collassare. NON procedere oltre: rigenera la Fase 1."
        )
    elif top_share > 0.95 or len(vals) < 8:
        print(
            "[Fase 2 - Training] ⚠️ TARGET DEGENERATO: la loss collasserà a ~0 (pattern osservato). "
            "Distribuzione troppo povera per un training significativo."
        )
    else:
        print(
            "[Fase 2 - Training] ✅ Distribuzione target sana: la loss avrà un valore significativo."
        )
    sys.stdout.flush()

    # FIX (DDD): rimossa la ricostruzione 'dataset = FullDuplexDataset(user_mel, agent_mel, agent_codes, ...)'
    # che sovrascriveva 'dataloader' (creato sopra con lo split train + text_by_window) col dataset
    # COMPLETO senza testo: causava (a) leakage dei frame val nel training e (b) text_loss=0 in training.
    # Ora 'dataloader' resta lo split train (0..val_start) con supervisione testuale.

    print(
        f"[Fase 2 - Training] ✅ COUPLING ATTIVO (ADR-005): codebook 1..15 dal sub-talker Qwen congelato "
        f"(loss peso {COUPLING_LOSS_WEIGHT}, warm-up {COUPLING_WARMUP_STEPS} step con h_t.detach())."
    )
    sys.stdout.flush()

    if duplex_model is not None:
        # FIX STABILITÀ (DDD): lr configurabile via env PHASE2_LR (default 2e-4).
        # Per l'epoca 2 si usa PHASE2_LR=5e-5: lr più basso = oscillazioni ridotte
        # dopo che il modello ha già raggiunto minimi buoni (best 5.6 vs media 18.9).
        _lr = float(os.environ.get("PHASE2_LR", "2e-4"))
        _max_gn = float(os.environ.get("PHASE2_MAX_GRAD_NORM", "0") or 0)
        print(f"[Fase 2 - Training] Optimizer: AdamW lr={_lr:g}, wd=0.01"
              f" | max_grad_norm={_max_gn if _max_gn > 0 else 'off'}")
        optimizer = torch.optim.AdamW(
            duplex_model.parameters(), lr=_lr, weight_decay=0.01
        )
        duplex_model.train()
        # RESUME (FIX ANTI-PERDITA, DDD): se PHASE2_RESUME_CKPT è impostato, ricarica
        # LoRA + teste custom dal checkpoint periodico e riparte da PHASE2_START_STEP
        # (step assoluti già completati). Il dataloader shuffle=True ha un ordine nuovo
        # rispetto al run interrotto: è training stocastico standard in resume, i pesi
        # del checkpoint vengono comunque usati. NOTA: lo stato dell'ottimizzatore AdamW
        # non è salvato nel checkpoint -> momentum perso (deviazione accettabile, lr 2e-4).
        _resume_ckpt = os.environ.get("PHASE2_RESUME_CKPT", "").strip()
        _start_step = int(os.environ.get("PHASE2_START_STEP", "0") or 0)
        if _resume_ckpt:
            load_gemma4_duplex_checkpoint(duplex_model, _resume_ckpt)
            print(
                f"[Fase 2 - Training] RESUME da '{_resume_ckpt}' (start_step={_start_step})."
            )
        else:
            print(
                f"[Fase 2 - Training] Avvio ciclo di ottimizzazione su {len(dataloader)} batch per epoca..."
            )
        sys.stdout.flush()

        # FIX ANTI-EPOCA-MAI-COMPLETA (DDD): il resume PRECEDENTE ripartiva sempre da
        # step=0 del dataloader con _abs_step = _start_step + step, quindi RI-ESEGUIVA
        # tutti i 12237 batch e l'epoca non finiva mai (la sessione muore ~ogni 2500
        # batch). Ora la fine epoca è definita su step ASSOLUTI: _abs_step raggiunge
        # _start_step + len(dataloader) => break + validation finale + sintesi.
        _tot_batches = len(dataloader)
        _end_step = _start_step + _tot_batches
        for epoch in range(num_epochs):
            pbar = tqdm(
                dataloader, desc=f"Epoca {epoch + 1}/{num_epochs}", unit="batch"
            )
            # Device di riferimento del modello base (PEFT/LoRA su cuda:0)
            base_device = next(duplex_model.base_model.parameters()).device

            for step, batch in enumerate(pbar):
                # SMOKE temporaneo: limita il numero di batch di training via env
                # (PHASE2_SMOKE_BATCHES=2 -> 2 batch, poi validation+checkpoint+sintesi).
                # In RESUME i batch già completati vengono saltati (step assoluto = start+step).
                _abs_step = _start_step + step
                if _abs_step < _start_step:
                    continue
                if _abs_step >= _end_step:
                    print(
                        f"[Fase 2 - Training] EPOCA COMPLETA @ step assoluto {_abs_step} "
                        f"(target {_end_step}): break + validation finale."
                    )
                    sys.stdout.flush()
                    break
                if (
                    os.environ.get("PHASE2_SMOKE_BATCHES")
                    and step >= int(os.environ["PHASE2_SMOKE_BATCHES"])
                ):
                    print(
                        f"[Fase 2 - Training] SMOKE: fermato dopo {step} batch "
                        "(PHASE2_SMOKE_BATCHES)."
                    )
                    sys.stdout.flush()
                    break
                # Invio del batch al device del modello (FIX: prima restavano su CPU
                # -> torch.embedding riceveva weight cuda:0 + indici CPU e crashava
                # con "Unhandled FakeTensor Device Propagation" durante il tracing dynamo)
                input_ids = batch["input_ids"].to(base_device)
                attention_mask = batch["attention_mask"].to(base_device)
                input_features = batch["input_features"].to(base_device)
                input_features_mask = batch["input_features_mask"].to(base_device)
                audio_targets = batch["audio_targets"].to(base_device)
                frame_weights = batch["frame_weights"].to(base_device)
                text_targets = batch["text_targets"].to(base_device)

                optimizer.zero_grad()
                outputs = duplex_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    input_features=input_features,
                    input_features_mask=input_features_mask,
                    audio_targets=audio_targets,
                    frame_weights=frame_weights,
                    text_targets=text_targets,
                )
                loss = outputs["loss"]
                loss.backward()
                # GRADIENT CLIPPING (FIX STABILITÀ, DDD): evita che un gradiente
                # anomalo butti fuori il modello dal minimo buono raggiunto (causa
                # delle oscillazioni 5.6 <-> 40 osservate). Attivo se PHASE2_MAX_GRAD_NORM>0.
                if _max_gn > 0:
                    torch.nn.utils.clip_grad_norm_(duplex_model.parameters(), _max_gn)
                optimizer.step()
                duplex_model.coupling_step += 1
                post = {"Loss": f"{loss.item():.4f}"}
                if (
                    outputs.get("coupling_loss") is not None
                    and outputs["coupling_loss"] > 0
                ):
                    post["Coup"] = f"{outputs['coupling_loss'].item():.4f}"
                if outputs.get("text_loss") is not None and outputs["text_loss"] > 0:
                    post["Txt"] = f"{outputs['text_loss'].item():.4f}"
                pbar.set_postfix(post)

                # OSSERVABILITA' run: log per-step (loss/coupling/text) su file per monitoraggio
                # senza dipendere dalla barra tqdm (utile per run lunghi e per il report finale).
                _run_log = os.environ.get("PHASE2_RUN_LOG", "data/phase2_run_log.tsv")
                _cl = outputs.get("coupling_loss")
                _tl = outputs.get("text_loss")
                with open(_run_log, "a", encoding="utf-8") as _lf:
                    _lf.write(
                        f"{epoch}\t{_abs_step}\t{loss.item():.4f}\t"
                        f"{(_cl.item() if _cl is not None else 0.0):.4f}\t"
                        f"{(_tl.item() if _tl is not None else 0.0):.4f}\n"
                    )

                # CHECKPOINT PERIODICO (FIX ANTI-PERDITA, DDD): con run da ~4.6h/epoca
                # la morte del kernel a metà costo 2+ ore di GPU. Salva checkpoint ogni
                # PHASE2_CKPT_EVERY batch (default 1000) + log del progresso, così si può
                # riprendere o almeno non perdere tutto. Il checkpoint è veloce (~30s).
                _ckpt_every = int(os.environ.get("PHASE2_CKPT_EVERY", "1000"))
                if _ckpt_every > 0 and _abs_step > 0 and (_abs_step + 1) % _ckpt_every == 0:
                    print(
                        f"[Fase 2 - Training] Checkpoint periodico @ step {step + 1}..."
                    )
                    sys.stdout.flush()
                    try:
                        save_gemma4_duplex_checkpoint(
                            duplex_model,
                            "checkpoints/gemma4_duplex_sota_ckpt",
                        )
                        with open(
                            os.environ.get("PHASE2_CKPT_LOG", "data/phase2_ckpt_log.txt"),
                            "a",
                            encoding="utf-8",
                        ) as _cf:
                            _cf.write(f"epoch={epoch}\tstep={_abs_step + 1}\n")
                    except Exception as _ce:
                        print(f"[Fase 2 - Training] ⚠️ checkpoint periodico fallito ({_ce})")
                        sys.stdout.flush()

            # ---- Loss di VALIDAZIONE (inference_mode, un solo forward) ----
            # NOTA: il validation DEVE restare in inference_mode: con grad attivo
            # unsloth 5.5.0/5.5.1 (gradient checkpointing, KV carrier condiviso)
            # vieta 2 forward senza backward -> RuntimeError. Il ricast a fp32 usa
            # un criterio robusto (classe *norm* O attributi normalized_shape/eps):
            # dopo il compile i LayerNorm possono essere wrappati e il nome classe
            # non basta. Se un batch di val fallisce ancora per dtype, lo si salta:
            # la metrica val è un nice-to-have, non deve bloccare checkpoint+sintesi.
            if val_dataset is not None and len(val_dataset) > 0:
                duplex_model.eval()
                n_rc = 0
                for _vn, _vm in duplex_model.base_model.named_modules():
                    if (
                        (
                            "norm" in type(_vm).__name__.lower()
                            or hasattr(_vm, "normalized_shape")
                        )
                        and getattr(_vm, "weight", None) is not None
                        and _vm.weight.dtype != torch.float32
                    ):
                        _vm.to(torch.float32)
                        n_rc += 1
                if n_rc:
                    print(f"[Fase 2 - Validation] Ricast {n_rc} LayerNorm a fp32.")
                    sys.stdout.flush()
                vloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
                val_losses, val_txt = [], []
                with torch.inference_mode():
                    for vb in vloader:
                        try:
                            vout = duplex_model(
                                input_ids=vb["input_ids"].to(base_device),
                                attention_mask=vb["attention_mask"].to(base_device),
                                input_features=vb["input_features"].to(base_device),
                                input_features_mask=vb["input_features_mask"].to(
                                    base_device
                                ),
                                audio_targets=vb["audio_targets"].to(base_device),
                                frame_weights=vb["frame_weights"].to(base_device),
                                text_targets=vb["text_targets"].to(base_device),
                            )
                        except RuntimeError as _ve:
                            if any(
                                _t in str(_ve)
                                for _t in ("scalar type", "Half", "Float", "dtype")
                            ):
                                print(
                                    f"[Fase 2 - Validation] ⚠️ batch val saltato ({_ve}); "
                                    "si prosegue con checkpoint e sintesi."
                                )
                                sys.stdout.flush()
                                break
                            raise
                        if vout["loss"] is not None:
                            val_losses.append(vout["loss"].item())
                        if vout.get("text_loss") is not None and vout["text_loss"] > 0:
                            val_txt.append(vout["text_loss"].item())
                if val_losses:
                    import statistics as _st

                    msg = f"[Fase 2 - Validation] Epoca {epoch + 1}: val_loss={_st.mean(val_losses):.4f}"
                    if val_txt:
                        msg += f" | val_text={_st.mean(val_txt):.4f}"
                    print(msg)
                    sys.stdout.flush()
                duplex_model.train()

    save_gemma4_duplex_checkpoint(duplex_model, "checkpoints/gemma4_duplex_sota")

    # ---- Sintesi di esempio + demo (ascolto della qualità appresa) ----
    try:
        synth_ds = (
            val_dataset
            if (val_dataset is not None and len(val_dataset) > 0)
            else train_dataset
        )
        synthesize_examples(duplex_model, synth_ds)
    except Exception as e:
        print(f"[Fase 2 - Sintesi] Avviso sintesi di esempio ({e}).")
        sys.stdout.flush()
    print("\n🎤 PER PARLARE COL MODELLO (demo):")
    print("  - carica il tuo parlato come wav (il kernel Kaggle non ha microfono):")
    print("      * samples/user_input.wav  oppure  percorso wav assoluto")
    print("  - poi esegui:  demo_talk(duplex_model)")
    print(
        "  - conversazione a catena multi-turno: demo_conversation(duplex_model, synth_ds)"
    )
    print("  - per ascoltare senza riallenare (kernel riavviato): ")
    print(
        "      load_gemma4_duplex_system() -> load_gemma4_duplex_checkpoint(m) -> demo_talk(m)"
    )
    print(
        "\n🔗 COUPLING (ADR-005): l'audio è renderizzato dal SUB-TALKER Qwen congelato (cb1..cb15) "
        "+ decoder Code2Wav; la qualità vocale attesa è quella nativa Qwen."
    )
    print(
        "  - wav ascoltabili generati a fine training in samples/ (generated_* = modello, "
        "oracle_* = codici GT decodificati)."
    )
    sys.stdout.flush()

    print("\n====================================================================")
    print("=== FASE 2 COMPLETATA CON SUCCESSO ===")
    print("====================================================================\n")
    sys.stdout.flush()


if __name__ == "__main__":
    print("=== PIPELINE DUAL-PHASE READY ===")
    sys.stdout.flush()
    # Nota: Su Kaggle eseguire prima execute_phase_1() nella Fase 1,
    # poi riavviare il kernel e lanciare run_training_pipeline() nella Fase 2.
    parquet_cache = (
        f"data/train_tokens_{DUPLEXCONV_MAX_HOURS}h_{MDC_MAX_SAMPLES}s.parquet"
    )
    cache_mel = parquet_cache.replace(".parquet", "_mel.parquet")
    if os.path.exists(cache_mel):
        duplex_model, _ = load_gemma4_duplex_system()
        run_training_pipeline(duplex_model, num_epochs=1)
    else:
        execute_phase_1()
