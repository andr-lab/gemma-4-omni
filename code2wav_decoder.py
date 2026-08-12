# -*- coding: utf-8 -*-
"""
code2wav_decoder.py — DECODER Code2Wav Qwen3-TTS-Tokenizer-12Hz in torch puro
(ZERO transformers).

Perché esiste: il decoder del tokenizer 12Hz (model_type
`qwen3_tts_tokenizer_12hz`) NON è registrato nelle transformers 5.x installate
in Fase 2 su Kaggle (verificato: assente in 5.14.1 e nel tag v4.57.3 del repo
transformers; la definizione vive SOLO nel pacchetto qwen-tts 0.1.1, presente
in Fase 1 e assente dopo il riavvio del kernel). Senza un decoder reale in Fase
2 l'ascolto della qualità appresa cadrebbe nel fallback mock (beep!). La
replica 1:1 (stesso pattern collaudato di qwen_subtalker.py) garantisce la
decodifica in Fase 2 a costo zero di dipendenze.

Sorgente di riferimento: qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py
del pacchetto qwen-tts 0.1.1 (classi e forward replicati riga per riga):

  * SplitResidualVectorQuantizer: rvq_first (1 quantizer semantico) + rvq_rest
    (15 acoustic), codebook RVQ a `codebook_dim//2 = 256` dim (embedding_sum /
    cluster_usage), input_proj/output_proj Conv1d(1, bias=False) forzati,
    project_out = Identity (codebook_dim == dim)  -> righe 676-824
  * CausalConvNet: pad = kernel_eff - stride a sinistra, extra pad a destra
    per lunghezze non allineate  -> righe 205-246
  * Transformer: 8 layer, hidden 512, 16 head / 16 kv (GQA groups=1),
    head_dim 64, scaling head_dim^-0.5, pre-norm RMSNorm(1e-5), layer_scale
    0.01 su attn e mlp, MLP gate/up/down SiLU, RoPE default theta 10000,
    attention SLIDING (window 72): per T<=72 (tutte le finestre della demo e
    della sintesi) coincide con la mask causale pura  -> righe 242-520
  * Upsample: 2x [CausalTransConvNet(1024, stride 2) + ConvNeXtBlock]
  * Decoder: CausalConvNet(1024->1536, k7) + 4x DecoderBlock (rate 8,5,4,3,
    ognuno SnakeBeta + TransConv(2*rate, stride rate) + 3 ResidualUnit dil 1,3,9
    con SnakeBeta + conv k7 + SnakeBeta + conv k1) + SnakeBeta + Conv(96->1, k7)
  * forward: clamp finale in [-1, 1]  -> righe 823-887

Uso:
  * load_code2wav_decoder(dir) — carica code2wav_decoder.safetensors +
    code2wav_decoder_config.json estratti in Fase 1 (prefissi `decoder.*` del
    safetensors del tokenizer, senza il prefisso).
  * Il forward si aspetta codici (B, CB, T): ogni blocco consumerà la matrice
    come gli upstream (12.5 Hz -> 24 kHz: upsample totale 1920x).
"""
import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config di default (verificata su Qwen/Qwen3-TTS-Tokenizer-12Hz config.json)
# ---------------------------------------------------------------------------
DEFAULT_DECODER_CONFIG = {
    "num_quantizers": 16,
    "codebook_dim": 512,
    "codebook_size": 2048,
    "latent_dim": 1024,
    "hidden_size": 512,
    "num_hidden_layers": 8,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "head_dim": 64,
    "intermediate_size": 1024,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-5,
    "layer_scale_initial_scale": 0.01,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "sliding_window": 72,
    "max_position_embeddings": 2048,
    "rope_theta": 10000.0,
    "decoder_dim": 1536,
    "upsample_rates": [8, 5, 4, 3],
    "upsampling_ratios": [2, 2],
}


class _SnakeBeta(nn.Module):
    """Modified Snake activation (periodic magnitude via beta, freq via alpha)."""

    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
        self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        self.no_div_by_zero = 1e-9

    def forward(self, hidden_state):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # (1, C, 1) ~ (B, C, T)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        hidden_state = hidden_state + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(hidden_state * alpha), 2
        )
        return hidden_state


class _CausalConvNet(nn.Module):
    """Conv1d causale: left-pad (kernel_eff - stride), extra pad a destra."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                              dilation=dilation, groups=groups)
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride

    def _get_extra_padding(self, length):
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.kernel_size - self.padding)
        return ideal_length - length

    def forward(self, hidden_state):
        extra = self._get_extra_padding(hidden_state.shape[-1])
        hidden_state = F.pad(hidden_state, (self.padding, extra), mode="constant", value=0)
        return self.conv(hidden_state).contiguous()


class _CausalTransConvNet(nn.Module):
    """ConvTranspose1d causale: taglia i `kernel - stride` campioni finali."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)
        self.right_pad = int(kernel_size - stride)

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        if self.right_pad > 0:
            hidden_state = hidden_state[..., : hidden_state.shape[-1] - self.right_pad]
        return hidden_state.contiguous()


class _ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = _CausalConvNet(dim, dim, kernel_size=7, groups=dim, dilation=1)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, hidden_states):
        # FIX (2026-08-08, verificato vs modeling_qwen3_tts_tokenizer_v2.py 0.1.1
        # del pacchetto qwen-tts + round-trip col decoder ufficiale): il blocco ha
        # una RESIDUAL CONNECTION finale (hidden_states = input + hidden_states)
        # che la replica originale ometteva -> decoder rotto (rumore in output).
        # Sorgente ufficiale:
        #   https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py
        #   (Qwen3TTSTokenizerV2ConvNeXtBlock.forward: `hidden_states = input + hidden_states`)
        inp = hidden_states
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        hidden_states = self.gamma * hidden_states
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = inp + hidden_states
        return hidden_states


class _DecoderResidualUnit(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.act1 = _SnakeBeta(dim)
        self.conv1 = _CausalConvNet(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = _SnakeBeta(dim)
        self.conv2 = _CausalConvNet(dim, dim, kernel_size=1)

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.act2(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state + residual


class _DecoderBlock(nn.Module):
    def __init__(self, decoder_dim, layer_idx, upsample_rates):
        super().__init__()
        in_dim = decoder_dim // 2 ** layer_idx
        out_dim = decoder_dim // 2 ** (layer_idx + 1)
        upsample_rate = upsample_rates[layer_idx]
        block = [
            _SnakeBeta(in_dim),
            _CausalTransConvNet(in_dim, out_dim, 2 * upsample_rate, upsample_rate),
        ]
        for dilation in (1, 3, 9):
            block.append(_DecoderResidualUnit(out_dim, dilation))
        self.block = nn.ModuleList(block)

    def forward(self, hidden):
        for b in self.block:
            hidden = b(hidden)
        return hidden


# ---------------------------------------------------------------------------
# Quantizer RVQ (split: 1 semantico + 15 acustici, codebook 256-dim)
# ---------------------------------------------------------------------------
class _EuclideanCodebook(nn.Module):
    def __init__(self, dim, codebook_size, epsilon=1e-5):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.epsilon = epsilon
        self.cluster_usage = nn.Parameter(torch.ones(codebook_size))
        self.embedding_sum = nn.Parameter(torch.zeros(codebook_size, dim))

    def decode(self, codes):
        embedding = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
        return F.embedding(codes, embedding)


class _VectorQuantization(nn.Module):
    """codebook_dim == dim nel decoder -> project_out Identity (nessun peso)."""

    def __init__(self, dim, codebook_size):
        super().__init__()
        self._codebook = _EuclideanCodebook(dim, codebook_size)

    def decode(self, codes):
        quantized = self._codebook.decode(codes)
        return quantized.transpose(1, 2)


class _ResidualVectorQuantization(nn.Module):
    def __init__(self, num_quantizers, dim, codebook_size):
        super().__init__()
        self.layers = nn.ModuleList(
            [_VectorQuantization(dim, codebook_size) for _ in range(num_quantizers)]
        )

    def decode(self, codes):
        quantized = torch.zeros(1, device=codes.device)[0]
        for idx, layer_codes in enumerate(codes):
            quantized = quantized + self.layers[idx].decode(layer_codes)
        return quantized


class _ResidualVectorQuantizer(nn.Module):
    def __init__(self, input_dimension, dimension, output_dimension, n_q, bins):
        super().__init__()
        # force_projection=True nel teacher: proiezioni sempre presenti
        self.input_proj = nn.Conv1d(input_dimension, dimension, 1, bias=False)
        self.output_proj = nn.Conv1d(dimension, output_dimension, 1, bias=False)
        self.vq = _ResidualVectorQuantization(n_q, dimension, bins)

    def decode(self, codes):
        codes = codes.transpose(0, 1)
        quantized = self.vq.decode(codes)
        return self.output_proj(quantized)


class _SplitResidualVectorQuantizer(nn.Module):
    def __init__(self, dimension, n_q, n_q_semantic, bins, input_dimension, output_dimension):
        super().__init__()
        assert n_q > n_q_semantic
        self.n_q_semantic = n_q_semantic
        self.rvq_first = _ResidualVectorQuantizer(input_dimension, dimension, output_dimension, n_q_semantic, bins)
        self.rvq_rest = _ResidualVectorQuantizer(input_dimension, dimension, output_dimension, n_q - n_q_semantic, bins)

    def decode(self, codes):
        quantized = self.rvq_first.decode(codes[:, : self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            quantized = quantized + self.rvq_rest.decode(codes[:, self.n_q_semantic:])
        return quantized


# ---------------------------------------------------------------------------
# Transformer pre-decoder (sliding attention)
# ---------------------------------------------------------------------------
class _RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class _RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_position_embeddings=2048, theta=10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states, n_rep):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class _DecoderAttention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = cfg["head_dim"]
        self.num_key_value_groups = cfg["num_attention_heads"] // cfg["num_key_value_heads"]
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = cfg["attention_dropout"]
        self.sliding_window = cfg["sliding_window"]
        bias = cfg["attention_bias"]
        self.q_proj = nn.Linear(cfg["hidden_size"], cfg["num_attention_heads"] * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(cfg["hidden_size"], cfg["num_key_value_heads"] * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(cfg["hidden_size"], cfg["num_key_value_heads"] * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(cfg["num_attention_heads"] * self.head_dim, cfg["hidden_size"], bias=bias)

    def forward(self, hidden_states, position_embeddings, attention_mask):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class _DecoderMlp(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(cfg["intermediate_size"], cfg["hidden_size"], bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _LayerScale(nn.Module):
    def __init__(self, hidden_size, initial_scale=0.01):
        super().__init__()
        self.scale = nn.Parameter(torch.full((hidden_size,), initial_scale))

    def forward(self, x):
        return self.scale * x


class _TransformerLayer(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.self_attn = _DecoderAttention(cfg, layer_idx)
        self.mlp = _DecoderMlp(cfg)
        self.input_layernorm = _RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.post_attention_layernorm = _RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.self_attn_layer_scale = _LayerScale(cfg["hidden_size"], cfg["layer_scale_initial_scale"])
        self.mlp_layer_scale = _LayerScale(cfg["hidden_size"], cfg["layer_scale_initial_scale"])
        self.attention_type = "sliding_attention"

    def forward(self, hidden_states, position_embeddings, attention_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + self.self_attn_layer_scale(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_layer_scale(hidden_states)
        return hidden_states


def _build_sliding_window_causal_mask(T, sliding_window, device, dtype):
    """Mask (1,1,T,T): -inf sopra la diagonale (causale) + fuori finestra.

    sliding_window=72: per T<=72 coincide con la causale pura (tutte le finestre
    di decodifica della pipeline hanno T<=64). Per T>72 limita a una finestra
    locale di 72 come il teacher (create_sliding_window_causal_mask).
    """
    mask = torch.zeros(1, 1, T, T, device=device, dtype=dtype)
    mask = mask.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1), float("-inf"))
    if sliding_window and sliding_window > 0 and sliding_window < T:
        # j - i >= sliding_window -> fuori finestra
        far = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=sliding_window)
        mask = mask.masked_fill(far, float("-inf"))
    return mask


class _TransformerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([_TransformerLayer(cfg, i) for i in range(cfg["num_hidden_layers"])])
        self.norm = _RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.rotary_emb = _RotaryEmbedding(cfg["head_dim"], cfg["max_position_embeddings"], cfg["rope_theta"])
        self.window_size = cfg["sliding_window"]
        self.input_proj = nn.Linear(cfg["latent_dim"], cfg["hidden_size"])
        self.output_proj = nn.Linear(cfg["hidden_size"], cfg["latent_dim"])

    def forward(self, inputs_embeds):
        inputs_embeds = self.input_proj(inputs_embeds)
        seq_len = inputs_embeds.shape[1]
        position_ids = torch.arange(0, seq_len, device=inputs_embeds.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        attention_mask = _build_sliding_window_causal_mask(
            seq_len, self.window_size, inputs_embeds.device, inputs_embeds.dtype
        )
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, attention_mask)
        hidden_states = self.norm(hidden_states)
        return self.output_proj(hidden_states)


# ---------------------------------------------------------------------------
# Decoder completo
# ---------------------------------------------------------------------------
class Qwen3TTSDecoder12Hz(nn.Module):
    """Replica pura del Qwen3TTSTokenizerV2Decoder (Fase 2, zero transformers)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        c = self.cfg
        self.total_upsample = 1
        for r in list(c["upsample_rates"]) + list(c["upsampling_ratios"]):
            self.total_upsample *= int(r)
        self.pre_transformer = _TransformerModel(c)
        self.quantizer = _SplitResidualVectorQuantizer(
            dimension=c["codebook_dim"] // 2,
            n_q=c["num_quantizers"],
            n_q_semantic=1,
            bins=c["codebook_size"],
            input_dimension=c["codebook_dim"],
            output_dimension=c["codebook_dim"],
        )
        self.pre_conv = _CausalConvNet(c["codebook_dim"], c["latent_dim"], kernel_size=3)

        upsample = []
        for factor in c["upsampling_ratios"]:
            upsample.append(
                nn.ModuleList(
                    [
                        _CausalTransConvNet(c["latent_dim"], c["latent_dim"], factor, factor),
                        _ConvNeXtBlock(c["latent_dim"]),
                    ]
                )
            )
        self.upsample = nn.ModuleList(upsample)

        decoder = [_CausalConvNet(c["latent_dim"], c["decoder_dim"], 7)]
        for i in range(len(c["upsample_rates"])):
            decoder.append(_DecoderBlock(c["decoder_dim"], i, c["upsample_rates"]))
        output_dim = c["decoder_dim"] // 2 ** len(c["upsample_rates"])
        decoder += [_SnakeBeta(output_dim), _CausalConvNet(output_dim, 1, 7)]
        self.decoder = nn.ModuleList(decoder)

    def forward(self, codes):
        """codes: (B, num_quantizers, T) -> (B, 1, T * total_upsample) in [-1, 1]."""
        if codes.shape[1] != self.cfg["num_quantizers"]:
            raise ValueError(
                f"Attesi {self.cfg['num_quantizers']} codebook, ottenuti codes.shape[1]={codes.shape[1]}."
            )
        hidden = self.quantizer.decode(codes)
        hidden = self.pre_conv(hidden).transpose(1, 2)
        hidden = self.pre_transformer(hidden)
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)


def load_code2wav_decoder(weights_dir, device=None, dtype=torch.float32):
    """Carica il decoder standalone da `weights_dir` (safetensors + config json).

    I pesi sono quelli estratti in Fase 1 (prefissi `decoder.*` del safetensors
    del tokenizer, con il prefisso rimosso). Ritorna (module, cfg).
    """
    import safetensors.torch as st

    cfg_path = os.path.join(weights_dir, "code2wav_decoder_config.json")
    st_path = os.path.join(weights_dir, "code2wav_decoder.safetensors")
    if not os.path.exists(cfg_path) or not os.path.exists(st_path):
        raise FileNotFoundError(
            f"Decoder Code2Wav non trovato in '{weights_dir}' (attesi "
            "code2wav_decoder_config.json e code2wav_decoder.safetensors). "
            "Esegui prima la FASE 1 (extract_code2wav_decoder_weights)."
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = {**DEFAULT_DECODER_CONFIG, **cfg}
    model = Qwen3TTSDecoder12Hz(cfg)
    state = st.load_file(st_path)
    model.load_state_dict(state, strict=True)
    model.eval()
    if device is not None:
        model.to(device)
    if dtype is not None:
        model.to(dtype=dtype)
    return model, cfg


# ---------------------------------------------------------------------------
# Self-test (TDD): shape del forward, total_upsample, clamp, parità matematica
# ---------------------------------------------------------------------------
def _selftest() -> int:
    torch.manual_seed(0)
    cfg = dict(DEFAULT_DECODER_CONFIG)
    model = Qwen3TTSDecoder12Hz(cfg).eval()
    failures = []

    def check(name, cond, extra=""):
        status = "OK" if cond else "FAIL"
        print(f"  [{status}] {name}{(' | ' + extra) if extra else ''}")
        if not cond:
            failures.append(name)

    # 1) total_upsample 1920 (12.5 Hz -> 24 kHz)
    check("total_upsample = 1920", model.total_upsample == 1920, str(model.total_upsample))

    # 2) Forward: (B=1, CB=16, T=32) -> (1, 1, 61440) clamp in [-1, 1]
    with torch.no_grad():
        codes = torch.randint(0, 2048, (1, 16, 32))
        wav = model(codes)
    check("shape out (1,1,61440)", tuple(wav.shape) == (1, 1, 61440), str(tuple(wav.shape)))
    check("clamp [-1,1]", float(wav.min()) >= -1.0 and float(wav.max()) <= 1.0,
          f"min={float(wav.min()):.4f} max={float(wav.max()):.4f}")

    # 3) Forward con T non multiplo di upsample (pad destro causale)
    with torch.no_grad():
        wav17 = model(torch.randint(0, 2048, (1, 16, 17)))
    check("forward T=17 (pad)", tuple(wav17.shape) == (1, 1, 32640), str(tuple(wav17.shape)))

    # 4) Causalità transitoria: i primi campioni NON dipendono dagli ultimi codici
    with torch.no_grad():
        a = torch.randint(0, 2048, (1, 16, 16))
        b = a.clone()
        b[:, :, -4:] = 0  # cambia solo la coda dei codici
        wa = model(a)[0, 0, : 8 * 1920].clone()
        wb = model(b)[0, 0, : 8 * 1920].clone()
    check("causale: inizio invariato cambiando la coda",
          torch.allclose(wa, wb, atol=1e-5), f"maxdiff={float((wa - wb).abs().max()):.2e}")

    # 5) Gradienti: il decoder è usato solo a inferenza (no grad), ma i pesi
    #    devono caricare con strict=True (controllo del conteggio dei tensori)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [INFO] parametri decoder: {n_params / 1e6:.1f}M")
    check("parametri > 100M", n_params > 100e6, f"{n_params / 1e6:.1f}M")

    if failures:
        print(f"\n[FAIL] selftest decoder: {len(failures)} controlli falliti: {failures}")
        return 1
    print("\n[OK] selftest decoder Code2Wav: tutti i controlli passati.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
