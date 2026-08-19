# Sorgenti ufficiali Qwen3-TTS (riferimento SDD)

Fetch da https://github.com/QwenLM/Qwen3-TTS (main) il 2026-08-19.

- `qwen3_tts_model.py` — `qwen_tts/inference/qwen3_tts_model.py`
- `modeling_qwen3_tts.py` — `qwen_tts/core/models/modeling_qwen3_tts.py`

Passaggi chiave per la pipeline adapter (citazioni da riga):

- `modeling_qwen3_tts.py:1652-1677` — nel `forward` del talker (fase generate), i codebook
  residui sono generati con `self.code_predictor.generate(..., do_sample=subtalker_dosample,
  top_p=subtalker_top_p, top_k=subtalker_top_k, temperature=subtalker_temperature)`.
  Default ufficiali (`qwen3_tts_model.py:320-328`): `subtalker_dosample=True`,
  `subtalker_top_k=50`, `subtalker_top_p=1.0`, `subtalker_temperature=0.9`.
- `modeling_qwen3_tts.py:2138-2148` — speaker embedding iniettato nel prefill codec SOLO se
  `speaker_embed is not None` (voice_clone/preset); il percorso "create speaker"
  (speaker=None) opera senza slot speaker.
- `modeling_qwen3_tts.py:1600-1610` — `forward_sub_talker_finetune` del teacher:
  input del code_predictor = [talker_hidden_states (h_t), cb0, cb1..cb14] (concat).
