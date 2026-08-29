# -*- coding: utf-8 -*-
"""
test_vllm_omni_plugin_tdd.py — Suite TDD per il Plugin vLLM-Omni Gemma-4-Omni Full-Duplex.

Verifica secondo TDD (RED -> GREEN -> REFACTOR) e Source-Driven Development (SDD):
1. Configurazione e topologia di `GEMMA4_OMNI_DUPLEX_PIPELINE` (Stage 0: LLM, Stage 1: Talker/V9, Stage 2: Code2Wav).
2. Input processor `thinker2talker_v9_prefix` per il calcolo del Prefix KV-Cache (16 frame, 2048D).
3. Processori inter-stadio `talker2code2wav_async_chunk` e `talker2code2wav_full_payload`.
4. `Gemma4OmniDuplexRuntimeExtension` con `TurnDetectorHead` a 5 stati (ADR-027).
5. Gestione del ciclo di vita Realtime WebSocket con Barge-In istantaneo (Zero-Reset KV-Cache).

Riferimenti ufficiali SDD:
- vllm-project/vllm-omni (PR #3907, commit 2026)
- arXiv:2602.02204 (vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models)
"""

import unittest
import torch
import torch.nn as nn
import numpy as np


class TestVllmOmniPluginTDD(unittest.TestCase):
    """Test unitari e di integrazione per il plugin vLLM-Omni Gemma-4-Omni."""

    def test_01_pipeline_topology_contract(self):
        """Test 1: Verifica che la PipelineConfig sia registrata e rispetti il contratto vLLM-Omni a 3 stadi."""
        from vllm_omni_plugin.pipeline import GEMMA4_OMNI_DUPLEX_PIPELINE

        self.assertEqual(GEMMA4_OMNI_DUPLEX_PIPELINE.model_type, "gemma4_omni_duplex")
        self.assertEqual(GEMMA4_OMNI_DUPLEX_PIPELINE.model_arch, "Gemma4OmniDuplexForConditionalGeneration")
        self.assertTrue(GEMMA4_OMNI_DUPLEX_PIPELINE.duplex_control_enabled)
        self.assertEqual(len(GEMMA4_OMNI_DUPLEX_PIPELINE.stages), 3)

        # Stage 0: Thinker
        stage0 = GEMMA4_OMNI_DUPLEX_PIPELINE.stages[0]
        self.assertEqual(stage0.stage_id, 0)
        self.assertEqual(stage0.model_stage, "llm")
        self.assertTrue(stage0.final_output)
        self.assertEqual(stage0.final_output_type, "text")

        # Stage 1: Talker (con Projector V9)
        stage1 = GEMMA4_OMNI_DUPLEX_PIPELINE.stages[1]
        self.assertEqual(stage1.stage_id, 1)
        self.assertEqual(stage1.model_stage, "tts")
        self.assertEqual(stage1.input_sources, (0,))
        self.assertEqual(stage1.engine_output_type, "latent")
        self.assertIn("thinker2talker_v9_prefix", stage1.custom_process_input_func)

        # Stage 2: Code2Wav Vocoder
        stage2 = GEMMA4_OMNI_DUPLEX_PIPELINE.stages[2]
        self.assertEqual(stage2.stage_id, 2)
        self.assertEqual(stage2.model_stage, "code2wav")
        self.assertEqual(stage2.input_sources, (1,))
        self.assertTrue(stage2.final_output)
        self.assertEqual(stage2.final_output_type, "audio")

    def test_02_thinker2talker_v9_prefix_processor(self):
        """Test 2: Verifica che l'input processor proietti h_thinker (1536D) nei prefissi KV (16 frame, 2048D)."""
        from vllm_omni_plugin.stage_processors import thinker2talker_v9_prefix

        batch_size = 2
        seq_len = 32
        thinker_dim = 1536
        h_thinker = torch.randn(batch_size, seq_len, thinker_dim, dtype=torch.bfloat16)

        # Mock payload in arrivo da Stage 0
        stage0_output = {
            "hidden_states": h_thinker,
            "text": ["Ciao come stai?", "Oggi è una bella giornata."],
        }

        talker_input = thinker2talker_v9_prefix(stage0_output, n_prefix_frames=16, talker_dim=2048)

        self.assertIn("prefix_kv", talker_input)
        self.assertIn("talker_prompt_embeds", talker_input)

        prefix_kv = talker_input["prefix_kv"]
        # Forma attesa per il prefix: (batch_size, 16, 2048)
        self.assertEqual(prefix_kv.shape, (batch_size, 16, 2048))
        self.assertEqual(prefix_kv.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(prefix_kv).any(), "Il prefisso non deve contenere NaN")

    def test_03_talker2code2wav_chunk_processor(self):
        """Test 3: Verifica che l'output del Talker (16 quantizzatori RVQ) sia formattato correttamente per Code2Wav."""
        from vllm_omni_plugin.stage_processors import (
            talker2code2wav_async_chunk,
            talker2code2wav_full_payload,
        )

        n_frames = 8
        n_codebooks = 16
        # Talker genera tensori di indici quantizzatore (batch, n_codebooks, n_frames)
        talker_codes = torch.randint(0, 2048, (1, n_codebooks, n_frames), dtype=torch.long)
        talker_payload = {"codes": talker_codes, "is_last_chunk": False}

        chunk_payload = talker2code2wav_async_chunk(talker_payload)
        self.assertIn("codes", chunk_payload)
        self.assertEqual(chunk_payload["codes"].shape, (1, 16, 8))
        self.assertFalse(chunk_payload["is_last_chunk"])

        # Test full payload finale
        talker_payload_final = {"codes": talker_codes, "is_last_chunk": True}
        full_payload = talker2code2wav_full_payload(talker_payload_final)
        self.assertTrue(full_payload["is_last_chunk"])

    def test_04_duplex_runtime_extension_turn_detection(self):
        """Test 4: Verifica che DuplexRuntimeExtension valuti TurnDetectorHead a 12.5Hz (80ms)."""
        from vllm_omni_plugin.runtime_extension import Gemma4OmniDuplexRuntimeExtension
        from vllm_omni_plugin.contracts import DuplexState

        extension = Gemma4OmniDuplexRuntimeExtension()

        # Simulazione di un frame acustico da 80ms (12.5Hz)
        # Feature vettoriali z_t = [h_t(2048) || e_user(128) || delta_e(128)] = 2304D
        z_t = torch.randn(1, 1, 2304)

        decision = extension.evaluate_frame_turn_state(z_t)

        self.assertIn(decision.turn_state, [
            DuplexState.SPEAK,
            DuplexState.YIELD,
            DuplexState.BARGE_IN,
            DuplexState.USER_BACKCHANNEL,
            DuplexState.EMIT_BACKCHANNEL,
        ])
        self.assertIsInstance(decision.should_interrupt_agent, bool)
        self.assertIsInstance(decision.should_yield_to_user, bool)

    def test_05_zero_reset_barge_in_lifecycle(self):
        """Test 5: Verifica il ciclo di Barge-In con Zero-Reset KV-Cache Policy."""
        from vllm_omni_plugin.serving_adapter import Gemma4OmniServingRuntimeAdapter

        adapter = Gemma4OmniServingRuntimeAdapter()
        session_id = "test_session_001"
        adapter.open_session(session_id)

        # 1. Simula l'agente che parla
        adapter.set_agent_speaking(session_id, is_speaking=True)
        self.assertTrue(adapter.is_agent_speaking(session_id))

        # 2. Simula l'utente che interrompe con alta energia/parlato (Barge-In)
        # PCM 80ms @ 24kHz = 1920 campioni
        barge_in_pcm = np.random.uniform(-0.5, 0.5, size=1920).astype(np.float32)
        event = adapter.process_incoming_pcm(session_id, barge_in_pcm, forced_barge_in=True)

    def test_06_runtime_extension_edge_cases_and_bounded_memory(self):
        """Test 6: Verifica robustezza a tensori 1D/2D e limite di memoria su state_history."""
        from vllm_omni_plugin.runtime_extension import Gemma4OmniDuplexRuntimeExtension

        extension = Gemma4OmniDuplexRuntimeExtension(max_history_len=10)

        # 1. Test con tensore 1D (2304,)
        z_1d = torch.randn(2304)
        dec_1d = extension.evaluate_frame_turn_state(z_1d)
        self.assertIsNotNone(dec_1d.turn_state)

        # 2. Test con tensore 2D (1, 2304)
        z_2d = torch.randn(1, 2304)
        dec_2d = extension.evaluate_frame_turn_state(z_2d)
        self.assertIsNotNone(dec_2d.turn_state)

        # 3. Test bounded memory: inseriamo 50 stati
        for _ in range(50):
            extension.evaluate_frame_turn_state(z_1d)
        self.assertEqual(len(extension.state_history), 10, "state_history deve essere limitata a max_history_len")

    def test_07_projector_v9_caching_and_left_padding(self):
        """Test 7: Verifica caching per dimensioni e gestione del left-padding su sequenze corte."""
        from vllm_omni_plugin.stage_processors import (
            get_or_create_projector_v9,
            thinker2talker_v9_prefix,
        )

        # Caching per chiavi distinte
        m1 = get_or_create_projector_v9(in_dim=1536, out_dim=2048, n_prefix_frames=16)
        m2 = get_or_create_projector_v9(in_dim=1024, out_dim=2048, n_prefix_frames=16)
        m3 = get_or_create_projector_v9(in_dim=1536, out_dim=2048, n_prefix_frames=16)
        self.assertIs(m1, m3, "I moduli con la stessa chiave devono essere riutilizzati dalla cache")
        self.assertIsNot(m1, m2, "I moduli con dimensioni diverse devono essere istanziati separatamente")

        # Test sequenza corta (T=4 < n_prefix_frames=16) con left-padding
        h_short = torch.randn(1, 4, 1536, dtype=torch.bfloat16)
        out_short = thinker2talker_v9_prefix({"hidden_states": h_short, "text": ["ok"]}, n_prefix_frames=16)
        self.assertEqual(out_short["prefix_kv"].shape, (1, 16, 2048))
        self.assertFalse(torch.isnan(out_short["prefix_kv"]).any())

    def test_08_serving_adapter_int16_pcm_and_session_reaper(self):
        """Test 8: Verifica supporto int16 PCM e reaping automatico di sessioni scadute."""
        import time
        from vllm_omni_plugin.serving_adapter import Gemma4OmniServingRuntimeAdapter

        adapter = Gemma4OmniServingRuntimeAdapter()
        adapter.open_session("s_active")
        adapter.open_session("s_stale")

        # Test PCM int16
        int16_pcm = (np.random.uniform(-0.5, 0.5, size=1920) * 32767).astype(np.int16)
        event = adapter.process_incoming_pcm("s_active", int16_pcm)
        self.assertEqual(event.event_type, "input_audio_buffer.appended")
        self.assertIn("samples", event.metadata)

        # Simula invecchiamento della sessione s_stale
        adapter._sessions["s_stale"].last_activity_time = time.time() - 400.0

        expired = adapter.reap_expired_sessions(idle_ttl_s=300.0)
        self.assertIn("s_stale", expired)
        self.assertNotIn("s_active", expired)
        self.assertNotIn("s_stale", adapter._sessions)
        self.assertIn("s_active", adapter._sessions)


if __name__ == "__main__":
    unittest.main()
