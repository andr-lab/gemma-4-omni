from __future__ import annotations

import io
import logging
import unittest
from unittest.mock import patch

import torch

from duplex_kernel import server


class _DiagnosticTokenizer:
    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return {55: "ciao", 56: " Marco", 256002: "<unused90>"}[int(token_ids[0])]


class AlignmentFailureDiagnosticsTests(unittest.TestCase):
    def test_h_txt_alignment_failure_logs_reproducible_context_and_reraises_original(self):
        original_error = ValueError("token pieces do not reconstruct expected text")

        def fail_alignment(*_args, **_kwargs):
            raise original_error

        original_align = server.align_generated_response_hidden_to_qwen_tokens
        logger = logging.getLogger("duplex_kernel.server")
        original_level = logger.level
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
        try:
            server.align_generated_response_hidden_to_qwen_tokens = fail_alignment
            tokenizer = _DiagnosticTokenizer()
            raw_hidden = torch.zeros((1, 10, 4))
            say_hidden = torch.zeros((1, 2, 4))
            with self.assertRaises(ValueError) as caught:
                server._align_generated_response_hidden_with_diagnostics(
                    response_hidden=say_hidden,
                    visible_generated_ids=[55, 56],
                    gemma_tokenizer=tokenizer,
                    qwen_tokenizer=object(),
                    qwen_processor=object(),
                    generated_text="Ciao Marco",
                    generated_ids=[55, 56, 256002],
                    hidden_prompt_len=8,
                    raw_hidden=raw_hidden,
                )
            self.assertIs(caught.exception, original_error)
        finally:
            server.align_generated_response_hidden_to_qwen_tokens = original_align
            logger.removeHandler(handler)
            logger.setLevel(original_level)

        log_text = output.getvalue()
        self.assertIn("token pieces do not reconstruct expected text", log_text)
        self.assertIn('"generated_ids": [55, 56, 256002]', log_text)
        self.assertIn('"visible_generated_ids": [55, 56]', log_text)
        self.assertIn('"generated_text_repr": "\'Ciao Marco\'"', log_text)
        self.assertIn('"per_token_decode_pieces"', log_text)
        self.assertIn("<unused90>", log_text)
        self.assertIn('"hidden_prompt_len": 8', log_text)
        self.assertIn('"raw_hidden_token_count": 10', log_text)
        self.assertIn('"say_hidden_token_count": 2', log_text)

    def test_diagnostic_serialization_cannot_mask_original_alignment_exception(self):
        original_error = ValueError("strict h_txt alignment error")

        def fail_alignment(*_args, **_kwargs):
            raise original_error

        original_align = server.align_generated_response_hidden_to_qwen_tokens
        logger = logging.getLogger("duplex_kernel.server")
        try:
            server.align_generated_response_hidden_to_qwen_tokens = fail_alignment
            with patch.object(logger, "exception", side_effect=RuntimeError("logger unavailable")):
                with self.assertRaises(ValueError) as caught:
                    server._align_generated_response_hidden_with_diagnostics(
                        response_hidden=torch.zeros((1, 2, 4)),
                        visible_generated_ids=[55],
                        gemma_tokenizer=_DiagnosticTokenizer(),
                        qwen_tokenizer=object(),
                        qwen_processor=object(),
                        generated_text="Ciao",
                        generated_ids=["malformed-token-id"],
                        hidden_prompt_len=8,
                        raw_hidden=torch.zeros((1, 10, 4)),
                    )
            self.assertIs(caught.exception, original_error)
        finally:
            server.align_generated_response_hidden_to_qwen_tokens = original_align


if __name__ == "__main__":
    unittest.main()
