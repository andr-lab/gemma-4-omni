"""Command-line entrypoints for the duplex QLoRA pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prepare import prepare_splits
from .sources import (
    build_real_conversational_views,
    read_duplexconv,
    read_smoothconv,
    read_synthetic_v3_clean_asr,
    read_synthetic_v3_manifest,
    summarize_preparation_counts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prep = commands.add_parser("prepare", help="convert source audio/transcripts and prepare split JSONL")
    prep.add_argument("--synthetic-root", "--synthetic-v3", dest="synthetic_root", type=Path, default=None,
                      help="synthetic Italian v3 root directory or manifest.jsonl")
    prep.add_argument("--synthetic-v3-clean-asr-root", type=Path, default=None,
                      help="optional v3 source for exact, complete, slot-free role-free ASR utterances")
    prep.add_argument("--smoothconv-root", type=Path, default=None,
                      help="SmoothConv slice directory or project data root")
    prep.add_argument("--smoothconv-evidence", type=Path, default=None,
                      help="optional label evidence JSONL; resolved from data/training_gates by default")
    prep.add_argument("--duplexconv-root", type=Path, default=None,
                      help="DuplexConv slice directory or project data root")
    prep.add_argument("--duplexconv-pairing", type=Path, default=None)
    prep.add_argument("--duplexconv-annotation-root", type=Path, action="append", default=None,
                      help="optional annotation directory; may be passed more than once")
    prep.add_argument("--real-conversational-views", action="store_true",
                      help="opt in to two directed perspectives and conservative 80 ms control labels for verified Smooth/Duplex stereo pairs")
    prep.add_argument("--real-conversational-max-conversations", type=int, default=128,
                      help="maximum original real conversations selected by native-label coverage; paired groups stay together (default: 128)")
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=20260930)

    audio_prep = commands.add_parser(
        "prepare-audio-responses",
        help="build direct native-audio response examples with causal cutoffs and no current-user transcript prompt",
    )
    audio_prep.add_argument("--prepared-dir", type=Path, required=True,
                            help="conversation-level prepared split JSONL directory")
    audio_prep.add_argument("--output-dir", type=Path, required=True)
    audio_prep.add_argument("--max-audio-seconds", type=float, default=30.0)
    audio_prep.add_argument("--response-protocol", choices=("v1", "v2"), default="v1",
                            help="v1 risposta testuale legacy o v2 JSON target heard/action/say")

    audio_asr_mix = commands.add_parser(
        "prepare-audio-asr-mix",
        help="join direct audio responses and clean native-audio ASR with disjoint held-out groups",
    )
    audio_asr_mix.add_argument("--prepared-dir", type=Path, required=True,
                               help="prepared conversation train/validation/test JSONL")
    audio_asr_mix.add_argument("--audio-response-dir", type=Path, required=True,
                               help="output di prepare-audio-responses")
    audio_asr_mix.add_argument("--output-dir", type=Path, required=True)
    audio_asr_mix.add_argument("--validation-asr-examples", type=int, default=32)
    audio_asr_mix.add_argument("--seed", type=int, default=20261002)
    audio_asr_mix.add_argument("--response-protocol", choices=("v1", "v2"), default="v1",
                               help="protocollo delle righe di risposta; ASR mantiene il proprio formato native-audio")

    audio_grounded_v5 = commands.add_parser(
        "prepare-audio-grounded-v5",
        help="verify candidate v5 WAVs and merge them into prepared v4 + clean-ASR splits",
    )
    audio_grounded_v5.add_argument("--candidate-dir", type=Path, required=True,
                                   help="candidate root with manifest_audio_grounded_v5.jsonl and WAVs; never smoke")
    audio_grounded_v5.add_argument("--base-prepared-dir", type=Path, required=True,
                                   help="prepared v4 + synthetic_v3_clean_asr train/validation/test splits")
    audio_grounded_v5.add_argument("--output-dir", type=Path, required=True)

    train = commands.add_parser("train", help="train Gemma backbone LoRA from audio/text conversation examples")
    train.add_argument("--train-jsonl", type=Path, required=True)
    train.add_argument("--preparation-manifest", type=Path, default=None)
    train.add_argument("--validation-jsonl", type=Path, default=None,
                       help="held-out direct-audio split; required by audio_response_conditioning_v1")
    train.add_argument("--model-path", default="google/gemma-4-e2b-it")
    train.add_argument("--adapter-dir", type=Path, default=Path("checkpoints/gemma4_duplex_qlora/adapter"))
    train.add_argument("--max-steps", type=int, default=1000)
    train.add_argument("--checkpoint-interval", type=int, default=100,
                       help="salva checkpoint recuperabili ogni N optimizer update")
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--audio-code-loss-weight", type=float, default=0.0)
    train.add_argument(
        "--conditioning-mode",
        choices=("baselineaudio_response", "runtime_asr_text", "audio_asr_mix_v1", "audio_asr_mix_v2"),
        required=True,
        help="esplicita se Stage1 usa audio response sperimentale o lo stesso transcript-to-text del runtime",
    )
    train.add_argument("--smoke", action="store_true", help="limit to 4 optimizer updates; GPU work still occurs")
    train.add_argument("--seed", type=int, default=20260930)
    train.add_argument("--resume-from", type=Path, default=None,
                       help="Stage1 step_N checkpoint directory; restores adapter, AdamW and RNG state")
    train.add_argument("--initial-adapter-dir", type=Path, default=None,
                       help="warm-start adapter PEFT trainable with fresh AdamW; distinct from --resume-from")
    train.add_argument("--schedule-jsonl", type=Path, default=None,
                       help="ordered train-record IDs for a task/source schedule; requires audio_asr_mix_v2")
    train.add_argument(
        "--use-cache-for-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use Gemma4's native DynamicCache KV-sharing semantics during training (required by Gemma4; checkpointing is disabled)",
    )

    extract = commands.add_parser("extract-control", help="merge final LoRA and extract native listener frames")
    extract.add_argument("--prepared-dir", type=Path, required=True)
    extract.add_argument("--base-model-path", required=True)
    extract.add_argument("--adapter-dir", required=True)
    extract.add_argument("--output-dir", type=Path, default=Path("checkpoints/gemma4_duplex_qlora/features"))
    extract.add_argument("--device", default="cuda")
    extract.add_argument(
        "--benchmark-only",
        action="store_true",
        help="merge the adapter and measure eight bounded native target clusters without extracting a dataset",
    )
    extract.add_argument(
        "--benchmark-then-extract-if-under-hours",
        type=float,
        default=None,
        metavar="HOURS",
        help="benchmark clusters and continue in the same loaded model process only when projection is within this limit",
    )
    extract.add_argument(
        "--verify-prefix-cache",
        action="store_true",
        help="compare cold and cached native canonical-prefix restore on the same eight real target windows",
    )

    head = commands.add_parser("train-control", help="train the separate six-action listener head")
    head.add_argument("--feature-dir", type=Path, required=True)
    head.add_argument("--output", type=Path, default=Path("checkpoints/gemma4_duplex_qlora/control_head.pt"))
    head.add_argument("--training-manifest", type=Path, default=Path("checkpoints/gemma4_duplex_qlora/training_manifest.json"))
    head.add_argument("--max-steps", type=int, default=500)
    head.add_argument("--learning-rate", type=float, default=1e-3)
    head.add_argument(
        "--architecture",
        choices=("linear", "mlp_256_gelu_dropout_0.1"),
        default="linear",
        help="head separato opzionale; il default lineare mantiene la compatibilità dei checkpoint esistenti",
    )
    head.add_argument(
        "--selection-metric",
        choices=("validation_natural_macro_f1",),
        default=None,
        help="seleziona il miglior checkpoint usando solo macro-F1 naturale validation",
    )
    head.add_argument("--selection-interval", type=int, default=25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        if args.real_conversational_views and not (args.smoothconv_root or args.duplexconv_root):
            raise SystemExit("--real-conversational-views richiede --smoothconv-root e/o --duplexconv-root")
        if args.real_conversational_views and args.real_conversational_max_conversations <= 0:
            raise SystemExit("--real-conversational-max-conversations deve essere positivo")
        conversations = []
        external_conversations = []
        if args.synthetic_root:
            conversations.extend(read_synthetic_v3_manifest(args.synthetic_root))
        if args.synthetic_v3_clean_asr_root:
            conversations.extend(read_synthetic_v3_clean_asr(args.synthetic_v3_clean_asr_root))
        if args.smoothconv_root:
            external_conversations.extend(read_smoothconv(args.smoothconv_root, evidence_path=args.smoothconv_evidence))
        if args.duplexconv_root:
            external_conversations.extend(read_duplexconv(
                args.duplexconv_root,
                pairing_path=args.duplexconv_pairing,
                annotation_roots=args.duplexconv_annotation_root,
            ))
        if args.real_conversational_views:
            if not external_conversations:
                raise SystemExit("--real-conversational-views richiede --smoothconv-root e/o --duplexconv-root")
            views = build_real_conversational_views(
                external_conversations,
                max_conversations=args.real_conversational_max_conversations,
            )
            if not views:
                raise SystemExit("nessuna conversazione SmoothConv/DuplexConv con pairing stereo verificato è eleggibile")
            conversations.extend(views)
        else:
            conversations.extend(external_conversations)
        if not conversations:
            raise SystemExit(
                "prepare richiede una sorgente sintetica o audio tra --synthetic-root, "
                "--synthetic-v3-clean-asr-root, --smoothconv-root e --duplexconv-root; "
                "MDC/CV26 non sono incluse nella pipeline predefinita"
            )
        result = prepare_splits(conversations, args.output_dir, seed=args.seed)
        result["source_readiness"] = summarize_preparation_counts(conversations)
        if args.real_conversational_views:
            action_counts: dict[str, int] = {}
            for view in views:
                for frame in view.get("frames", ()) or ():
                    action = frame.get("target_action")
                    if frame.get("action_mask") is True and isinstance(action, str):
                        action_counts[action] = action_counts.get(action, 0) + 1
            result["real_conversational_view_selection"] = {
                "eligible_verified_pair_conversations_seen": int(views[0]["source_view_eligible_conversations_seen"]),
                "selected_source_conversations": int(views[0]["source_view_selected_conversations"]),
                "directed_views": len(views),
                "max_source_conversations": args.real_conversational_max_conversations,
                "source_groups": len({str(view.get("source_group_id")) for view in views}),
                "sparse_control_frame_candidates": sum(len(view.get("frames", ()) or ()) for view in views),
                "supervised_control_frames": sum(action_counts.values()),
                "supervised_frames_by_action": dict(sorted(action_counts.items())),
                "selected_views_by_source": {
                    source: sum(view.get("source") == source for view in views)
                    for source in sorted({str(view.get("source")) for view in views})
                },
            }
    elif args.command == "prepare-audio-responses":
        if not 0 < args.max_audio_seconds <= 30:
            raise SystemExit("--max-audio-seconds deve essere >0 e <=30")
        from .action_conditioned_data import prepare_action_conditioned_dataset

        result = prepare_action_conditioned_dataset(
            args.prepared_dir,
            args.output_dir,
            max_audio_ms=round(args.max_audio_seconds * 1000),
            response_protocol=("heard_action_say_v2" if args.response_protocol == "v2" else "audio_response_conditioning_v1"),
        )
    elif args.command == "prepare-audio-asr-mix":
        from .action_conditioned_data import prepare_audio_asr_mix_dataset

        result = prepare_audio_asr_mix_dataset(
            args.prepared_dir,
            args.audio_response_dir,
            args.output_dir,
            validation_asr_examples=args.validation_asr_examples,
            seed=args.seed,
            response_protocol=args.response_protocol,
        )
    elif args.command == "prepare-audio-grounded-v5":
        from .action_conditioned_data import prepare_audio_grounded_v5_dataset

        result = prepare_audio_grounded_v5_dataset(
            args.candidate_dir,
            args.base_prepared_dir,
            args.output_dir,
        )
    elif args.command == "train":
        from .stage1 import train_qlora

        result = train_qlora(
            train_jsonl=args.train_jsonl,
            preparation_manifest=args.preparation_manifest,
            validation_jsonl=args.validation_jsonl,
            model_path=args.model_path,
            adapter_dir=args.adapter_dir,
            max_steps=4 if args.smoke else args.max_steps,
            checkpoint_interval=args.checkpoint_interval,
            learning_rate=args.learning_rate,
            audio_code_loss_weight=args.audio_code_loss_weight,
            seed=args.seed,
            resume_from=args.resume_from,
            initial_adapter_dir=args.initial_adapter_dir,
            schedule_jsonl=args.schedule_jsonl,
            conditioning_mode=args.conditioning_mode,
            use_cache_for_training=args.use_cache_for_training,
        )
    elif args.command == "extract-control":
        from .stage2 import extract_stage2_features

        result = extract_stage2_features(
            prepared_dir=args.prepared_dir,
            base_model_path=args.base_model_path,
            adapter_dir=args.adapter_dir,
            output_dir=args.output_dir,
            device=args.device,
            benchmark_only=args.benchmark_only,
            benchmark_then_extract_if_under_hours=args.benchmark_then_extract_if_under_hours,
            verify_prefix_cache=args.verify_prefix_cache,
        )
    else:
        from .stage2 import train_control_head

        result = train_control_head(
            feature_dir=args.feature_dir,
            output_path=args.output,
            training_manifest_path=args.training_manifest,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            architecture=args.architecture,
            selection_metric=args.selection_metric,
            selection_interval=args.selection_interval,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
