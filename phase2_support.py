# Funzioni Fase 2 non incluse come celle del notebook
# (caricate dal pipeline locale via kaggle_execute)

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
        user_mel, agent_mel, agent_codes, text_by_window = load_tokens_from_parquet(
            parquet_cache
        )
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

    dataset = FullDuplexDataset(user_mel, agent_mel, agent_codes, seq_len=64)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    print(
        f"[Fase 2 - Training] ✅ COUPLING ATTIVO (ADR-005): codebook 1..15 dal sub-talker Qwen congelato "
        f"(loss peso {COUPLING_LOSS_WEIGHT}, warm-up {COUPLING_WARMUP_STEPS} step con h_t.detach())."
    )
    sys.stdout.flush()

    if duplex_model is not None:
        optimizer = torch.optim.AdamW(
            duplex_model.parameters(), lr=2e-4, weight_decay=0.01
        )
        duplex_model.train()
        print(
            f"[Fase 2 - Training] Avvio ciclo di ottimizzazione su {len(dataloader)} batch per epoca..."
        )
        sys.stdout.flush()

        for epoch in range(num_epochs):
            pbar = tqdm(
                dataloader, desc=f"Epoca {epoch + 1}/{num_epochs}", unit="batch"
            )
            # Device di riferimento del modello base (PEFT/LoRA su cuda:0)
            base_device = next(duplex_model.base_model.parameters()).device

            for step, batch in enumerate(pbar):
                # SMOKE temporaneo: limita il numero di batch di training via env
                # (PHASE2_SMOKE_BATCHES=2 -> 2 batch, poi validation+checkpoint+sintesi).
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
    print("      * samples/user_input.wav  oppure  /kaggle/input/<dataset>/parlato.wav")
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


