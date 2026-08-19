            # --- sintesi wav v3: varianti sampling residui + isolamento RVQ (diagnosi corruzione) ---
            SAMPLE_RATE = KP.SYNTH_SAMPLE_RATE
            import scipy.io.wavfile as wavfile
            _outd = "/kaggle/output" if os.path.exists("/kaggle/output") else "."
            _sub_dtype = next(sub_st2.parameters()).dtype

            EVAL_SAMPLING_TEMP = 0.9
            EVAL_SAMPLING_TOP_K = 50
            EVAL_SAMPLING_TOP_P = 1.0
            EVAL_SAMPLING_SEED = 1234

            _cand_idxs = val_idx[:min(len(val_idx), 15)]
            _scored_cands = []
            for _ci in _cand_idxs:
                try:
                    _nr = nreal_map.get(_ci, 0)
                    _na = min(_nr, ST2_T)
                    if _na <= 0:
                        continue
                    _gt_c = np.asarray(ds_train[_ci]["audio_targets"], dtype=np.int64)[:_na]
                    _w_c = decoder.decode(torch.from_numpy(_gt_c).long())
                    _w_arr = np.asarray(_w_c, dtype=np.float32)
                    _rms = float(np.sqrt(np.mean(np.square(_w_arr)))) if _w_arr.size > 0 else 0.0
                    _scored_cands.append((_ci, _rms, _nr, _na))
                except Exception:
                    pass

            _energetic = [c for c in _scored_cands if c[1] > 0.01]
            _energetic.sort(key=lambda x: x[1], reverse=True)
            if len(_energetic) >= 3:
                _chosen_cands = _energetic[:3]
            else:
                _scored_cands.sort(key=lambda x: x[1], reverse=True)
                _chosen_cands = _scored_cands[:min(3, len(_scored_cands))]
                if not _chosen_cands and val_idx:
                    _chosen_cands = [(val_idx[0], 0.0, nreal_map.get(val_idx[0], 0), min(nreal_map.get(val_idx[0], 0), ST2_T))]

            print(f"[{_eval_tag}] v3 finestre selezionate per sintesi:", flush=True)
            for _tag_c, _cand_info in zip(["a", "b", "c"], _chosen_cands):
                print(f"  [{_tag_c}] idx={_cand_info[0]} | n_real={_cand_info[2]} | n_active={_cand_info[3]} | rms_gt={_cand_info[1]:.6f}", flush=True)

            # Varianti di sampling sui 15 livelli residui (stesso seed, seed per variante)
            _VARIANTS = [
                ("greedy", {"temperature": 0.0}),
                ("t01",    {"temperature": 0.1}),
                ("t03",    {"temperature": 0.3}),
                ("t09",    {"temperature": 0.9}),
                ("tdecay", {"temperatures_per_level": [round(0.8 * (0.75 ** (g - 1)), 6) for g in range(1, 16)]}),
            ]

            def _sample_variant(sub, h, cb0, var_cfg):
                return sub.sample_acoustic_codes(h, cb0, top_k=EVAL_SAMPLING_TOP_K,
                                                 top_p=EVAL_SAMPLING_TOP_P, seed=EVAL_SAMPLING_SEED, **var_cfg)

            synth_results = []
            _rvq_results = []
            for _tag, _cand_info in zip(["a", "b", "c"], _chosen_cands):
                _i, _rms_cand, _nr, _na = _cand_info
                try:
                    _gt_full = np.asarray(ds_train[_i]["audio_targets"], dtype=np.int64)[:_na]
                    _oracle = _gt_full[:, 1:]
                    cb0 = torch.from_numpy(_gt_full[:, 0])
                    _hp = torch.from_numpy(h_pred_map[_i][:_na]).to(DEV2, dtype=_sub_dtype).unsqueeze(0)
                    _ht = torch.from_numpy(h_tgt_map[_i][:_na]).to(DEV2, dtype=_sub_dtype).unsqueeze(0)
                    _cb0 = cb0.unsqueeze(0).to(DEV2)
                    _gt_full_t = torch.from_numpy(_gt_full).long()

                    # 1. Riferimento massimo GT
                    wav_gt = decoder.decode(_gt_full_t)
                    wav_gt_arr = np.asarray(wav_gt, dtype=np.float32)
                    rms_gt = float(np.sqrt(np.mean(np.square(wav_gt_arr)))) if wav_gt_arr.size > 0 else 0.0
                    _f_gt_name = f"{_eval_wav_pfx}_gt_ref_{_tag}.wav"
                    wavfile.write(os.path.join(_outd, _f_gt_name), SAMPLE_RATE,
                                  (wav_gt_arr * 32767).astype(np.int16))

                    # 2. Varianti di sampling per oracle e pred
                    _sampled = {}
                    with torch.inference_mode():
                        for _vname, _vcfg in _VARIANTS:
                            _sampled[("oracle", _vname)] = _sample_variant(sub_st2, _ht, _cb0, _vcfg)
                            _sampled[("pred", _vname)] = _sample_variant(sub_st2, _hp, _cb0, _vcfg)

                    _acc = {}
                    _wav_names = {}
                    for _src in ("oracle", "pred"):
                        for _vname, _vcfg in _VARIANTS:
                            _ac = _sampled[(_src, _vname)].squeeze(0).cpu().long()
                            _full = torch.cat([cb0.unsqueeze(-1), _ac], dim=-1)
                            _gen = _full[:, 1:].cpu().numpy()
                            _acc[f"{_src}_{_vname}"] = float((_gen == _oracle).mean()) if _na > 0 else 0.0
                            _wv = decoder.decode(_full)
                            _fname = f"{_eval_wav_pfx}_{_src}_{_tag}_{_vname}.wav"
                            wavfile.write(os.path.join(_outd, _fname), SAMPLE_RATE,
                                          (np.asarray(_wv, dtype=np.float32) * 32767).astype(np.int16))
                            _wav_names[f"{_src}_{_vname}"] = _fname

                    # 3. Isolamento RVQ sulla finestra a (e su tutte, costo basso): GT livelli 0..N + zeri
                    _zeros = torch.zeros_like(_gt_full_t)
                    for _N in (0, 1, 3, 7, 15):
                        _mix = _gt_full_t.clone()
                        if _N < 15:
                            _mix[:, _N + 1:] = _zeros[:, _N + 1:]
                        _wv = decoder.decode(_mix)
                        _fname = f"{_eval_wav_pfx}_rvq_gtz_{_tag}_N{_N}.wav"
                        wavfile.write(os.path.join(_outd, _fname), SAMPLE_RATE,
                                      (np.asarray(_wv, dtype=np.float32) * 32767).astype(np.int16))
                        _rvq_results.append({"tag": _tag, "kind": "gt+zeros", "N": _N, "file": _fname})

                    # 4. Isolamento RVQ: GT livelli 0..N + residui greedy del modello (oracle e pred)
                    _gr_t = _sampled[("oracle", "greedy")].squeeze(0).cpu().long()
                    _gr_p = _sampled[("pred", "greedy")].squeeze(0).cpu().long()
                    for _src, _gr in (("oracle", _gr_t), ("pred", _gr_p)):
                        for _N in (0, 1, 3, 7):
                            _mix = _gt_full_t.clone()
                            _mix[:, _N + 1:] = _gr[:, _N + 1:]
                            _wv = decoder.decode(_mix)
                            _fname = f"{_eval_wav_pfx}_rvq_{_src}_{_tag}_N{_N}.wav"
                            wavfile.write(os.path.join(_outd, _fname), SAMPLE_RATE,
                                          (np.asarray(_wv, dtype=np.float32) * 32767).astype(np.int16))
                            _rvq_results.append({"tag": _tag, "kind": f"gt+{_src}_greedy", "N": _N, "file": _fname})

                    print(f"[{_eval_tag}] v3 synth {_tag}: idx={_i} n_active={_na} rms_gt={rms_gt:.4f}", flush=True)
                    for _src in ("oracle", "pred"):
                        _acc_line = " | ".join(f"{_v}={_acc[f'{_src}_{_v}']:.4f}" for _v, _ in _VARIANTS)
                        print(f"  {_src}: {_acc_line}", flush=True)
                    print(f"  rvq: 10 wav salvate (gtz N=0,1,3,7,15 + gt+oracle/pred greedy N=0,1,3,7)", flush=True)

                    synth_results.append({
                        "tag": _tag,
                        "idx": int(_i),
                        "n_active": int(_na),
                        "rms_gt": round(rms_gt, 6),
                        "acc": {k: round(v, 4) for k, v in _acc.items()},
                        "sampling": {"top_k": EVAL_SAMPLING_TOP_K, "top_p": EVAL_SAMPLING_TOP_P, "seed": EVAL_SAMPLING_SEED},
                        "variants": [v for v, _ in _VARIANTS],
                        "files": {"gt_ref": _f_gt_name, **_wav_names},
                        "status": "OK",
                    })
                except Exception as _e_win:
                    print(f"[{_eval_tag}] ERRORE sintesi v3 finestra {_tag} (idx={_i}): {_e_win}", flush=True)
                    synth_results.append({"tag": _tag, "idx": int(_i), "status": "ERROR", "error": str(_e_win)})

            _acc_flat = [v for s in synth_results if "acc" in s for v in s["acc"].values()]
            report = {
                "status": "OK",
                "phase": f"EVAL_{_eval_name}",
                "ckpt": _ck,
                "n_val": len(val_idx),
                "val_idx_range": [val_idx[0], val_idx[-1]],
                "cos_sim": round(cos_v, 4),
                "mse": round(mse_v, 4),
                "coupling_ce": round(ce_v, 4),
                "rank_svd_pred": rk_p,
                "rank_svd_target": rk_t,
                "cum_energy_pred": round(cum_p, 4),
                "cum_energy_target": round(cum_t, 4),
                "sampling_protocol": {"temperature": EVAL_SAMPLING_TEMP, "top_k": EVAL_SAMPLING_TOP_K,
                                      "top_p": EVAL_SAMPLING_TOP_P, "seed": EVAL_SAMPLING_SEED},
                "variants": [v for v, _ in _VARIANTS],
                "synth_samples": synth_results,
                "rvq_isolation": _rvq_results,
                "mean_rms_gt": round(float(np.mean([s["rms_gt"] for s in synth_results if "rms_gt" in s])), 6) if any("rms_gt" in s for s in synth_results) else None,
                "gates": {
                    "cos_sim_ge_0.88": bool(cos_v >= 0.88),
                    "ce_val_le_5.95": bool(ce_v <= 5.95),
                    "rank_ge_256": bool(rk_p >= 256),
                },
            }
            if _eval_is_r2:
                report["variant"] = "r2"
            with open(os.path.join(_outd, _eval_rep_file), "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(json.dumps(report, indent=2), flush=True)
            print(f">>> EVAL {_eval_name} COMPLETATO <<<", flush=True)
