import os, sys, json, traceback, time, math, gc
import torch
torch.manual_seed(0)
class _MockSub:
    def __init__(self):
        self.cfg = {"num_code_groups": 16, "vocab_size": 2048, "talker_hidden_size": 8}
    def _forward_flat(self, h_flat, full_codes):
        B = full_codes.shape[0]; V = self.cfg["vocab_size"]
        out = torch.zeros(B, 15, V)
        for g in range(15):
            out[:, g, :] = (full_codes[:, g] % V).unsqueeze(-1).float() * 0.01
            out[:, g, (g + 3) % V] += 5.0
        return out
    def sample_acoustic_codes(self):
        return "old"
QS = type("QS", (), {})()
QS.Qwen3TTSSubTalker = _MockSub
if True:
        # ---- PATCH sample_acoustic_codes aggiornata (top_k/top_p/seed/per-level) ----
        # Il dataset input porta qwen_subtalker.py VECCHIO (firma senza top_k): la
        # synth v3 falliva con TypeError. Si sostituisce il metodo sulla classe con
        # la versione del repo (stessa classe, stesso _forward_flat, stesso cfg).
        try:
            import inspect as _inspect_p
            _sig_p = _inspect_p.signature(QS.Qwen3TTSSubTalker.sample_acoustic_codes)
            if 'top_k' not in _sig_p.parameters:
                print('[PATCH] upgrade sample_acoustic_codes (top_k/top_p/seed/per-level)', flush=True)
                def _sample_acoustic_codes_v2(self, h_t, cb0, temperature=0.0, top_k=50, top_p=1.0,
                        seed=None, device=None, warn_cb0_clamp=True, temperatures_per_level=None):
                    if temperatures_per_level is not None:
                        expected_len = self.cfg["num_code_groups"] - 1
                        if len(temperatures_per_level) != expected_len:
                            raise ValueError(
                                f"temperatures_per_level ha lunghezza {len(temperatures_per_level)}, "
                                f"attesa {expected_len} (num_code_groups - 1)."
                            )
    
                    if seed is not None:
                        torch.manual_seed(seed)
    
                    B, T = cb0.shape
                    dev = device if device is not None else cb0.device
                    cb0 = cb0.to(dev).long()
                    # cb0 dal semantic_head di Gemma ha vocabolario 4096: clamp nel range
                    # semantico GT [0, 2048) per non uscire dalla distribuzione del teacher.
                    if cb0.numel() > 0 and cb0.max() >= self.cfg["vocab_size"]:
                        if warn_cb0_clamp:
                            print(
                                f"[QwenSubTalker] [WARN] clamp cb0 campionato: max={int(cb0.max().item())} -> "
                                f"{self.cfg['vocab_size'] - 1} (semantic_head 4096 vs range GT 2048)."
                            )
                        cb0 = cb0.clamp(max=self.cfg["vocab_size"] - 1)
    
                    h_flat = h_t.reshape(B * T, 1, self.cfg["talker_hidden_size"])
                    full = torch.zeros(B, T, self.cfg["num_code_groups"], dtype=torch.long, device=dev)
                    full[:, :, 0] = cb0
                    seq = cb0.unsqueeze(-1)  # (B, T, 1)
                    V = self.cfg["vocab_size"]
                    for g in range(1, self.cfg["num_code_groups"]):
                        # full: col0=cb0, col1..g-1 = codici gia' campionati, resto 0 (junk:
                        # la causalita' rende innocue le posizioni future per la posizione g)
                        if g > 1:
                            full[:, :, 1:g] = seq[:, :, 1:]
                        logits = self._forward_flat(h_flat, full.reshape(B * T, self.cfg["num_code_groups"]))  # (B*T, 15, V)
                        lg = logits[:, g - 1, :].reshape(B, T, V)  # predice cb_g
    
                        lvl_temp = (
                            float(temperatures_per_level[g - 1])
                            if temperatures_per_level is not None
                            else float(temperature)
                        )
    
                        if lvl_temp <= 0.0:
                            sampled = lg.argmax(dim=-1)
                        else:
                            logits_step = lg / lvl_temp
    
                            # 1. Filtro Top-K (HF style)
                            if top_k is not None and 0 < top_k < V:
                                topk_vals, _ = torch.topk(logits_step, min(int(top_k), V), dim=-1)
                                kth_val = topk_vals[..., -1, None]
                                logits_step = torch.where(
                                    logits_step < kth_val,
                                    torch.tensor(float("-inf"), device=logits_step.device, dtype=logits_step.dtype),
                                    logits_step,
                                )
    
                            # 2. Filtro Top-P / Nucleus (HF style)
                            if top_p is not None and 0.0 < top_p < 1.0:
                                sorted_logits, sorted_indices = torch.sort(logits_step, descending=True, dim=-1)
                                sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
                                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
                                # Maschera token sopra la soglia cumulativa top_p
                                sorted_indices_to_remove = cumulative_probs > top_p
                                # Shift a destra di 1 per mantenere almeno il primo token che sfora
                                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                                sorted_indices_to_remove[..., 0] = False
    
                                indices_to_remove = sorted_indices_to_remove.scatter(
                                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                                )
                                logits_step = logits_step.masked_fill(indices_to_remove, float("-inf"))
    
                            probs = torch.nn.functional.softmax(logits_step, dim=-1)
                            flat_probs = probs.reshape(-1, V)
                            sampled = torch.multinomial(flat_probs, num_samples=1).reshape(B, T)
    
                        seq = torch.cat([seq, sampled.unsqueeze(-1)], dim=-1)
                    return seq[:, :, 1:]  # (B, T, 15)
                QS.Qwen3TTSSubTalker.sample_acoustic_codes = _sample_acoustic_codes_v2
        except Exception:
            traceback.print_exc()
            sub = _MockSub()
            h = torch.randn(2, 4, 8)
            cb0 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
            c1 = sub.sample_acoustic_codes(h, cb0, temperature=0.0, top_k=50, top_p=1.0, seed=7)
            assert c1.shape == (2, 4, 15), c1.shape
            c2 = sub.sample_acoustic_codes(h, cb0, temperature=0.9, top_k=50, top_p=1.0, seed=7)
            assert c2.shape == (2, 4, 15)
            c3 = sub.sample_acoustic_codes(h, cb0, temperatures_per_level=[0.8*(0.75**(g-1)) for g in range(1,16)], top_k=50, seed=7)
            assert c3.shape == (2, 4, 15)
            print("TEST A GREEN: patch applicato, 3 sampling OK")
