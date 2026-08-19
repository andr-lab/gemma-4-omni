import sys
import torch
import torch.nn.functional as F
import qwen_subtalker as QS

def run_tests():
    print('=== RUNNING SAMPLING TDD SUITE ===')
    torch.manual_seed(42)
    cfg = dict(QS.DEFAULT_CONFIG)
    cfg['num_hidden_layers'] = 2
    model = QS.Qwen3TTSSubTalker(cfg).eval()

    B, T = 2, 3
    h_t = torch.randn(B, T, cfg['talker_hidden_size'])
    cb0 = torch.randint(0, cfg['vocab_size'], (B, T))

    failures = []
    def check(name, cond, extra=''):
        status = 'PASS' if cond else 'FAIL'
        print(f'  [{status}] {name}' + (f' | {extra}' if extra else ''))
        if not cond:
            failures.append((name, extra))

    # Test 1: temperature <= 0.0 -> shape (B, T, 15) e identico ad argmax (greedy)
    try:
        out_greedy = model.sample_acoustic_codes(h_t, cb0, temperature=0.0)
        check('Test 1a: Shape greedy is (B, T, 15)', out_greedy.shape == (B, T, 15), f'got {out_greedy.shape}')
        check('Test 1b: Greedy output non-empty', out_greedy.numel() == B * T * 15)
    except Exception as e:
        check('Test 1: Greedy sampling', False, str(e))

    # Test 2: temperature > 0 con seed fisso -> RIPRODUCIBILITA
    try:
        out_s1 = model.sample_acoustic_codes(h_t, cb0, temperature=0.9, top_k=50, top_p=1.0, seed=1234)
        out_s2 = model.sample_acoustic_codes(h_t, cb0, temperature=0.9, top_k=50, top_p=1.0, seed=1234)
        check('Test 2: Reproducibility with same seed', torch.equal(out_s1, out_s2))
    except Exception as e:
        check('Test 2: Reproducibility with same seed', False, str(e))

    # Test 3: temperature > 0 -> diverso da greedy
    try:
        out_s = model.sample_acoustic_codes(h_t, cb0, temperature=0.9, top_k=50, top_p=1.0, seed=9999)
        diff_count = (out_s != out_greedy).sum().item()
        check('Test 3: Stochastic sampling differs from greedy', diff_count > 0, f'differing tokens: {diff_count}/{B*T*15}')
    except Exception as e:
        check('Test 3: Stochastic sampling differs from greedy', False, str(e))

    # Test 4: top_k=1, temperature > 0 -> coincide con argmax
    try:
        out_top1 = model.sample_acoustic_codes(h_t, cb0, temperature=0.9, top_k=1, top_p=1.0, seed=42)
        check('Test 4: top_k=1 matches greedy argmax', torch.equal(out_top1, out_greedy))
    except Exception as e:
        check('Test 4: top_k=1 matches greedy argmax', False, str(e))

    # Test 5: top_k=5 -> token campionati sono strettamente nel set top-5 dei logit
    try:
        # Per verificare top_k=5 sul primo step (g=1):
        # Eseguiamo _forward_flat con cb0
        h_flat = h_t.reshape(B * T, 1, cfg['talker_hidden_size'])
        full = torch.zeros(B, T, cfg['num_code_groups'], dtype=torch.long)
        full[:, :, 0] = cb0
        logits = model._forward_flat(h_flat, full.reshape(B * T, cfg['num_code_groups']))
        lg1 = logits[:, 0, :].reshape(B, T, cfg['vocab_size'])
        top5_indices = torch.topk(lg1, 5, dim=-1).indices # (B, T, 5)

        out_top5 = model.sample_acoustic_codes(h_t, cb0, temperature=0.9, top_k=5, top_p=1.0, seed=777)
        cb1_sampled = out_top5[:, :, 0] # (B, T)
        # Check if cb1_sampled is in top5_indices
        in_top5 = False
        match_count = 0
        for b in range(B):
            for t in range(T):
                if cb1_sampled[b, t] in top5_indices[b, t]:
                    match_count += 1
        check('Test 5: Sampled tokens are strictly within top-5 set', match_count == B * T, f'{match_count}/{B*T} matches')
    except Exception as e:
        check('Test 5: top_k=5 sampling', False, str(e))

    print('\n' + '=' * 50)
    if failures:
        print(f'RESULT: RED ({len(failures)} failures)')
        for f, extra in failures:
            print(f'  - {f}: {extra}')
        return 1
    else:
        print('RESULT: GREEN (All tests passed)')
        return 0

if __name__ == '__main__':
    sys.exit(run_tests())
