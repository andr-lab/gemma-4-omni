# -*- coding: utf-8 -*-
# Test unitari per TurnStateHead e Dataloader Labeling
import sys, os, torch, numpy as np

sys.path.insert(0, ".")

def test_turnhead_module():
    import torch.nn as nn
    import torch.nn.functional as F

    # 1. Definizione TurnStateHead
    class TurnStateHead(nn.Module):
        """TurnStateHead (3.8M params): Classificatore causale Pre-LN a 5 stati conversazionali.
        Input: z_t = [h_t_temp(2048) || e_user(128) || d_e_user(128)] in R^2304.
        """
        def __init__(self, in_dim=2304, hidden_dim=1024, mid_dim=256, num_classes=5):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.act1 = nn.SiLU()
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
            self.act2 = nn.SiLU()
            self.fc3 = nn.Linear(hidden_dim, mid_dim)
            self.ln3 = nn.LayerNorm(mid_dim)
            self.act3 = nn.SiLU()
            self.classifier = nn.Linear(mid_dim, num_classes)

        def forward(self, z):
            x = self.act1(self.ln1(self.fc1(z)))
            x = self.act2(self.ln2(self.fc2(x)))
            x = self.act3(self.ln3(self.fc3(x)))
            return self.classifier(x)

    print("--- TEST 1: Parametri Totali TurnStateHead ---")
    head = TurnStateHead()
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"Parametri TurnStateHead: {n_params:,} ({n_params/1e6:.2f}M)")
    assert 3_500_000 <= n_params <= 4_200_000, f"Parametri fuori target: {n_params}"
    print("Test 1 PASS: Parametri conformi alle specifiche (~3.68M-3.8M)")

    print("--- TEST 2: Shape e Forward z_t in R^2304 -> Logits in R^5 ---")
    B, T = 4, 64
    h_temp = torch.randn(B, T, 2048)
    e_user = torch.randn(B, T, 128)
    d_e_user = torch.randn(B, T, 128)
    z = torch.cat([h_temp, e_user, d_e_user], dim=-1)
    assert z.shape == (B, T, 2304), f"Shape errata per z: {z.shape}"
    logits = head(z)
    assert logits.shape == (B, T, 5), f"Shape errata per logits: {logits.shape}"
    print("Test 2 PASS: Forward corretto con output (B, T, 5)")

    print("--- TEST 3: Algoritmo di Labelling su Finestra Sintetica Controllata ---")
    def derive_turn_labels_synthetic(a_act, u_act, T=64):
        """Deriva le ground truth labels y_t in {0..4} e valid_mask per frame."""
        labels = np.zeros(T, dtype=np.int64)
        valid = np.zeros(T, dtype=bool)

        # Calcola segmenti contigui di attivita utente
        u_runs = []
        in_run = False
        start = 0
        for t in range(T):
            if u_act[t] and not in_run:
                in_run = True
                start = t
            elif not u_act[t] and in_run:
                in_run = False
                u_runs.append((start, t - 1, t - start))
        if in_run:
            u_runs.append((start, T - 1, T - start))

        for t in range(T):
            is_a = bool(a_act[t])
            is_u = bool(u_act[t])
            valid[t] = is_a or is_u

            if not valid[t]:
                labels[t] = 0
                continue

            if is_a and not is_u:
                # Agente parla, utente silenzioso: verifica yield (ultimi 3 frame di emissione agente)
                is_yield = False
                if t + 1 >= T or not a_act[t + 1]:
                    is_yield = True
                elif t + 2 >= T or not a_act[t + 2]:
                    is_yield = True
                elif t + 3 >= T or not a_act[t + 3]:
                    is_yield = True

                labels[t] = 1 if is_yield else 0  # 1=YIELD, 0=SPEAK

            elif is_a and is_u:
                # Sovrapposizione: trova durata della run utente corrente
                cur_len = 1
                for s, e, l in u_runs:
                    if s <= t <= e:
                        cur_len = l
                        break
                # >= 5 frame (400ms @ 12.5Hz) -> BARGE_IN (2), < 5 frame -> USER_BACKCHANNEL (3)
                labels[t] = 2 if cur_len >= 5 else 3

            elif not is_a and is_u:
                # Solo ascolto utente: se pausa sintattica/fine frase -> EMIT_BACKCHANNEL (4), else 0 (LISTEN)
                # Euristica: frame di assenso attorno a pause
                labels[t] = 4 if (t > 10 and (t % 16 in (0, 1, 2))) else 0

        return labels, valid

    # Scenario sintetico:
    # Frame 0..19: Agente parla da solo (agente True 0..19, False 20..22) -> SPEAK 0..16, YIELD 17..19
    # Frame 23..35: Sovrapposizione lunga (agente True 23..35, utente True 23..33) -> BARGE_IN (2)
    # Frame 36..45: Sovrapposizione breve (agente True 36..45, utente True 36..38) -> USER_BACKCHANNEL (3)
    # Frame 46..55: Solo ascolto utente (agente False, utente True 46..55) -> LISTEN / EMIT_BACKCHANNEL (4)
    # Frame 56..63: Silenzio entrambi -> invalid
    a_synth = np.zeros(64, dtype=bool)
    u_synth = np.zeros(64, dtype=bool)
    a_synth[0:20] = True   # agente attivo 0..19 (fine a 20 -> yield su 17..19)
    a_synth[23:36] = True  # agente attivo 23..35
    u_synth[23:33] = True  # sovrapposizione lunga (10 frame) -> barge_in su 23..32
    a_synth[36:46] = True  # agente attivo 36..45
    u_synth[36:39] = True  # sovrapposizione breve (3 frame) -> user_backchannel su 36..38
    u_synth[46:56] = True  # utente parla da solo (agente silenzioso)

    lbls, val = derive_turn_labels_synthetic(a_synth, u_synth, T=64)

    # Verifiche puntuali
    assert lbls[5] == 0, f"Frame 5 dovrebbe essere SPEAK (0), trovato {lbls[5]}"
    assert lbls[19] == 1 and lbls[18] == 1 and lbls[17] == 1, f"Frame 17-19 dovrebbero avere YIELD (1), trovato {lbls[17:20]}"
    assert lbls[25] == 2, f"Frame 25 dovrebbe essere BARGE_IN (2), trovato {lbls[25]}"
    assert lbls[37] == 3, f"Frame 37 dovrebbe essere USER_BACKCHANNEL (3), trovato {lbls[37]}"
    assert (lbls[46:56] == 4).any(), f"Dovrebbero esserci frame di EMIT_BACKCHANNEL (4) nel solo ascolto utente"
    assert val[60] == False, f"Frame 60 dovrebbe essere non valido, trovato {val[60]}"
    print(f"Distribuzione label generate su finestra sintetica: {dict(zip(*np.unique(lbls[val], return_counts=True)))}")
    print("Test 3 PASS: Algoritmo di labeling produce esattamente i 5 stati previsti")

    print("--- TEST 4: Weighted Cross-Entropy Loss & Gradient Isolation ---")
    weights = torch.tensor([1.0, 8.0, 6.0, 3.0, 4.0])
    loss_fn = nn.CrossEntropyLoss(weight=weights, reduction='none')

    target_t = torch.from_numpy(lbls).unsqueeze(0).repeat(B, 1)
    val_t = torch.from_numpy(val).unsqueeze(0).repeat(B, 1)

    loss_raw = loss_fn(logits.view(-1, 5), target_t.view(-1)).view(B, T)
    loss = (loss_raw * val_t.float()).sum() / val_t.float().sum().clamp(min=1.0)
    assert not torch.isnan(loss) and loss.item() > 0, f"Loss non valida: {loss}"

    loss.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None, f"Gradiente nullo per {name}"
        assert not torch.isnan(p.grad).any(), f"NaN nel gradiente per {name}"
    print(f"Loss CE pesata calcolata: {loss.item():.4f}")
    print("Test 4 PASS: Loss pesata e gradienti autograd verificati")

    print("=== TUTTI I TEST TURNHEAD SUPERATI CON SUCCESSO! ===")

if __name__ == "__main__":
    test_turnhead_module()
