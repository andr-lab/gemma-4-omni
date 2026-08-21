# -*- coding: utf-8 -*-
"""
test_turnhead_v2_balance.py — Test unitario per il campionatore bilanciato a livello di frame (TurnHead v2).
Verifica:
  1. Calcolo corretto della distribuzione frame globale (TURN_CLASS_DIST_TRAIN).
  2. Pesi di campionamento delle finestre e partizionamento a 4 bucket.
  3. Costruzione di un validation set bilanciato con frame-cap per classe (supporto <= 4x).
  4. Calcolo della Loss Bilanciata per Classe (Mean-per-Class Loss).
  5. Presenza garantita di tutte le 5 classi nei batch.
"""
import unittest
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def simulate_window_labels(n_windows=300, T=64):
    """Simula una serie di finestre con distribuzione realistica."""
    windows = []
    for idx in range(n_windows):
        labels = np.zeros(T, dtype=np.int64)
        valid = np.ones(T, dtype=bool)
        wtype_cat = np.random.choice(["monologue", "listen", "overlap", "rare_bc"], p=[0.45, 0.25, 0.20, 0.10])
        
        if wtype_cat == "monologue":
            labels[:] = 0
            labels[-3:] = 1  # YIELD
        elif wtype_cat == "listen":
            labels[:] = 0
            labels[10:20] = 4  # EMIT_BC
            labels[35:45] = 4
        elif wtype_cat == "overlap":
            labels[:30] = 0
            labels[30:50] = 2  # BARGE_IN
            labels[50:] = 3   # USER_BC
        elif wtype_cat == "rare_bc":
            labels[:20] = 0
            labels[20:45] = 3  # USER_BC
            labels[45:] = 2   # BARGE_IN

        e_user = np.random.randn(T, 128).astype(np.float32)
        d_e_user = np.zeros_like(e_user)
        d_e_user[1:] = e_user[1:] - e_user[:-1]

        windows.append({
            "idx": idx,
            "labels": labels,
            "valid": valid,
            "e_user": e_user,
            "d_e_user": d_e_user,
            "wtype": wtype_cat,
            "n_valid": int(valid.sum())
        })
    return windows

class TurnHeadV2BalancedDataset:
    """Gestione dati bilanciata a livello di frame e finestra per TurnHead v2."""
    def __init__(self, windows, n_classes=5):
        self.windows = windows
        self.n_classes = n_classes
        
        # 1. Calcolo distribuzione globale frame
        self.class_counts = np.zeros(n_classes, dtype=np.int64)
        self.bucket_mono = []
        self.bucket_listen = []
        self.bucket_overlap = []
        self.bucket_bc = []

        for w in windows:
            lbls = w["labels"][w["valid"]]
            for c in range(n_classes):
                self.class_counts[c] += int((lbls == c).sum())
            
            c_counts = {c: int((lbls == c).sum()) for c in range(n_classes)}
            if c_counts[3] > 5:
                self.bucket_bc.append(w)
            elif c_counts[2] > 5 or (c_counts[2] + c_counts[3] > 5):
                self.bucket_overlap.append(w)
            elif c_counts[4] > 5:
                self.bucket_listen.append(w)
            else:
                self.bucket_mono.append(w)

        # Fallback non vuoti
        all_w = self.bucket_mono + self.bucket_listen + self.bucket_overlap + self.bucket_bc
        if not self.bucket_mono: self.bucket_mono = all_w[:1]
        if not self.bucket_listen: self.bucket_listen = all_w[:1]
        if not self.bucket_overlap: self.bucket_overlap = all_w[:1]
        if not self.bucket_bc: self.bucket_bc = all_w[:1]

        self.class_dist_train = {c: int(self.class_counts[c]) for c in range(n_classes)}

    def sample_batch(self, batch_size=8):
        """Campiona un batch distribuito tra i 4 bucket per garantire tutte le 5 classi."""
        n_each = max(1, batch_size // 4)
        rem = batch_size - 4 * n_each
        
        b = []
        for _ in range(n_each + (1 if rem > 0 else 0)):
            b.append(self.bucket_mono[np.random.randint(len(self.bucket_mono))])
        for _ in range(n_each + (1 if rem > 1 else 0)):
            b.append(self.bucket_listen[np.random.randint(len(self.bucket_listen))])
        for _ in range(n_each + (1 if rem > 2 else 0)):
            b.append(self.bucket_overlap[np.random.randint(len(self.bucket_overlap))])
        for _ in range(n_each):
            b.append(self.bucket_bc[np.random.randint(len(self.bucket_bc))])

        batch_dist = {c: 0 for c in range(self.n_classes)}
        for w in b:
            lbls = w["labels"][w["valid"]]
            for c in range(self.n_classes):
                batch_dist[c] += int((lbls == c).sum())

        return b, batch_dist

    def build_balanced_val_set(self, n_windows=100, max_frames_per_class=500):
        """Costruisce un validation set bilanciato dove ogni classe ha supporto comparabile."""
        n_per_bucket = n_windows // 4
        val_set = []
        for i in range(n_per_bucket):
            val_set.append(self.bucket_mono[i % len(self.bucket_mono)])
            val_set.append(self.bucket_listen[i % len(self.bucket_listen)])
            val_set.append(self.bucket_overlap[i % len(self.bucket_overlap)])
            val_set.append(self.bucket_bc[i % len(self.bucket_bc)])

        # Calcola supporto raw e supporto controllato
        raw_val_support = {c: 0 for c in range(self.n_classes)}
        for w in val_set:
            lbls = w["labels"][w["valid"]]
            for c in range(self.n_classes):
                raw_val_support[c] += int((lbls == c).sum())

        capped_val_support = {c: min(max_frames_per_class, raw_val_support[c]) for c in range(self.n_classes)}
        return val_set, raw_val_support, capped_val_support

def compute_mean_per_class_loss_numpy(logits, labels, valid, weights=None, n_classes=5):
    """Calcola la Mean-per-Class Cross-Entropy in numpy."""
    exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    
    flat_probs = probs.reshape(-1, n_classes)
    flat_labels = labels.reshape(-1)
    flat_valid = valid.reshape(-1)

    if weights is None:
        weights = [1.0] * n_classes

    class_losses = []
    class_weights_active = []

    for c in range(n_classes):
        mask_c = (flat_labels == c) & flat_valid
        cnt = int(mask_c.sum())
        if cnt > 0:
            p_c = flat_probs[mask_c, c]
            nll_c = -np.mean(np.log(np.maximum(1e-12, p_c)))
            class_losses.append(weights[c] * nll_c)
            class_weights_active.append(weights[c])

    if not class_losses:
        return 0.0
    return float(sum(class_losses) / sum(class_weights_active))

class TestTurnHeadV2Balance(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.windows = simulate_window_labels(n_windows=400, T=64)
        self.dataset = TurnHeadV2BalancedDataset(self.windows, n_classes=5)

    def test_global_distribution(self):
        dist = self.dataset.class_dist_train
        self.assertEqual(len(dist), 5)
        for c in range(5):
            self.assertGreater(dist[c], 0, f"Classe {c} deve avere frame nel dataset")

    def test_batch_sampling_coverage(self):
        total_batch_counts = {c: 0 for c in range(5)}
        for _ in range(30):
            batch, batch_dist = self.dataset.sample_batch(batch_size=16)
            self.assertEqual(len(batch), 16)
            for c in range(5):
                total_batch_counts[c] += batch_dist[c]

        # Nessuna classe deve avere 0 frame nei batch cumulati
        for c in range(5):
            self.assertGreater(total_batch_counts[c], 100, f"Classe {c} deve avere supporto consistente (>100)")

    def test_balanced_val_set_ratio(self):
        val_set, raw_sup, capped_sup = self.dataset.build_balanced_val_set(n_windows=100, max_frames_per_class=500)
        self.assertEqual(len(val_set), 100)
        
        max_sup = max(capped_sup.values())
        min_sup = min(capped_sup.values())
        ratio = max_sup / max(1, min_sup)

        print("\n[TEST] Supporto Raw per classe nel Val Set:", raw_sup)
        print("[TEST] Supporto Capped per classe nel Val Set:", capped_sup)
        print(f"[TEST] Ratio max/min nel val set controllato: {ratio:.2f}x")

        # Verifica che il supporto controllato abbia ratio <= 4.0x
        self.assertLessEqual(ratio, 4.0, f"Il supporto controllato deve avere ratio <= 4.0x (ottenuto: {ratio:.2f}x)")

    def test_mean_per_class_loss_calculation(self):
        logits = np.random.randn(8, 64, 5)
        labels = np.random.randint(0, 5, (8, 64))
        valid = np.ones((8, 64), dtype=bool)
        loss = compute_mean_per_class_loss_numpy(logits, labels, valid, weights=[1.0, 1.5, 2.0, 2.0, 1.5], n_classes=5)
        self.assertGreater(loss, 0.0)
        self.assertFalse(np.isnan(loss))

    def test_turn_state_head_forward(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch non disponibile localmente (test eseguito in Fase 2)")

        class TurnStateHead(nn.Module):
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

        model = TurnStateHead(in_dim=2304, num_classes=5)
        dummy_z = torch.randn(4, 64, 2304)
        logits = model(dummy_z)
        self.assertEqual(logits.shape, (4, 64, 5))

if __name__ == "__main__":
    unittest.main()
