# -*- coding: utf-8 -*-
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
    windows = []
    for idx in range(n_windows):
        labels = np.zeros(T, dtype=np.int64)
        valid = np.ones(T, dtype=bool)
        wtype_cat = np.random.choice(['monologue', 'listen', 'overlap', 'rare_bc'], p=[0.45, 0.25, 0.20, 0.10])
        
        if wtype_cat == 'monologue':
            labels[:] = 0
            labels[-3:] = 1
        elif wtype_cat == 'listen':
            labels[:] = 0
            labels[10:20] = 4
            labels[35:45] = 4
        elif wtype_cat == 'overlap':
            labels[:30] = 0
            labels[30:50] = 2
            labels[50:] = 3
        elif wtype_cat == 'rare_bc':
            labels[:20] = 0
            labels[20:45] = 3
            labels[45:] = 2

        e_user = np.random.randn(T, 128).astype(np.float32)
        d_e_user = np.zeros_like(e_user)
        d_e_user[1:] = e_user[1:] - e_user[:-1]

        windows.append({
            'idx': idx,
            'labels': labels,
            'valid': valid,
            'e_user': e_user,
            'd_e_user': d_e_user,
            'wtype': wtype_cat,
            'n_valid': int(valid.sum())
        })
    return windows

class TurnHeadV3CalibratedDataset:
    def __init__(self, windows, n_classes=5):
        self.windows = windows
        self.n_classes = n_classes
        
        self.class_counts = np.zeros(n_classes, dtype=np.int64)
        self.bucket_mono = []
        self.bucket_listen = []
        self.bucket_overlap = []
        self.bucket_bc = []

        for w in windows:
            lbls = w['labels'][w['valid']]
            for c in range(n_classes):
                self.class_counts[c] += int((lbls == c).sum())
            
            c_counts = {c: int((lbls == c).sum()) for c in range(n_classes)}
            if c_counts[3] >= 4:
                self.bucket_bc.append(w)
            elif c_counts[2] >= 4 or (c_counts[2] + c_counts[3] >= 6):
                self.bucket_overlap.append(w)
            elif c_counts[4] >= 4:
                self.bucket_listen.append(w)
            else:
                self.bucket_mono.append(w)

        all_w = self.bucket_mono + self.bucket_listen + self.bucket_overlap + self.bucket_bc
        if not self.bucket_mono: self.bucket_mono = all_w[:1]
        if not self.bucket_listen: self.bucket_listen = all_w[:1]
        if not self.bucket_overlap: self.bucket_overlap = all_w[:1]
        if not self.bucket_bc: self.bucket_bc = all_w[:1]

        self.class_dist_train = {c: int(self.class_counts[c]) for c in range(n_classes)}

    def sample_calibrated_batch(self, batch_size=8):
        n_mono = max(1, int(round(batch_size * 0.375)))
        n_overlap = max(1, int(round(batch_size * 0.25)))
        n_bc = max(1, int(round(batch_size * 0.25)))
        n_listen = max(1, batch_size - n_mono - n_overlap - n_bc)

        b = []
        for _ in range(n_mono):
            b.append(self.bucket_mono[np.random.randint(len(self.bucket_mono))])
        for _ in range(n_overlap):
            b.append(self.bucket_overlap[np.random.randint(len(self.bucket_overlap))])
        for _ in range(n_bc):
            b.append(self.bucket_bc[np.random.randint(len(self.bucket_bc))])
        for _ in range(n_listen):
            b.append(self.bucket_listen[np.random.randint(len(self.bucket_listen))])

        batch_dist = {c: 0 for c in range(self.n_classes)}
        for w in b:
            lbls = w['labels'][w['valid']]
            for c in range(self.n_classes):
                batch_dist[c] += int((lbls == c).sum())

        return b, batch_dist

def compute_calibrated_loss_torch(logits, targets, valids, class_weights_t, gamma=1.0, speak_boost=1.2, smoothing=0.05):
    n_classes = logits.shape[-1]
    flat_logits = logits.view(-1, n_classes)
    flat_targets = targets.view(-1)
    flat_valids = valids.view(-1)

    mask = flat_valids & (flat_targets >= 0) & (flat_targets < n_classes)
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    v_logits = flat_logits[mask]
    v_targets = flat_targets[mask]

    probs = F.softmax(v_logits, dim=-1)
    p_t = probs.gather(1, v_targets.unsqueeze(1)).squeeze(1).clamp(min=1e-7, max=1.0 - 1e-7)

    ce_raw = F.cross_entropy(v_logits, v_targets, label_smoothing=smoothing, reduction='none')
    focal_factor = (1.0 - p_t) ** gamma

    w_t = class_weights_t[v_targets]
    speak_mask = (v_targets == 0)
    w_t = torch.where(speak_mask, w_t * speak_boost, w_t)

    focal_ce = w_t * focal_factor * ce_raw

    class_losses = []
    active_weights = []
    for c in range(n_classes):
        c_idx = (v_targets == c)
        if c_idx.any():
            c_mean = focal_ce[c_idx].mean()
            w_c = class_weights_t[c] * (speak_boost if c == 0 else 1.0)
            class_losses.append(w_c * c_mean)
            active_weights.append(w_c)

    if not class_losses:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    return sum(class_losses) / sum(active_weights)

class TestTurnHeadV3Balance(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.windows = simulate_window_labels(n_windows=400, T=64)
        self.dataset = TurnHeadV3CalibratedDataset(self.windows, n_classes=5)

    def test_calibrated_batch_sampling_speak_presence(self):
        total_counts = {c: 0 for c in range(5)}
        for _ in range(50):
            batch, b_dist = self.dataset.sample_calibrated_batch(batch_size=8)
            self.assertEqual(len(batch), 8)
            for c in range(5):
                total_counts[c] += b_dist[c]
                self.assertGreater(b_dist[c], 0, f'Classe {c} deve essere presente nel batch (got 0)')

        tot_frames = sum(total_counts.values())
        speak_pct = 100.0 * total_counts[0] / tot_frames
        print(f'\n[TEST v3] Distribuzione Frame Batch Calibrato (50 batch):')
        for c in range(5):
            print(f'  Classe {c}: {total_counts[c]} frame ({100.0 * total_counts[c]/tot_frames:.2f}%)')

        self.assertGreaterEqual(speak_pct, 20.0, f'SPEAK deve essere >= 20% dei frame nel batch (got {speak_pct:.1f}%)')

    def test_calibrated_loss_gradient_flow(self):
        if not HAS_TORCH:
            self.skipTest('PyTorch non disponibile localmente')

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
        dummy_z = torch.randn(8, 64, 2304, requires_grad=True)
        logits = model(dummy_z)

        labels = torch.randint(0, 5, (8, 64), dtype=torch.long)
        for c in range(5):
            labels[c, :10] = c
        valids = torch.ones((8, 64), dtype=torch.bool)

        weights = torch.tensor([1.2, 1.4, 1.8, 2.2, 1.5], dtype=torch.float32)
        loss = compute_calibrated_loss_torch(logits, labels, valids, weights, gamma=1.0, speak_boost=1.2)

        loss.backward()
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, f'Gradiente per {name} e None')
            self.assertGreater(p.grad.abs().sum().item(), 0.0, f'Gradiente nullo per {name}')

if __name__ == '__main__':
    unittest.main()
