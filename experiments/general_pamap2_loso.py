"""PAMAP2 full leave-one-subject-out robustness check.

Confirms the mid-K AE-perturb advantage over variance/random is not a single-fold
artifact. Retrains the AE on each fold's training subjects, runs ONE selection at
K=max and evaluates K prefixes (MMR is greedy, so selected[:k] == the K=k selection).

Run: PYTORCH_ENABLE_MPS_FALLBACK=1 python experiments/general_pamap2_loso.py
"""
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from channel_select.adapters.pamap2_monster import load_pamap2_monster
from channel_select.models.temporal import TemporalGroupedAutoencoder
from channel_select.models.training import train_autoencoder
from channel_select.engine import run_selection
from channel_select.protocols import SelectionConfig

DATA_DIR = Path("Data/Raw/PAMAP2_MONSTER")
SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]   # 9 has only 126 windows -> excluded as a test fold
KS = [5, 7, 10]
KMAX = max(KS)
SELECT_SUBSAMPLE = 4000
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def feats(ds, idx, selected):
    out = []
    for g in ds.groups:
        arr = ds.data[g].numpy()[idx]
        for c in range(arr.shape[-1]):
            if (g, c) in selected:
                out.append(arr[:, :, c].mean(axis=1))
                out.append(arr[:, :, c].std(axis=1))
    return np.stack(out, axis=1)


def knn_f1(ds, tr, te, selected):
    Xtr, Xte = feats(ds, tr, selected), feats(ds, te, selected)
    ytr, yte = ds.labels[tr].numpy(), ds.labels[te].numpy()
    sc = StandardScaler().fit(Xtr)
    clf = KNeighborsClassifier(n_neighbors=5).fit(sc.transform(Xtr), ytr)
    return f1_score(yte, clf.predict(sc.transform(Xte)), average="macro")


def variance_select(sel_data, K):
    scored = []
    for g, t in sel_data.items():
        arr = t.numpy()
        for c in range(arr.shape[-1]):
            scored.append(((g, c), float(np.var(arr[:, :, c]))))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [k for k, _ in scored[:K]]


def main():
    ds = load_pamap2_monster(DATA_DIR)
    flat = [(g, c) for g in ds.groups for c in range(ds.channels_per_group[g])]
    results = {K: {"ae": [], "var": [], "rand": []} for K in KS}

    for holdout in SUBJECTS:
        torch.manual_seed(0)
        tr, te = ds.loso_split(holdout)
        rng = np.random.default_rng(holdout)
        sub = rng.choice(tr, size=min(SELECT_SUBSAMPLE, len(tr)), replace=False)
        sel_data = {g: ds.data[g][sub] for g in ds.groups}

        time_len = ds.data[ds.groups[0]].shape[1]
        model = TemporalGroupedAutoencoder(ds.channels_per_group, time_len, latent_dim=16)
        train_autoencoder(model, sel_data, epochs=15, lr=1e-3, batch_size=128, device=DEVICE)
        model = model.to("cpu")

        cfg = SelectionConfig(
            n_important_dimensions=40, n_channels_to_select=KMAX,
            normalization_method="max_per_group", diversity_method="mmr",
            perturbation_method="standard_deviation", perturbation_magnitudes=[20, 40],
        )
        ae_order = run_selection(model, sel_data, cfg).selected   # greedy MMR order
        var_order = variance_select(sel_data, KMAX)

        for K in KS:
            results[K]["ae"].append(knn_f1(ds, tr, te, set(ae_order[:K])))
            results[K]["var"].append(knn_f1(ds, tr, te, set(var_order[:K])))
            rs = [knn_f1(ds, tr, te,
                         set(flat[i] for i in np.random.default_rng(s).choice(len(flat), K, False)))
                  for s in range(3)]
            results[K]["rand"].append(float(np.mean(rs)))
        print(f"subject {holdout} done")

    lines = ["PAMAP2 full LOSO (subjects 1-8) | KNN-5 macro-F1 | mean +/- std across folds"]
    for K in KS:
        for m in ("ae", "var", "rand"):
            a = np.array(results[K][m])
            lines.append(f"K={K}\t{m}\t{a.mean():.4f} +/- {a.std():.4f}")
    out = Path("generalization/reports/pamap2_loso.txt")
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
