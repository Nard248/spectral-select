"""PAMAP2 vertical slice: prove the channel_select engine transfers to HAR.

Uses the MONSTER-preprocessed PAMAP2 (ungated HF mirror). Trains the temporal
grouped autoencoder unsupervised, runs perturbation-based channel selection,
and evaluates accuracy-vs-K with LOSO KNN, plus a random-K control.

Run: PYTORCH_ENABLE_MPS_FALLBACK=1 python experiments/general_pamap2_slice.py
"""
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from channel_select.adapters.pamap2_monster import load_pamap2_monster
from channel_select.models.temporal import TemporalGroupedAutoencoder
from channel_select.models.training import train_autoencoder
from channel_select.engine import run_selection
from channel_select.protocols import SelectionConfig

DATA_DIR = Path("Data/Raw/PAMAP2_MONSTER")
HOLDOUT_SUBJECT = 5          # substantial subject for the LOSO test fold
SELECT_SUBSAMPLE = 4000      # windows used to train AE + run selection (speed)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0


def flatten_features(ds, indices, selected=None):
    """Per-window features = per-channel (mean, std) over time, restricted to
    selected (group, channel) pairs if given."""
    feats = []
    for g in ds.groups:
        arr = ds.data[g].numpy()[indices]   # (n, time, ch)
        for c in range(arr.shape[-1]):
            if selected is not None and (g, c) not in selected:
                continue
            feats.append(arr[:, :, c].mean(axis=1))
            feats.append(arr[:, :, c].std(axis=1))
    return np.stack(feats, axis=1)


def knn_loso(ds, selected, holdout):
    tr, te = ds.loso_split(holdout)
    Xtr = flatten_features(ds, tr, selected)
    Xte = flatten_features(ds, te, selected)
    ytr = ds.labels[tr].numpy(); yte = ds.labels[te].numpy()
    scaler = StandardScaler().fit(Xtr)
    clf = KNeighborsClassifier(n_neighbors=5).fit(scaler.transform(Xtr), ytr)
    pred = clf.predict(scaler.transform(Xte))
    return accuracy_score(yte, pred), f1_score(yte, pred, average="macro")


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    ds = load_pamap2_monster(DATA_DIR)
    total_ch = sum(ds.channels_per_group.values())
    print(f"Loaded {ds.n_windows} windows | groups={ds.groups} | "
          f"channels/group={ds.channels_per_group} | total={total_ch} | "
          f"classes={sorted(set(ds.labels.tolist()))}")

    # Train AE + run selection on a subsample drawn from the TRAIN subjects only.
    train_idx, _ = ds.loso_split(HOLDOUT_SUBJECT)
    sub = rng.choice(train_idx, size=min(SELECT_SUBSAMPLE, len(train_idx)), replace=False)
    sel_data = {g: ds.data[g][sub] for g in ds.groups}

    time_len = ds.data[ds.groups[0]].shape[1]
    model = TemporalGroupedAutoencoder(ds.channels_per_group, time_len=time_len, latent_dim=16)
    hist = train_autoencoder(model, sel_data, epochs=20, lr=1e-3, batch_size=128, device=DEVICE)
    print(f"AE train loss {hist[0]:.4f} -> {hist[-1]:.4f}")
    model = model.to("cpu")

    rows = []
    for K in [3, 5, 7, 10, 15]:
        cfg = SelectionConfig(
            n_important_dimensions=40, n_channels_to_select=K,
            normalization_method="max_per_group", diversity_method="mmr",
            perturbation_method="standard_deviation", perturbation_magnitudes=[20, 40],
        )
        res = run_selection(model, sel_data, cfg)
        acc, f1 = knn_loso(ds, set(res.selected), HOLDOUT_SUBJECT)
        # random-K control (mean over 5 seeds)
        flat = [(g, c) for g in ds.groups for c in range(ds.channels_per_group[g])]
        racc = []
        for s in range(5):
            r = np.random.default_rng(s)
            rsel = set(flat[i] for i in r.choice(len(flat), K, replace=False))
            racc.append(knn_loso(ds, rsel, HOLDOUT_SUBJECT)[1])  # macro-F1
        rmean = float(np.mean(racc))
        rows.append((K, acc, f1, rmean, res.selected))
        print(f"K={K:2d}  AE-perturb: acc={acc:.3f} F1={f1:.3f} | random F1={rmean:.3f} | "
              f"picks={res.selected}")

    out_dir = Path("generalization/reports"); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "pamap2_slice.txt"
    out.write_text(
        f"PAMAP2 MONSTER slice | holdout subject {HOLDOUT_SUBJECT} | LOSO KNN-5\n"
        + "\n".join(f"K={k}\tacc={a:.4f}\tmacroF1={f:.4f}\trandom_macroF1={rm:.4f}\t{s}"
                    for k, a, f, rm, s in rows)
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
