"""Diagnostic: is PAMAP2 (with mean+std features) discriminative for channel
selection at all? Runs LOSO for non-AE baselines only (fast, no training):
  - random-K (floor)
  - variance (unsupervised)
  - mutual-information (SUPERVISED upper-reference: uses labels)
  - full-27 ceiling
If even supervised MI cannot beat random, the dataset/feature is the bottleneck
and Opportunity becomes the discriminating test.

Run: python experiments/general_pamap2_baseline_diag.py
"""
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_classif

from channel_select.adapters.pamap2_monster import load_pamap2_monster

DATA_DIR = Path("Data/Raw/PAMAP2_MONSTER")
SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]
KS = [5, 7, 10]


def channel_features(ds, idx):
    """Returns (X[n, 2*nch], channel_of_column[2*nch], pairs[nch])."""
    cols, owner, pairs = [], [], []
    for g in ds.groups:
        arr = ds.data[g].numpy()[idx]
        for c in range(arr.shape[-1]):
            cols.append(arr[:, :, c].mean(axis=1)); owner.append((g, c))
            cols.append(arr[:, :, c].std(axis=1));  owner.append((g, c))
            pairs.append((g, c))
    return np.stack(cols, axis=1), owner, pairs


def knn_f1(Xtr, ytr, Xte, yte, keep_cols):
    sc = StandardScaler().fit(Xtr[:, keep_cols])
    clf = KNeighborsClassifier(n_neighbors=5).fit(sc.transform(Xtr[:, keep_cols]), ytr)
    return f1_score(yte, clf.predict(sc.transform(Xte[:, keep_cols])), average="macro")


def cols_for(owner, selected):
    return [i for i, o in enumerate(owner) if o in selected]


def main():
    ds = load_pamap2_monster(DATA_DIR)
    res = {K: {m: [] for m in ("rand", "var", "mi")} for K in KS}
    ceil = []

    for holdout in SUBJECTS:
        tr, te = ds.loso_split(holdout)
        Xtr, owner, pairs = channel_features(ds, tr)
        Xte, _, _ = channel_features(ds, te)
        ytr, yte = ds.labels[tr].numpy(), ds.labels[te].numpy()
        ceil.append(knn_f1(Xtr, ytr, Xte, yte, list(range(Xtr.shape[1]))))

        # variance per channel (sum of the two column variances)
        var_rank = sorted(pairs, key=lambda p: -float(
            Xtr[:, cols_for(owner, {p})].var(axis=0).sum()))
        # supervised MI per channel (max MI over its two columns)
        mi = mutual_info_classif(StandardScaler().fit_transform(Xtr), ytr, random_state=0)
        mi_by_ch = {p: max(mi[cols_for(owner, {p})]) for p in pairs}
        mi_rank = sorted(pairs, key=lambda p: -mi_by_ch[p])

        for K in KS:
            res[K]["var"].append(knn_f1(Xtr, ytr, Xte, yte, cols_for(owner, set(var_rank[:K]))))
            res[K]["mi"].append(knn_f1(Xtr, ytr, Xte, yte, cols_for(owner, set(mi_rank[:K]))))
            rs = []
            for s in range(5):
                rsel = set(pairs[i] for i in np.random.default_rng(s).choice(len(pairs), K, False))
                rs.append(knn_f1(Xtr, ytr, Xte, yte, cols_for(owner, rsel)))
            res[K]["rand"].append(float(np.mean(rs)))
        print(f"subject {holdout} done")

    lines = [f"PAMAP2 LOSO baseline diagnostic | KNN-5 macro-F1 | ceiling(27)="
             f"{np.mean(ceil):.4f}+/-{np.std(ceil):.4f}"]
    for K in KS:
        for m in ("mi", "var", "rand"):
            a = np.array(res[K][m]); lines.append(f"K={K}\t{m:4s}\t{a.mean():.4f} +/- {a.std():.4f}")
    out = Path("generalization/reports/pamap2_baseline_diag.txt")
    out.write_text("\n".join(lines)); print("\n".join(lines)); print(f"wrote {out}")


if __name__ == "__main__":
    main()
