"""Does a richer feature representation de-saturate PAMAP2 for channel selection?

Cheap, no-AE LOSO check. Per-channel features = [mean, std, min, max, mean|diff|]
(5/channel) with a RandomForest classifier. Compares SUPERVISED mutual-info,
variance, and random. If even supervised MI cannot separate from random with these
richer features, the dataset is fundamentally saturated and no feature change will
rescue a PAMAP2 headline -> keep current results, write abstract accordingly.

Run: python experiments/general_pamap2_richfeat_diag.py
"""
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_classif

from channel_select.adapters.pamap2_monster import load_pamap2_monster

DATA_DIR = Path("Data/Raw/PAMAP2_MONSTER")
SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]
KS = [5, 7, 10]


def rich_features(ds, idx):
    cols, owner, pairs = [], [], []
    for g in ds.groups:
        arr = ds.data[g].numpy()[idx]  # (n, time, ch)
        for c in range(arr.shape[-1]):
            x = arr[:, :, c]
            for feat in (x.mean(1), x.std(1), x.min(1), x.max(1),
                         np.abs(np.diff(x, axis=1)).mean(1)):
                cols.append(feat); owner.append((g, c))
            pairs.append((g, c))
    return np.stack(cols, axis=1), owner, pairs


def rf_f1(Xtr, ytr, Xte, yte, keep):
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    clf.fit(Xtr[:, keep], ytr)
    return f1_score(yte, clf.predict(Xte[:, keep]), average="macro")


def cols_for(owner, sel):
    return [i for i, o in enumerate(owner) if o in sel]


def main():
    ds = load_pamap2_monster(DATA_DIR)
    res = {K: {m: [] for m in ("mi", "var", "rand")} for K in KS}
    ceil = []
    for holdout in SUBJECTS:
        tr, te = ds.loso_split(holdout)
        Xtr, owner, pairs = rich_features(ds, tr)
        Xte, _, _ = rich_features(ds, te)
        ytr, yte = ds.labels[tr].numpy(), ds.labels[te].numpy()
        ceil.append(rf_f1(Xtr, ytr, Xte, yte, list(range(Xtr.shape[1]))))

        var_rank = sorted(pairs, key=lambda p: -float(Xtr[:, cols_for(owner, {p})].var(0).sum()))
        mi = mutual_info_classif(Xtr, ytr, random_state=0)
        mi_rank = sorted(pairs, key=lambda p: -max(mi[cols_for(owner, {p})]))
        for K in KS:
            res[K]["var"].append(rf_f1(Xtr, ytr, Xte, yte, cols_for(owner, set(var_rank[:K]))))
            res[K]["mi"].append(rf_f1(Xtr, ytr, Xte, yte, cols_for(owner, set(mi_rank[:K]))))
            rs = [rf_f1(Xtr, ytr, Xte, yte,
                        cols_for(owner, set(pairs[i] for i in
                                            np.random.default_rng(s).choice(len(pairs), K, False))))
                  for s in range(3)]
            res[K]["rand"].append(float(np.mean(rs)))
        print(f"subject {holdout} done")

    lines = [f"PAMAP2 LOSO rich-feature (mean/std/min/max/mad) + RandomForest | macro-F1 | "
             f"ceiling(27)={np.mean(ceil):.4f}+/-{np.std(ceil):.4f}"]
    for K in KS:
        for m in ("mi", "var", "rand"):
            a = np.array(res[K][m]); lines.append(f"K={K}\t{m:4s}\t{a.mean():.4f} +/- {a.std():.4f}")
    out = Path("generalization/reports/pamap2_richfeat_diag.txt")
    out.write_text("\n".join(lines)); print("\n".join(lines)); print(f"wrote {out}")


if __name__ == "__main__":
    main()
