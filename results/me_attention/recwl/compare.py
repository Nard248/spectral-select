"""night (shifted wavelength labels) vs recwl (recorded wavelengths) for the wavelength-aware models; writes JSON for the audit doc."""
import json, sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]
DS = [("collagen_pepsin", "Collagen Pepsin"), ("lichens1", "Lichens 1"), ("lichens2", "Lichens 2")]
ATT = [("ex_tokens_v2", "ex tokens v2"), ("meas_tokens", "ME-Attn (meas.)"), ("smartattention_v0", "SmartAttn v0"),
       ("hybrid_conv_attn", "hybrid conv attn"), ("spatial_attn", "ME-Attn (spatial)")]
BASE = ["lr", "rbfsvm", "rf", "hgb", "knn", "mlp", "cnn1d", "spectralformer"]
def recs(sweep, ds, m):
    p = R / sweep / ds / f"{m}.jsonl"
    out = {}
    if p.exists():
        for l in p.read_text().splitlines():
            try: r = json.loads(l)
            except Exception: continue
            if "error" in r: continue
            out[(r["repeat"], r["fold"])] = 100 * r["test"]["balanced_acc"]
    return out
rows = []
for m, lab in ATT:
    row = [lab]
    for ds, _ in DS:
        a, b = recs("night", ds, m), recs("recwl", ds, m)
        k = sorted(set(a) & set(b))
        if not k: row.append("--"); continue
        va, vb = np.array([a[x] for x in k]), np.array([b[x] for x in k])
        row.append(f"{va.mean():.1f} -> {vb.mean():.1f} ({vb.mean()-va.mean():+.1f}; n={len(k)})")
    rows.append(row)
best = ["Best spectral baseline (unchanged)"]
for ds, _ in DS:
    ms = {m: recs("night", ds, m) for m in BASE}
    mm = {m: np.mean(list(v.values())) for m, v in ms.items() if v}
    b = max(mm, key=mm.get); best.append(f"{b}: {mm[b]:.1f}")
rows.append(best)
out = dict(head=["Model (balanced acc., %)"] + [d for _, d in DS], rows=rows, widths=[2426, 2200, 2200, 2200],
           notes=["Format: 19 Oct sweep (shifted labels) -> rerun with recorded wavelengths (difference; matched repeat/fold units). Spatial attention uses 6/4/4 units, spectral models 15/4/10."])
json.dump(out, open(sys.argv[1], "w"), indent=1)
for r in rows: print(" | ".join(r))
