"""E9 step 2: unsupervised clustering benchmark on the E9 synthetic dataset (see E9_make_dataset.py).

Task: label-free segmentation of 3 buried tissue types (NADH-rich / FAD-rich / stroma) under a collagen cover wedge;
clustering uses K = 3 on the tissue pixels only (background excluded).  Methods:
  kmeans_pca    z-scored channels -> PCA(10) -> k-means (k-means++, 10 inits)
  gmm_pca       same features -> full-covariance GMM (3 components, 5 inits)
  kmeans_l2     as kmeans_pca on L2-normalised spectra (shape only, removes overall brightness)
  cae_kmeans    the repo's own research line: spectral_select HyperspectralCAEWithMasking (per-excitation Conv3d encoder,
                shared latent, masked loss, relu hidden activation), trained per capture for EPOCHS epochs
                (train_with_masking), latent (k3 = 20 dims) -> k-means
Channel sets (7 excitations 310/325/340/365/400/415/430 nm x 31 emission bands = 217 channels, dataset parts A+B with the same
design and seeds; 'all'):
  all; e5_thick15 / e5_thickamp26 / e5_jointD5: the information-optimal channel sets of E5 (read from results/E5/results.json,
  selection.<thick|thick+amp|joint_D>.0.acquisition.labels: sequential greedy for cover thickness / thickness+amplitude / a joint
  D-optimal design, photon-limited eps = 0); uniformK (K evenly spaced channels in the (excitation, emission) ordering) and
  randomK (mean over R random sets) with the same K as the E5 set; uniform31.
Metrics: ARI / NMI vs the tissue-type label, per capture and per 0.25 mm bin of the local analytic cover thickness
(pixels of the bin only, clusters fitted on the whole capture), mean +- SE over captures; also NMI between the clusters and
the thickness bin (does the clustering encode burial rather than tissue type?).
    .venv/bin/python publications/spectralforge_studio/experiments/E9_clustering.py [--quick] [--agg]
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from C3_common import RES  # noqa: E402
from spectraforge.studio.dataset import load_index, load_sample  # noqa: E402
from spectraforge.studio.provenance import provenance  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

DS = Path.home() / ".cache" / "spectralforge" / "e9_dataset"
DS_B = Path.home() / ".cache" / "spectralforge" / "e9_dataset_b"
BIN_EDGES = np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 1.5])
K = 3
EPOCHS = 30
QUICK = "--quick" in sys.argv
if QUICK:
    DS = Path.home() / ".cache" / "spectralforge" / "e9_dataset_quick"
    DS_B = None
    EPOCHS = 3
OUT = RES / ("E9_quick" if QUICK else "E9")


def channels(sd):
    """Ordered list of (excitation_nm, band index, emission_nm) and the stacked (H, W, C) cube."""
    ch, cubes = [], []
    for ex in sd.excitation_wavelengths:
        e = sd.get_excitation(ex)
        for b, wl in enumerate(e.emission_wavelengths):
            ch.append((float(ex), b, float(wl)))
        cubes.append(np.asarray(e.cube, float))
    return ch, np.concatenate(cubes, axis=2)


E5_KEYS = {"e5_thick15": "thick", "e5_thickamp26": "thick+amp", "e5_jointD5": "joint_D"}


def channel_sets(ch, rng):
    n = len(ch)
    sets = {"all": list(range(n))}
    sets["uniform31"] = [int(round(x)) for x in np.linspace(0, n - 1, 31)]
    lab = {f"{int(e)}->{int(w)}": i for i, (e, _, w) in enumerate(ch)}
    try:
        r = json.loads((RES / "E5" / "results.json").read_text())
        for name, key in E5_KEYS.items():
            labels = r["selection"][key]["0"]["acquisition"]["labels"]
            sets[name] = sorted(lab[l] for l in labels)
            k = len(labels)
            sets[f"uniform{k}"] = [int(round(x)) for x in np.linspace(0, n - 1, k)]
    except Exception as exc:  # noqa: BLE001 - E5 channels unavailable or not in this dataset: documented fallback
        print("E5 channels unavailable -> fallback: uniform subsets only:", repr(exc))
        for k in (5, 15, 26):
            sets[f"uniform{k}"] = [int(round(x)) for x in np.linspace(0, n - 1, k)]
    return sets


def smooth(cube, mask, s):
    """Masked s x s spatial mean (normalised convolution inside the tissue mask): MC/sensor noise is pixel-independent."""
    if s <= 1:
        return cube
    from scipy.ndimage import uniform_filter
    mf = mask.astype(float)
    den = np.maximum(uniform_filter(mf, s, mode="constant"), 1e-9)[..., None]
    return np.stack([uniform_filter(cube[:, :, i] * mf, s, mode="constant") for i in range(cube.shape[2])], -1) / den


def features(cube, idx, mask, l2=False):
    X = cube[:, :, idx][mask]
    if l2:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
    X = (X - X.mean(0)) / (X.std(0) + 1e-30)
    nc = min(10, X.shape[1])
    return PCA(nc, random_state=0).fit_transform(X)


def features_full(cube, idx, mask):
    X = cube[:, :, idx][mask]
    return (X - X.mean(0)) / (X.std(0) + 1e-30)


def km(Z, seed):
    return KMeans(K, n_init=10, random_state=seed).fit_predict(Z)


def cae_latent(sd, sel, mask, seed):
    """Train spectral_select's CAE on the (channel-reduced) capture and return the latent (npix, k3)."""
    import torch
    from spectral_select.models.autoencoder import HyperspectralCAEWithMasking
    from spectral_select.models.dataset import MaskedHyperspectralDataset
    from spectral_select.models.training import train_with_masking
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = {"data": {}, "excitation_wavelengths": sd.excitation_wavelengths}
    for ex in sd.excitation_wavelengths:
        e = sd.get_excitation(ex)
        keep = [b for (x, b, _) in sel if x == ex]
        if not keep:
            data["excitation_wavelengths"] = [x for x in data["excitation_wavelengths"] if x != ex]
            continue
        data["data"][str(ex)] = {"cube": np.asarray(e.cube)[:, :, keep], "wavelengths": [e.emission_wavelengths[b] for b in keep]}
    ds = MaskedHyperspectralDataset(data_dict=data, mask=mask.astype(float), normalize=True)
    exd = {ex: v for ex, v in ds.get_all_data().items()}
    model = HyperspectralCAEWithMasking(exd, k1=20, k3=20, filter_size=5, sparsity_target=0.1, sparsity_weight=1.0,
                                        dropout_rate=0.2, hidden_activation="relu", output_activation="sigmoid")
    # Workaround (src/ is not edited): enc_conv1 uses kernel (5, 5, min(5, n_bands)) on [bands, H, W] with padding k//2, which
    # breaks (shape mismatch between excitations) when an excitation has an EVEN number (< 5) of bands.  Use odd kernels there.
    fs = 5
    for ex in model.excitation_wavelengths:
        nb = model.emission_bands[ex]
        kk = min(5, nb)
        if kk % 2 == 0:
            kk -= 1
            model.enc_conv1[model.ex_to_key[ex]] = torch.nn.Conv3d(1, model.k1, (fs, fs, kk), padding=(fs // 2, fs // 2, kk // 2))
    model, losses = train_with_masking(model, ds, num_epochs=EPOCHS, learning_rate=1e-3, chunk_size=64, chunk_overlap=8,
                                       device="cpu", mask=ds.processed_mask, output_dir=str(OUT / "_cae_tmp"), verbose=False)
    model.train(False)
    with torch.no_grad():
        batch = {ex: v.unsqueeze(0) for ex, v in ds.get_all_data().items()}
        z = model.encode(batch)
    z = z.squeeze().cpu().numpy()          # (k3, H, W)
    z = np.moveaxis(z, 0, -1)
    return z[mask], float(losses[-1])


def lda_scores(Z, true, thick, seed=0):
    """Supervised reference (NOT label-free): 4-fold CV accuracy of shrinkage-LDA per thickness bin."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    bins = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        m = (thick >= lo) & (thick < hi) if hi < 1.5 else (thick >= lo)
        cnt = np.bincount(true[m], minlength=4)[1:]
        acc = None
        if m.sum() >= 15 and cnt.min() >= 4:
            acc = float(cross_val_score(LDA(solver="lsqr", shrinkage="auto"), Z[m], true[m],
                                        cv=StratifiedKFold(4, shuffle=True, random_state=seed)).mean())
        bins.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()), ARI=None, NMI=None, acc=acc))
    return dict(ARI_all=None, NMI_all=None, NMI_cluster_vs_thickbin=None, bins=bins)


def binfit_scores(Z, true, thick, seed=0):
    """Clustering fitted separately inside each thickness bin (removes the burial gradient as a confound)."""
    bins, allp = [], []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        m = (thick >= lo) & (thick < hi) if hi < 1.5 else (thick >= lo)
        if m.sum() >= 15 and len(np.unique(true[m])) > 1:
            p = KMeans(K, n_init=10, random_state=seed).fit_predict(Z[m])
            bins.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()), ARI=float(ARI(true[m], p)), NMI=float(NMI(true[m], p))))
        else:
            bins.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()), ARI=None, NMI=None))
    return dict(ARI_all=None, NMI_all=None, NMI_cluster_vs_thickbin=None, bins=bins)


def score(true, pred, thick):
    out = {"ARI_all": float(ARI(true, pred)), "NMI_all": float(NMI(true, pred)),
           "NMI_cluster_vs_thickbin": float(NMI(np.digitize(thick, BIN_EDGES[1:-1]), pred))}
    bins = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        m = (thick >= lo) & (thick < hi) if hi < 1.5 else (thick >= lo)
        if m.sum() >= 15 and len(np.unique(true[m])) > 1:
            bins.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()), ARI=float(ARI(true[m], pred[m])),
                             NMI=float(NMI(true[m], pred[m]))))
        else:
            bins.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()), ARI=None, NMI=None))
    out["bins"] = bins
    return out


def load_sample_merged(e):
    """Sample of dataset A (340/365/400/430) merged with the same sample of extension B (310/325/415), if present."""
    from spectral_select.types import SpectraData
    sd, lab, ex_ = load_sample(DS, e)
    if DS_B is None:
        return sd, lab, ex_
    sdb, labb, _ = load_sample(DS_B, e["id"])
    assert np.array_equal(lab, labb), "label maps of A and B differ"
    exc = {**sd.excitations, **sdb.excitations}
    return SpectraData(excitations={k: exc[k] for k in sorted(exc)}, mask=sd.mask, sample_name=sd.sample_name,
                       loading_options=sd.loading_options, metadata=sd.metadata), lab, ex_


def main():
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")
    idx = load_index(DS)
    rng = np.random.default_rng(0)
    AGG = "--agg" in sys.argv   # re-aggregate records.json only
    recs = json.loads((OUT / "records.json").read_text()) if AGG else []
    preds = {}
    R_RAND = 3 if QUICK else 10
    for e in ([] if AGG else idx["samples"]):
        sd, lab, ex_ = load_sample_merged(e)
        thick = np.nansum([ex_["maps"][f"layer_thickness:cover_{i}"] for i in range(3)], axis=0)  # per-block GT maps (disjoint)
        mask = lab > 0
        ch, cube = channels(sd)
        sets = channel_sets(ch, rng)
        true = lab[mask]
        th = thick[mask]
        ks = sorted({len(v) for k_, v in sets.items() if k_.startswith("uniform")} - {31})
        rand_sets = {f"random{k}_{j}": sorted(rng.choice(len(ch), k, replace=False).tolist()) for k in ks for j in range(R_RAND)}
        for sname, sidx in {**sets, **rand_sets}.items():
            fam = sname.rsplit("_", 1)[0] if sname.startswith("random") else sname
            res = {}
            extra = {}
            for s in (1, 3):
                cs = smooth(cube, mask, s)
                Z = features(cs, sidx, mask)
                res[f"kmeans_pca_s{s}"] = km(Z, 0)
                res[f"gmm_pca_s{s}"] = GaussianMixture(K, covariance_type="full", n_init=5, random_state=0).fit(Z).predict(Z)
                if s == 1:
                    res["kmeans_l2_s1"] = km(features(cs, sidx, mask, l2=True), 0)
                if fam in ("all", "e5_thick15", "e5_thickamp26", "e5_jointD5", "uniform15", "uniform26", "uniform5") or sname.startswith("random"):
                    extra[f"binfit_kmeans_s{s}"] = binfit_scores(Z, true, th)
                if fam in ("all", "e5_thick15", "e5_thickamp26", "e5_jointD5", "uniform15", "uniform26", "uniform5", "uniform31"):
                    Zf = features_full(cs, sidx, mask)
                    extra[f"lda_oracle_s{s}"] = lda_scores(Zf, true, th)
            if fam in ("all", "e5_thick15", "e5_jointD5", "uniform15", "uniform5"):
                sel = [ch[i] for i in sidx]
                t0 = time.time()
                zc, loss = cae_latent(sd, sel, mask, seed=int(e["index"]))
                res["cae_kmeans"] = km(zc, 0)
                print(f"  cae {e['id']} {sname}: {time.time() - t0:.0f}s loss {loss:.4f}", flush=True)
            for mname, pred in res.items():
                if int(e['index']) < 2 and sname in ('all', 'e5_thick15', 'uniform15'):
                    full = np.full(mask.shape, -1, np.int16); full[mask] = pred; preds[f"{e['id']}|{sname}|{mname}"] = full
                r = score(true, pred, th)
                r.update(sample=e["id"], set=sname, family=fam, method=mname, n_ch=len(sidx), t0=e["params"]["t0_mm"],
                         t1=e["params"]["t1_mm"], n_px=int(mask.sum()))
                recs.append(r)
            for mname, r in extra.items():
                r.update(sample=e["id"], set=sname, family=fam, method=mname, n_ch=len(sidx), t0=e["params"]["t0_mm"],
                         t1=e["params"]["t1_mm"], n_px=int(mask.sum()))
                recs.append(r)
        print(f"[{e['id']}] {time.time() - t_start:.0f}s", flush=True)
        (OUT / "records.json").write_text(json.dumps(recs))
        np.savez_compressed(OUT / "example_preds.npz", **preds)
    # ---- aggregate: per (family, method): mean +- SE over captures (randomK: mean over sets within capture first)
    agg = {}
    fams = sorted({r["family"] for r in recs})
    methods = sorted({r["method"] for r in recs})
    for fam in fams:
        for m in methods:
            rs = [r for r in recs if r["family"] == fam and r["method"] == m]
            if not rs:
                continue
            by_s = {}
            for r in rs:
                by_s.setdefault(r["sample"], []).append(r)
            samples = sorted(by_s)

            def cap_mean(f):
                out = []
                for s in samples:
                    v = [f(r) for r in by_s[s]]
                    v = [x for x in v if x is not None and np.isfinite(x)]
                    out.append(np.mean(v) if v else np.nan)
                return np.array(out, float)

            def stat(a):
                a = a[np.isfinite(a)]
                if len(a) == 0:
                    return dict(mean=None, se=None, n=0)
                return dict(mean=float(a.mean()), se=float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None, n=int(len(a)))
            ent = dict(n_captures=len(samples), n_ch=rs[0]["n_ch"], ARI_all=stat(cap_mean(lambda r: r["ARI_all"])),
                       NMI_all=stat(cap_mean(lambda r: r["NMI_all"])),
                       NMI_cluster_vs_thickbin=stat(cap_mean(lambda r: r["NMI_cluster_vs_thickbin"])), bins=[])
            for b in range(len(BIN_EDGES) - 1):
                def fb(key):
                    return cap_mean(lambda r: r["bins"][b].get(key))
                ent["bins"].append(dict(lo=float(BIN_EDGES[b]), hi=float(BIN_EDGES[b + 1]), n_px_mean=float(np.mean([r["bins"][b]["n"] for r in rs])),
                                        ARI=stat(fb("ARI")), NMI=stat(fb("NMI")), acc=stat(fb("acc"))))
            agg[f"{fam}|{m}"] = ent
    out = dict(experiment="E9 unsupervised clustering vs cover thickness", quick=QUICK,
               settings=dict(dataset=str(DS), n_captures=idx["n"], K=K, epochs_cae=EPOCHS, bin_edges_mm=BIN_EDGES.tolist(),
                             random_sets=R_RAND, classes=idx["class_names"], excitations_nm=[310.0,325.0,340.0,365.0,400.0,415.0,430.0] if DS_B else idx["samples"][0]["excitations_nm"],
                             mode=idx["spec"]["mode"], pixels=idx["samples"][0]["shape"],
                             e5_channels="results/E5/results.json selection.{thick,thick+amp,joint_D}.0.acquisition.labels (E5 full run, eps=0)"),
               units=dict(thickness="mm", ARI="-", NMI="-"), aggregate=agg, runtime_s=time.time() - t_start, provenance=provenance())
    (OUT / "results.json").write_text(json.dumps(out, indent=1))
    print("done", time.time() - t_start)


if __name__ == "__main__":
    main()
