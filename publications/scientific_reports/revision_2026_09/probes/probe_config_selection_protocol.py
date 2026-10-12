"""Feasibility probe for the revision: evaluate (a) the paper's oracle selections, (b) a fixed
pre-specified configuration, (c) label-free consensus rankings, under three evaluation variants:
 P = paper protocol (train on ROI px, evaluate on ALL labelled px incl. ROI),
 H = held-out (evaluate on labelled px EXCLUDING the ROI training px),
 R = rows 2-4 specimens only (excludes the whole first specimen row that hosts the ROIs).
No autoencoder is touched; everything comes from cached rankings + the processed cube."""
import json, sys, time, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
ROOT = Path("/Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering")
sys.path.insert(0, str(ROOT/"src"))
from spectral_select import SpectraData
DATA = ROOT/"Data/processed/Lichens Dataset 1"; EXP = ROOT/"results/Lichens_Dataset_1_MasterRun/experiments"
SP = Path(__file__).resolve().parent
t0=time.time(); sd = SpectraData.from_pickle(DATA/"spectra_masked.pkl"); print("loaded pkl in %.0fs"%(time.time()-t0), flush=True)
exs = list(sd.excitations.keys()); cols=[]; cubes=[]
for ex in exs:
    ed = sd.excitations[ex]; wl = list(ed.emission_wavelengths); cubes.append(np.asarray(ed.cube, dtype=np.float32))
    cols += [(float(ex), float(w)) for w in wl]
X = np.concatenate(cubes, axis=2); H,W,B = X.shape; print("cube", X.shape, "bands", len(cols), flush=True)
col_index = {c:i for i,c in enumerate(cols)}
# ground truth + ROIs (same decoding as experiments/rerun_knn.py)
mask = np.array(Image.open(DATA/"class_mask.png")); roi = json.load(open(DATA/"roi_regions.json"))
gt = np.full((H,W), -1, int)
for c in roi["classes"]:
    color = tuple(c["color"]); m = np.all(mask[:,:,:3]==color, axis=2) if mask.shape[-1]==4 else np.all(mask==color, axis=2); gt[m]=c["id"]
valid = gt>=0; train_mask = np.zeros((H,W), bool)
for r in roi["regions"]:
    rc=r["rect"]; train_mask[rc["row_min"]:rc["row_max"], rc["col_min"]:rc["col_max"]] = True
train_mask &= valid
lab, n = ndimage.label(valid); centers = ndimage.center_of_mass(valid, lab, range(1,n+1)); print("components:", n, flush=True)
row1_ids = [i+1 for i,(cy,cx) in enumerate(centers) if cy < 300]; rows234 = valid & ~np.isin(lab, row1_ids)
print("row-1 components:", len(row1_ids), "| eval px: all=%d heldout=%d rows2-4=%d"%(valid.sum(), (valid&~train_mask).sum(), rows234.sum()), flush=True)
Xv = np.nan_to_num(X[valid]); yv = gt[valid]; tr = train_mask[valid]; ho = ~tr; r234 = rows234[valid]
def evaluate(idx, name):
    F = Xv[:, idx]; sc = StandardScaler().fit(F[tr]); Ft = sc.transform(F)
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(Ft[tr], yv[tr]); p = knn.predict(Ft)
    out = {"name":name, "K":len(idx), "P":accuracy_score(yv,p), "H":accuracy_score(yv[ho],p[ho]), "R":accuracy_score(yv[r234],p[r234]), "kappa_P":cohen_kappa_score(yv,p)}
    print("%-42s K=%3d  P=%.4f  H=%.4f  R=%.4f  kappa=%.4f"%(name,len(idx),out["P"],out["H"],out["R"],out["kappa_P"]), flush=True); return out
def idx_from_json(path, K=None):
    d = json.load(open(path)); d = sorted(d, key=lambda e:e["rank"]); d = d[:K] if K else d
    return [col_index[(float(e["excitation"]), float(e["emission"]))] for e in d]
res=[]
res.append(evaluate(list(range(B)), "baseline all 192"))
for K,cfg in [(9,"bands_9_pca_dim_3_abso_mag_medium_max"),(13,"bands_13_pca_dim_3_perc_mag_medium_max"),(80,"bands_80_pca_dim_1_perc_mag_medium_non")]:
    res.append(evaluate(idx_from_json(EXP/cfg/"wavelengths.json"), f"ORACLE paper {cfg}"))
for K in [9,13,20,30,80]:
    res.append(evaluate(idx_from_json(EXP/f"bands_{K}_pca_dim_1_perc_mag_medium_non"/"wavelengths.json"), f"FIXED D pca/k1/perc/med/none K={K}"))
for K,norm in [(9,"max"),(10,"max"),(13,"max"),(20,"non"),(30,"non"),(80,"non")]:
    res.append(evaluate(idx_from_json(EXP/f"bands_{K}_pca_dim_3_perc_mag_medium_{norm}"/"wavelengths.json"), f"RULE k3 (max if K<16 else none) K={K}"))
cons = json.load(open(SP/"consensus_rankings_lichens.json"))
for fam in ["all48","pca24"]:
    for K in [9,13,30,80]:
        idx=[col_index[(float(e),float(m))] for e,m in cons[fam][:K]]; res.append(evaluate(idx, f"CONSENSUS {fam} top-{K}"))
json.dump(res, open(SP/"knn_probe_results.json","w"), indent=1); print("DONE total %.0fs"%(time.time()-t0))
