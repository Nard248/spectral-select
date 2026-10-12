"""Random K-band subsets evaluated under the SAME KNN/ROI protocol as the paper (P = all labelled px; H = excluding ROI px)."""
import json, sys, time, numpy as np
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
ROOT=Path("/Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering"); sys.path.insert(0,str(ROOT/"src"))
from spectral_select import SpectraData
DATA=ROOT/"Data/processed/Lichens Dataset 1"; SP=Path(__file__).resolve().parent
sd=SpectraData.from_pickle(DATA/"spectra_masked.pkl"); cubes=[np.asarray(sd.excitations[ex].cube,dtype=np.float32) for ex in sd.excitations]
X=np.concatenate(cubes,axis=2); H,W,B=X.shape
mask=np.array(Image.open(DATA/"class_mask.png")); roi=json.load(open(DATA/"roi_regions.json")); gt=np.full((H,W),-1,int)
for c in roi["classes"]:
    col=tuple(c["color"]); m=np.all(mask[:,:,:3]==col,axis=2) if mask.shape[-1]==4 else np.all(mask==col,axis=2); gt[m]=c["id"]
valid=gt>=0; tm=np.zeros((H,W),bool)
for r in roi["regions"]:
    rc=r["rect"]; tm[rc["row_min"]:rc["row_max"],rc["col_min"]:rc["col_max"]]=True
tm&=valid; Xv=np.nan_to_num(X[valid]); yv=gt[valid]; tr=tm[valid]; ho=~tr
# subsample the evaluation pixels (stratified random 40k) to keep each draw ~0.5 s; training px always all used
rng=np.random.default_rng(0); ev_idx=rng.choice(np.where(ho)[0], 40000, replace=False)
def acc(idx):
    F=Xv[:,idx]; sc=StandardScaler().fit(F[tr]); knn=KNeighborsClassifier(n_neighbors=5,n_jobs=-1).fit(sc.transform(F[tr]),yv[tr])
    p=knn.predict(sc.transform(F[ev_idx])); return float((p==yv[ev_idx]).mean())
out={}
for K in [9,13]:
    t0=time.time(); accs=[acc(sorted(rng.choice(B,K,replace=False))) for _ in range(300)]
    accs=np.array(accs); out[K]={"n":300,"mean":accs.mean(),"std":accs.std(),"min":accs.min(),"max":accs.max(),"median":float(np.median(accs)),"p95":float(np.percentile(accs,95)),"p99":float(np.percentile(accs,99))}
    print(f"K={K}: n=300 random subsets (held-out 40k px): mean={accs.mean():.4f} std={accs.std():.4f} min={accs.min():.4f} median={np.median(accs):.4f} p95={np.percentile(accs,95):.4f} p99={np.percentile(accs,99):.4f} max={accs.max():.4f}  [{time.time()-t0:.0f}s]", flush=True)
# reference: learned selections on the same 40k subset
E=ROOT/"results/Lichens_Dataset_1_MasterRun/experiments"
cols=[]; 
for ex in sd.excitations: cols+=[(float(ex),float(w)) for w in sd.excitations[ex].emission_wavelengths]
ci={c:i for i,c in enumerate(cols)}
for cfg in ["bands_9_pca_dim_3_abso_mag_medium_max","bands_13_pca_dim_3_perc_mag_medium_max","bands_9_pca_dim_3_perc_mag_medium_max"]:
    d=json.load(open(E/cfg/"wavelengths.json")); idx=[ci[(float(e["excitation"]),float(e["emission"]))] for e in d]
    a=acc(idx); out[cfg]=a; print(f"learned {cfg}: {a:.4f}", flush=True)
out["baseline192"]=acc(list(range(B))); print("baseline192:", round(out["baseline192"],4))
json.dump(out,open(SP/"probe_random_subsets_results.json","w"),indent=1); print("DONE")
