#!/usr/bin/env python3
"""Export normalized Lichens 2 data for the native RL environment."""
import argparse, pickle, struct
from pathlib import Path
import numpy as np
from scipy import ndimage

MAGIC=b"HSIRL001"

def key(ex,em): return round(float(ex),6),round(float(em),6)
def valid(ex,em,d): return em>=ex+d and not (2*ex-d<=em<=2*ex+d)

def main():
    personal=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser()
    p.add_argument("--processed",type=Path,default=personal/"Processed/Lichens_2/data_cutoff_40nm_exposure_max_power_min.pkl")
    p.add_argument("--mask",type=Path,default=personal/"Processed/Lichens_2/lichens_2_class_mask.npy")
    p.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"data/lichens2.bin")
    p.add_argument("--offset-nm",type=float,default=40);p.add_argument("--max-pixels-per-object",type=int,default=200);p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    with a.processed.open("rb") as f: payload=pickle.load(f)
    mask=np.load(a.mask); available={}
    for k,r in payload["data"].items():
        ex=float(r.get("excitation",r.get("ex",k)));ems=np.asarray(r.get("wavelengths",r.get("em_arr")));cube=np.asarray(r.get("cube",r.get("data")))
        for i,em in enumerate(ems):available[key(ex,em)]=cube[...,i]
    raw=[]
    for k,r in payload.get("raw_data",payload["data"]).items():
        ex=float(r.get("ex",r.get("excitation",k)))
        raw.extend(key(ex,em) for em in np.asarray(r.get("em_arr",r.get("wavelengths"))))
    pairs=[v for v in raw if valid(*v,a.offset_nm) and v in available]
    group_image=np.full(mask.shape,-1,dtype=np.int32);gid=0
    for cls in sorted(int(v) for v in np.unique(mask) if v>=0):
        labels,count=ndimage.label(mask==cls)
        for component in range(1,count+1):group_image[labels==component]=gid;gid+=1
    rng=np.random.default_rng(a.seed);coords=[]
    for group in np.unique(group_image[mask>=0]):
        c=np.argwhere(group_image==group)
        if a.max_pixels_per_object and len(c)>a.max_pixels_per_object:c=c[rng.choice(len(c),a.max_pixels_per_object,replace=False)]
        coords.append(c)
    coords=np.concatenate(coords);rows,cols=coords[:,0],coords[:,1]
    X=np.column_stack([available[v][rows,cols] for v in pairs]).astype("<f4");y=mask[rows,cols].astype("<i4");groups=group_image[rows,cols].astype("<i4");pair_array=np.asarray(pairs,dtype="<f4")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("wb") as f:
        f.write(MAGIC);f.write(struct.pack("<IIII",X.shape[0],X.shape[1],int(y.max()+1),int(groups.max()+1)));f.write(pair_array.tobytes());f.write(y.tobytes());f.write(groups.tobytes());f.write(X.tobytes())
    print(f"Exported {len(y)} samples, {X.shape[1]} pairs, {y.max()+1} classes, {groups.max()+1} objects")

if __name__=="__main__":main()
