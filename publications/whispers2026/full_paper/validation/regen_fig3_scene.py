import sys, pathlib
sys.path.insert(0,'/Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering/reports')
import paper_figures as pf
import numpy as np, matplotlib.pyplot as plt
F = pathlib.Path('/Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering/publications/whispers2026/full_paper/figures')
plt.rcParams.update({"font.size":8,"axes.grid":True,"grid.alpha":0.3,"font.family":"serif","font.serif":["Times New Roman","STIXGeneral"],"axes.titlesize":8,"legend.fontsize":7})
def sv(fig,name): fig.savefig(F/name,dpi=400,bbox_inches="tight"); plt.close(fig)
# scene
from classification_experiment import feature_matrix
from sklearn.feature_selection import f_classif
sp,gt,y,acq=pf.build_dataset(seed=1,size=64,em_step=5,nuisance_amp=1.2,photon_scale=2000,read_sigma=0.005)
X,colmap=feature_matrix(sp)
fig,ax=plt.subplots(1,2,figsize=(3.45,1.6),gridspec_kw=dict(width_ratios=[1,1.9]))
ax[0].imshow(y.reshape(64,64),cmap="tab10"); ax[0].set_title("class labels")
j=[k for k,(e,m) in enumerate(colmap) if e==470.0][20]
for a in ax[:1]: a.set_xticks([]);a.set_yticks([]);a.grid(False)
disc=pf.band_is_discriminative(colmap); var=X.var(0); Fv=np.nan_to_num(f_classif(X,y)[0])
ax[1].scatter(var,Fv,s=4,c=disc,cmap="coolwarm",alpha=0.6)
ax[1].set_xlabel("band variance"); ax[1].set_ylabel("discriminability (F)")
ax[1].set_title(f"variance vs. informativeness, corr={np.corrcoef(var,Fv)[0,1]:+.2f}")
fig.tight_layout(pad=0.3); sv(fig,"fig_scene_v2.png")
