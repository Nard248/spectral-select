import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
H = Path(__file__).parent; F = H.parent / "figures"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.labelpad": 2, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2})
num = json.load(open(H / "validation_numbers.json")); cur = json.load(open(H / "drop_curves.json"))
rows = num["drops"]["rows"]; med = np.median([r["sam_deg"] for r in rows])
rep = min(rows, key=lambda r: abs(r["sam_deg"] - med))["drop"]
cc = json.load(open(H / "collagen_rep_curves.json")); err = num["collagen"]["error_twin_corrected"]
fig = plt.figure(figsize=(7.16, 2.4))
gs = fig.add_gridspec(1, 3, wspace=0.62, left=0.055, right=0.985, bottom=0.2, top=0.92, width_ratios=[1, 1, 0.8])
gsr = gs[0, 1].subgridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.1)
a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gsr[0]); br = fig.add_subplot(gsr[1], sharex=b)
c_ax = fig.add_subplot(gs[0, 2])
cols = {"365.0": "#1f5fa8", "400.0": "#c0392b"}
norm = max(max(cur[rep][k]["meas"]) for k in cols)
for ex, c in [(k, v) for k, v in cur[rep].items() if k in cols]:
    wl = np.array(c["wl"]); m = np.array(c["meas"]) / norm; p = np.array(c["pred"]) / norm
    a.plot(wl, m, "o", ms=2.2, color=cols[ex], mfc="none", mew=0.6, label=f"{float(ex):.0f} nm")
    a.plot(wl, p, "-", lw=1.0, color=cols[ex])
a.set_xlabel("Emission wavelength (nm)"); a.set_ylabel("Intensity (norm.)")
a.set_xlim(430, 720); a.set_ylim(0, 1.12)
a.legend(frameon=False, loc="upper right", handlelength=1, borderpad=0.1, labelspacing=0.2, handletextpad=0.3)
LET = dict(transform=None, va="bottom", ha="left", fontweight="bold", fontsize=8.5)
for _ax, _t in ((a, "(a)"), (b, "(b)"), (c_ax, "(c)")): _ax.text(-0.02, 1.03, _t, transform=_ax.transAxes, va="bottom", ha="left", fontweight="bold", fontsize=8.5)
r = [x for x in rows if x["drop"] == rep][0]
a.text(0.97, 0.62, f"SAM {r['sam_deg']:.1f}$^\\circ$", transform=a.transAxes, va="top", ha="right", linespacing=1.1)
a.spines[["top", "right"]].set_visible(False)
# (b) collagen held-out
ccols = {"310.0": "#6a3d9a", "325.0": "#1f5fa8", "340.0": "#e08a00"}
for ex, c in cc["curves"].items():
    wl = np.array(c["wl"]); m = np.array(c["meas"]); p = np.array(c["pred"]); s = m.max()
    b.plot(wl, m / s, "o", ms=2.0, color=ccols[ex], mfc="none", mew=0.6, label=f"{float(ex):.0f} nm")
    b.plot(wl, p / s, "-", lw=1.0, color=ccols[ex])
    br.plot(wl, (m - p) / s, "-", lw=0.8, color=ccols[ex])
    br.fill_between(wl, 0, (m - p) / s, color=ccols[ex], alpha=0.15, lw=0)
br.axhline(0, color="k", lw=0.4)
b.set_ylabel("Intensity (norm.)"); b.set_ylim(0, 1.12); b.set_xlim(415, 725)
b.legend(frameon=False, loc="upper right", handlelength=1, borderpad=0.1, labelspacing=0.2, handletextpad=0.3)
plt.setp(b.get_xticklabels(), visible=False)
br.set_ylim(-0.3, 0.3); br.set_yticks([-0.2, 0, 0.2]); br.set_ylabel("Resid. (norm.)")
br.set_xlabel("Emission wavelength (nm)")
for ax in (b, br): ax.spines[["top", "right"]].set_visible(False)
# (c) SAM bars
c = num["collagen"]["sam_deg_all"]
order = [("twin_corrected", "twin"), ("empirical", "empirical copy"), ("twin_nominal", "nominal twin"), ("flat", "flat")]
colr = ["#1f5fa8", "#2e8b57", "#7fa8d6", "#d0d0d0"]
y = np.arange(len(order))[::-1]
c_ax.barh(y, [c[k]["mean"] for k, _ in order], xerr=[c[k]["sd"] for k, _ in order], color=colr, height=0.68,
          error_kw=dict(lw=0.6, capsize=1.2, capthick=0.6), edgecolor="k", linewidth=0.3)
c_ax.set_yticks(y); c_ax.set_yticklabels([l for _, l in order], ha="right"); c_ax.set_xlabel("Held-out SAM (deg)")
c_ax.set_xlim(0, 36); c_ax.spines[["top", "right"]].set_visible(False)
for yy, (k, _) in zip(y, order):
    c_ax.text(c[k]["mean"] + c[k]["sd"] + 0.6, yy, f"{c[k]['mean']:.1f}", va="center")
fig.align_ylabels([b, br]); fig.align_xlabels([a, br, c_ax])
for e in ("png", "pdf"): fig.savefig(F / f"fig_realdata.{e}", dpi=400)
print("rep drop", rep)
