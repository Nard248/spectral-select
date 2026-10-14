import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
H = Path(__file__).parent; F = H.parent / "figures"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2})
num = json.load(open(H / "validation_numbers.json")); cur = json.load(open(H / "drop_curves.json"))
rows = num["drops"]["rows"]; med = np.median([r["sam_deg"] for r in rows])
rep = min(rows, key=lambda r: abs(r["sam_deg"] - med))["drop"]
fig, (a, b) = plt.subplots(1, 2, figsize=(3.4, 2.0), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.75))
cols = {"365.0": "#1f5fa8", "400.0": "#c0392b"}
norm = max(max(cur[rep][k]["meas"]) for k in cols)
for ex, c in [(k, v) for k, v in cur[rep].items() if k in cols]:
    wl = np.array(c["wl"]); m = np.array(c["meas"]) / norm; p = np.array(c["pred"]) / norm
    a.plot(wl, m, "o", ms=2.2, color=cols[ex], mfc="none", mew=0.6, label=f"{float(ex):.0f} nm")
    a.plot(wl, p, "-", lw=1.0, color=cols[ex])
a.set_xlabel("Emission wavelength (nm)"); a.set_ylabel("Intensity (norm.)")
a.set_xlim(430, 720); a.set_ylim(0, None)
a.legend(frameon=False, loc="upper right", handlelength=1, borderpad=0.1, labelspacing=0.2, handletextpad=0.3)
a.text(0.03, 0.97, "(a)", transform=a.transAxes, va="top", fontweight="bold")
r = [x for x in rows if x["drop"] == rep][0]
a.text(0.97, 0.60, f"markers: data\nlines: twin\nSAM {r['sam_deg']:.1f}$^\\circ$", transform=a.transAxes, va="top", ha="right", fontsize=7, linespacing=1.1)
c = num["collagen"]["sam_deg_all"]
order = [("in_sample", "refit (in-sample)"), ("twin_corrected", "twin"), ("empirical", "empirical copy"), ("twin_nominal", "nominal twin"), ("flat", "flat")]
colr = ["#9a9a9a", "#1f5fa8", "#2e8b57", "#7fa8d6", "#d0d0d0"]
y = np.arange(len(order))[::-1]
b.barh(y, [c[k]["mean"] for k, _ in order], xerr=[c[k]["sd"] for k, _ in order], color=colr, height=0.68,
       error_kw=dict(lw=0.6, capsize=1.2, capthick=0.6), edgecolor="k", linewidth=0.3)
b.set_yticks(y); b.set_yticklabels([l for _, l in order]); b.set_xlabel("Held-out SAM (deg)")
b.set_xlim(0, 33); b.spines[["top", "right"]].set_visible(False)
b.text(0.97, 0.97, "(b)", transform=b.transAxes, va="top", ha="right", fontweight="bold")
for yy, (k, _) in zip(y, order):
    b.text(c[k]["mean"] + c[k]["sd"] + 0.6, yy, f"{c[k]['mean']:.1f}", va="center", fontsize=7)
a.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(left=0.12, right=0.985, bottom=0.2, top=0.93)
for e in ("png", "pdf"): fig.savefig(F / f"fig_realdata.{e}", dpi=600)
print("rep drop", rep)
