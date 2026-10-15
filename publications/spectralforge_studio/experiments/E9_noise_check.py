"""E9 control: per-pixel noise of the 'fast' E9 captures. Same scene (t0=0.1, t1=1.2) captured with two seeds at MC preview
fraction 0.15 (dataset setting) and 0.6 (4x photons); relative noise per channel = std(a-b)/sqrt(2)/mean over tissue pixels,
median over channels.  Total = MC + sensor; the 4x-photon run separates the MC share (MC variance ~ 1/photons)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from E9_scene import build_e9, api
from spectraforge.studio.dataset import label_map
out = {}
for f in (0.15, 0.6):
    caps = []
    for seed in (1, 2):
        exp = build_e9(0.1, 1.2, seed=seed, excitations=[340.0, 430.0])
        exp = api.set(exp, "/acquisition/volume_mc/preview_fraction", f)
        caps.append(api.capture(exp, mode="fast", excitations=[340.0, 430.0]))
    lab, _ = label_map(caps[0], "object"); m = lab > 0
    cubes = [np.concatenate([np.asarray(r.adu[k], float) for k in sorted(r.adu)], axis=-1)[m] for r in caps]  # (n_ex,H,W,B) -> (H,W,n_ex*B)
    a, b = cubes; mu = (a + b) / 2
    rel = (np.std(a - b, axis=0) / np.sqrt(2)) / np.maximum(np.abs(mu.mean(0)), 1e-12)
    pix = np.abs(a - b) / np.sqrt(2) / np.maximum(np.abs(mu), 1e-9)
    ok = mu.mean(0) > 0.05 * mu.mean(0).max()
    out[str(f)] = dict(median_rel_noise_per_channel=float(np.median(rel[ok])), n_channels_used=int(ok.sum()), n_px=int(m.sum()),
                       mc_photons="2e6 x fraction (0.15 -> 3e5; 0.6 -> 1.2e6) excitation and emission")
    print(f, out[str(f)], flush=True)
p = out["0.15"]["median_rel_noise_per_channel"]; q = out["0.6"]["median_rel_noise_per_channel"]
mc2 = max((p**2 - q**2) / (1 - 0.25), 0.0); out["mc_rel_noise_at_0.15"] = float(np.sqrt(mc2))
out["sensor_plus_other_rel_noise"] = float(np.sqrt(max(p**2 - mc2, 0)))
print(out)
Path("publications/spectralforge_studio/results/E9").mkdir(parents=True, exist_ok=True)
Path("publications/spectralforge_studio/results/E9/noise_check.json").write_text(json.dumps(out, indent=1))
