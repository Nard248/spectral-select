# Phase 2: experiment campaign (common rules)

Read `20-v12-common.md` (workspace, rules, quality bar). All v1.2 engine features are now implemented:
- tissue optics (`tissue_optics.py`, `presets_tissue.py`);
- inclusions in the 3D MC;
- the study engine (`api.py`, `study.py`, `sensitivity.py`, `dataset.py`, `methods_text.py`, `cli.py`; see
  `docs/studio/STUDY_API.md`);
- inverse tools (`unmix.py`, `surrogate.py`, `inverse.py`, `information.py`; see `docs/studio/INVERSE_API.md`);
- sim-to-real (`realdata.py`, `calibration.py`; see `docs/studio/SIM2REAL.md`).

Read the module docstrings before use.

You are a **computational scientist** running studies for a journal paper on SpectralForge Studio (target:
IEEE Transactions on Computational Imaging). Use the PUBLIC APIs (`spectraforge.studio.api`, `study`,
`information`, ...), the way a user would. If an API is missing or buggy, make the smallest fix in that
module, add a regression test, and list it in your report. Do not rewrite other engineers' modules.

## Output conventions
- The script goes in `publications/spectralforge_studio/experiments/<id>_<name>.py`. It must be runnable
  as `.venv/bin/python publications/spectralforge_studio/experiments/<file>.py [--quick]`. `--quick`
  finishes in under 2 min for CI; the full run takes 25 minutes or less. Fix all seeds.
- Results go in `publications/spectralforge_studio/results/<id>/`: `results.json` holds every number
  quoted in the paper (keys are self-explanatory, units in the key names or a `units` dict), plus
  `runs.parquet` where applicable and the provenance manifest (`provenance.provenance()`).
- Figures go in `publications/spectralforge_studio/figures/<id>_*.pdf` (vector) **and** `.png` (300 dpi).
  Use the IEEE column width 3.5 in (or double column 7.16 in), serif font matching Times, ≥ 8 pt labels, axis
  units, colour-blind-safe palettes (viridis/cividis or Okabe-Ito), and no chart junk. Look at every PNG you
  make, and iterate until it is publication quality.
- `publications/spectralforge_studio/results/<id>/FINDINGS.md`: 5–15 bullet findings, each with numbers
  and uncertainty (MC replicate SE or bootstrap CI), plus the honest caveats. These feed the paper directly.
- Statistical honesty matters: report the MC noise, replicate seeds, and CIs. Do not cherry-pick, and
  report negative results too.
- Do not commit. Touch only your own experiment files, plus minimal bug fixes as described above.
