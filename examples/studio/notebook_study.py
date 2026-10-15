# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
# ---

# %% [markdown]
# # A reproducible global-sensitivity study with SpectralForge Studio
#
# Which design parameter of a layered tissue phantom controls the fluorescence signal: the
# cover thickness or the thickness of the fluorescent layer?  We run a Saltelli design (Sobol'
# indices, Saltelli 2010) on the `layered_wedge_phantom` preset in preview mode, fully headless.

# %%
import tempfile
from pathlib import Path

from spectraforge.studio import api
from spectraforge.studio.sensitivity import analyse_study, format_table
from spectraforge.studio.study import run_study
from spectraforge.studio.study_schema import StudySpec

# %% [markdown]
# ## 1. Base experiment (fewer Monte Carlo photons so the demo runs in a few minutes)

# %%
base = api.preset("layered_wedge_phantom")
base = api.set_many(base, {"/acquisition/volume_mc/excitation_photons": 100_000,
                           "/acquisition/volume_mc/emission_photons": 100_000})
print(api.methods_text(base).split("\n\n")[1])           # the generated Methods text for the scene

# %% [markdown]
# ## 2. Study specification
# The cover wedge is replaced by a uniform cover (start and end thickness are linked with `also_set`).
# Metric: summed emission over a central region of interest (ROI coordinates are image fractions).

# %%
roi = {"shape": "rect", "x0": 0.3, "y0": 0.3, "x1": 0.7, "y1": 0.7}
spec = StudySpec(
    name="cover vs fluorescent layer",
    base_experiment=base,
    parameters=[
        {"pointer": "/objects/0/shape/layers/2/thickness/thickness_mm", "label": "cover_mm",
         "also_set": ["/objects/0/shape/layers/2/thickness/end_thickness_mm"], "low": 0.1, "high": 1.5},
        {"pointer": "/objects/0/shape/layers/1/thickness/thickness_mm", "label": "fluor_mm", "low": 0.1, "high": 1.0},
    ],
    design={"kind": "saltelli", "n": 8, "seed": 0},      # N (k + 2) = 32 design points (use 2**10+ for papers)
    replicates=2,                                         # Monte Carlo seeds 0, 1: gives the noise floor
    mode="preview",
    metrics=[{"name": "signal", "kind": "band_sum", "roi": roi}],
)

# %% [markdown]
# ## 3. Run (resumable: re-running this cell skips finished runs) and analyse

# %%
out = Path(tempfile.gettempdir()) / "spectraforge_demo_study"
result = run_study(spec, out, progress=lambda f, m: print(f"\r{f:5.0%} {m}", end=""))
print("\n", result)
sens = analyse_study(out, n_boot=200)
print(format_table(sens))

# %% [markdown]
# Indices are accompanied by bootstrap 95 % confidence intervals and the replicate-noise floor
# (the fraction of output variance that is pure Monte Carlo noise).  `runs.parquet`, `runs.csv`,
# `manifest.json` (spec hash, git commit, package versions) and `sensitivity.json` are in `out`.
