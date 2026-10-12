# SpectralForge Studio — common rules for every engineer (read before your own brief)

You are one of six parallel engineers building **SpectralForge Studio**: a beautiful 3D experiment
designer + a physically comprehensive multi-excitation fluorescence hyperspectral simulator. The user
(a researcher, sole author of the SpectraForge engine) will demo it at IEEE WHISPERS 2026 (Glasgow,
Nov 17–19). **Top priorities: (1) very comprehensive, correct physics; (2) very presentable UI.**

## Workspace
- Worktree: `/Users/narekmeloyan/PycharmProjects/spectral-select-studio` (branch `studio/3d-editor`).
  Work ONLY inside it. Never modify `/Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering`
  (you only use its venv python).
- Python command (numpy 2.1, scipy 1.15, numba 0.63, pydantic 2.14, fastapi 0.143, uvicorn, httpx, hypothesis):
  `cd /Users/narekmeloyan/PycharmProjects/spectral-select-studio && PYTHONPATH=src /Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering/.venv/bin/python -m pytest tests/studio -q -o addopts=""`
- Web app: `apps/studio-web` (Vite 6, React 19, TS 5.8, three r186, @react-three/fiber 9, drei 10,
  @react-three/postprocessing 3, zustand 5, immer, vitest, @playwright/test + chromium installed).
- Do NOT git commit or push. Do NOT install Python packages. Only the web-app engineer may `npm install`.
- Do NOT edit the contracts: `src/spectraforge/studio/schema.py`, `contracts.py`, `presets.py`,
  `apps/studio-web/src/generated/*`, `apps/studio-web/src/api/types.ts`, `apps/studio-web/src/three/types.ts`
  (3D engineer may only ADD optional props), `apps/studio-web/src/three/pose.ts`.
  If a contract change is truly unavoidable, work around it and describe the needed change in your final report.

## Required reading
- `docs/superpowers/specs/2026-10-10-spectralforge-studio-design.md` (approved design)
- `src/spectraforge/studio/contracts.py` (engine data contracts + radiometric definitions — Python engineers MUST follow exactly)
- `src/spectraforge/studio/schema.py`, `presets.py` (experiment contract + realistic presets)
- `apps/studio-web/src/api/types.ts` (HTTP/WS API contract), `apps/studio-web/src/generated/experiment.ts`

## Ownership map (touch only your own files)
| engineer | owns |
|---|---|
| tracer | `src/spectraforge/studio/{geometry,surface,source,tracer}.py`, `tests/studio/test_{geometry,surface,source,tracer}.py` |
| transport | `src/spectraforge/studio/transport*.py`, `tests/studio/test_transport*.py` |
| capture | `src/spectraforge/studio/{spectra,optics,sensor,capture}.py`, `tests/studio/test_{spectra,optics,sensor,capture}.py` |
| server | `src/spectraforge/studio/{server,exports,jobs,render_png}.py`, `src/spectraforge/studio/__main__.py`, `tests/studio/test_{server,exports}.py`, `pyproject.toml` (scripts + `[studio]` extra only) |
| webapp | `apps/studio-web/**` EXCEPT `src/three/**`, `src/generated/**`, `src/api/types.ts` |
| three | `apps/studio-web/src/three/**` |

`tests/studio/__init__.py` must exist and be empty (create it if absent; harmless if two people do).
Modules being written concurrently by others may not exist yet: code against the contracts, import
other engineers' modules lazily (inside functions), and use stubs/fakes in your unit tests. Near the end
of your work, check whether the real modules have appeared and, if so, run an integration check and adapt
YOUR side if their API differs. Never edit their files.

## Quality bar
TDD where sensible; vectorised NumPy; docstrings with equations and literature references for physics;
small focused files; no silent failures. Finish with a concise final report (what you built, public API
signatures, decisions/approximations, measured performance, test results, contract changes needed).
