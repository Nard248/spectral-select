# recwl sweep (started 2026-10-22 ~22:05)
Rerun of the wavelength-aware models of the 'night' sweep with the emission axes recorded in the .im3 headers
(ME_ATTENTION_RECORDED_WL=1, see RECORDED_WL_OFFSET in src/me_attention/data.py and docs/datasets/audit_2026-10-22).
Same datasets, folds, seeds, budget (T=3, E=25), repeats and preprocessing as 'night'. Baselines are not rerun: their
inputs (feature values and order) are identical under the relabelling, so their 'night' results stand.
Not corrected here: the Lichens 2 400 nm exposure (metadata 12.5 ms vs 125 ms in the file).
