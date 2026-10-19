"""Analysis pipeline on tiny synthetic records (no real data, no torch)."""
import json
import math

import numpy as np
import pytest

from me_attention import analysis as A


def _rec(ds, model, track, fam, r, f, bacc, n_train=900, n_test=300, **kw):
    return dict(dataset=ds, model=model, track=track, family=fam, repeat=r, fold=f, n_train=n_train, n_test=n_test,
                budget=4, max_epochs=10, n_params=1000 if fam == "ours" else None, t_tune_s=1.0, t_final_s=1.0,
                test=dict(balanced_acc=bacc, macro_f1=bacc - 0.01, pixel_acc=bacc, specimen_avg_acc=bacc,
                          specimen_vote_acc=1.0, per_class_recall=[bacc, bacc, bacc]), **kw)


def _write(root, sweep, recs):
    for r in recs:
        p = root / sweep / r["dataset"] / f"{r['model']}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as fh:
            fh.write(json.dumps(r) + "\n")


@pytest.fixture
def sweep(tmp_path):
    rng = np.random.default_rng(0)
    recs = []
    for ds in ["collagen_pepsin", "lichens1", "lichens2"]:
        for r in range(2):
            for f in range(3):
                base = 0.70 + 0.01 * rng.standard_normal()
                recs += [_rec(ds, "lr", "spectral", "baseline", r, f, base),
                         _rec(ds, "rf", "spectral", "baseline", r, f, base - 0.05),
                         _rec(ds, "mlp", "spectral", "baseline", r, f, base - 0.02),
                         _rec(ds, "meas_tokens", "spectral", "ours", r, f, base + 0.05 + 0.002 * f),
                         _rec(ds, "axial", "spectral", "ours", r, f, base + 0.01)]
    recs.append({"dataset": "lichens1", "model": "lr", "repeat": 9, "fold": 0, "error": "boom"})
    _write(tmp_path, "t", recs)
    # partial: knn only has 2 units on one dataset; axial missing a unit on lichens2
    _write(tmp_path, "t", [_rec("lichens1", "knn", "spectral", "baseline", 0, f, 0.6) for f in range(2)])
    return tmp_path


def test_load_skips_errors_and_partial(sweep):
    df, info = A.load_sweep("t", sweep)
    assert info["n_errors"] == 1
    assert len(df) == 3 * 6 * 5 + 2
    assert set(df.dataset) == {"collagen_pepsin", "lichens1", "lichens2"}


def test_stats_known_case(sweep):
    df, _ = A.load_sweep("t", sweep)
    st = A.compute_stats(df)
    e = st["spectral"]["collagen_pepsin"]
    assert e["best_ours"] == "meas_tokens" and e["best_base"] == "lr"
    h = e["headline"]
    assert h["n"] == 6 and h["wins"] == 6 and h["losses"] == 0
    assert h["mean_diff"] == pytest.approx(0.05 + 0.002, abs=1e-9)
    assert h["ci95"][0] <= h["mean_diff"] <= h["ci95"][1]
    assert h["ratio"] == pytest.approx(300 / 900)
    assert 0 < h["wilcoxon_p"] <= 1
    # Holm across baselines never below raw p
    for v in e["ours"]["meas_tokens"]["vs_base"].values():
        assert v["wilcoxon_holm"] >= v["wilcoxon_p"] - 1e-12


def test_partial_units_use_intersection(sweep):
    df, _ = A.load_sweep("t", sweep)
    v = A.paired(df, "lichens1", "meas_tokens", "knn")
    assert v["n"] == 2
    st = A.compute_stats(df)
    assert st["spectral"]["lichens1"]["ours"]["meas_tokens"]["vs_base"]["knn"]["n"] == 2
    assert A.paired(df, "lichens2", "meas_tokens", "knn") is None  # model absent from this dataset


def test_ranks_ignore_sparse_models(sweep):
    df, _ = A.load_sweep("t", sweep)
    r = A.cross_dataset_ranks(df, "spectral")
    assert "knn" not in r["avg"]                      # < 80 % unit coverage on lichens1
    assert r["avg"]["meas_tokens"] == pytest.approx(1.0)
    assert r["n_datasets"] == 3 and r["k"] == 5 and r["cd_datasets"] > 0


def test_macro_names_latex_safe():
    for s in (A.macro_name("BestOurs", "cnn2d", "Spatial"), A.macro_name("AvgRank", "spectralformer_patch")):
        assert s.isalpha()


def test_end_to_end(sweep, tmp_path):
    out = tmp_path / "gen"
    A.run("t", out, sweep, figures=True, names=False)
    for f in ["tab_spectral.tex", "tab_spectral.md", "tab_spectral_extra.tex", "tab_stats.tex", "tab_stats.md",
              "numbers.tex", "summary.json", "figures/fig_box_spectral.pdf", "figures/fig_paired_spectral.pdf",
              "figures/fig_ranks_spectral.pdf", "figures/fig_cost_spectral.pdf", "figures/fig_recall_spectral.pdf"]:
        assert (out / f).exists(), f
    tex = (out / "tab_spectral.tex").read_text()
    assert r"\textbf{" in tex and r"\underline{" in tex and r"\toprule" in tex
    nums = (out / "numbers.tex").read_text()
    assert r"\newcommand{\BestOursNamePepsinSpectral}" in nums
    assert r"\newcommand{\NUnitsPepsinSpectral}{6}" in nums
    import re
    for name in re.findall(r"\\newcommand\{\\([^}]*)\}", nums):
        assert name.isalpha(), name


def test_empty_sweep(tmp_path):
    s = A.run("none", tmp_path / "o", tmp_path)
    assert s["n_records"] == 0
