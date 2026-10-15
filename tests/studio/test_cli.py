import json
import subprocess
import sys

import numpy as np
import pytest

from spectraforge.studio import cli
from spectraforge.studio import api

from .fakes_study import COVER, FLUOR, base_experiment


def spec_json(tmp_path, kind="saltelli", n=8):
    p = tmp_path / "study.json"
    p.write_text(json.dumps({
        "name": "cli", "base_experiment": "generic_macro_bench", "replicates": 1, "mode": "preview",
        "parameters": [{"pointer": "/acquisition/preview_scale", "label": "scale", "low": 0.1, "high": 0.2}],
        "design": {"kind": kind, "n": n, "seed": 1},
        "metrics": [{"name": "sig", "kind": "band_sum"}]}))
    return p


def test_presets_and_methods(capsys):
    assert cli.main(["presets"]) == 0
    assert "layered_wedge_phantom" in capsys.readouterr().out
    assert cli.main(["methods", "generic_macro_bench", "--latex"]) == 0
    out = capsys.readouterr().out
    assert r"\paragraph{Simulation.}" in out
    assert cli.main(["methods", "no_such_preset"]) == 2


def test_default_is_serve(monkeypatch):
    seen = {}
    import spectraforge.studio.server as server
    monkeypatch.setattr(server, "main", lambda argv=None: seen.setdefault("argv", argv))
    cli.main(["--port", "9999"])
    assert seen["argv"] == ["--host", "127.0.0.1", "--port", "9999"]
    seen.clear()
    cli.main([])
    assert seen["argv"][-2:] == ["--port", "8765"]


def test_run_study_sensitivity_dataset(tmp_path, capsys):
    out = tmp_path / "run"
    assert cli.main(["run", "generic_macro_bench", "--mode", "preview", "--out", str(out), "--png", "--seed", "2"]) == 0
    npz = next(out.glob("*.npz"))
    res = api.load_capture(npz)
    assert res.meta["provenance"]["seed"] == 2 and list(out.glob("*.png")) and list(out.glob("*.experiment.json"))
    sdir = tmp_path / "st"
    assert cli.main(["study", str(spec_json(tmp_path)), "--out", str(sdir), "--quiet"]) == 0
    assert (sdir / "runs.parquet").exists()
    capsys.readouterr()
    assert cli.main(["sensitivity", str(sdir), "--boot", "20"]) == 0
    assert "S1" in capsys.readouterr().out and (sdir / "sensitivity.json").exists()
    ds = tmp_path / "ds.json"
    ds.write_text(json.dumps({"base_experiment": "generic_macro_bench", "n": 2, "label_scheme": "object",
                              "parameters": [{"pointer": "/acquisition/preview_scale", "label": "scale",
                                              "low": 0.1, "high": 0.2}]}))
    assert cli.main(["dataset", str(ds), "--out", str(tmp_path / "dsout"), "--quiet"]) == 0
    assert (tmp_path / "dsout" / "datasets.json").exists()


def test_bad_inputs_return_2(tmp_path, capsys):
    sdir = tmp_path / "st"
    assert cli.main(["study", str(spec_json(tmp_path, "lhs", 3)), "--out", str(sdir), "--quiet"]) == 0
    assert cli.main(["sensitivity", str(sdir)]) == 2                      # lhs design
    assert "saltelli" in capsys.readouterr().err
    assert cli.main(["run", "nope_preset", "--out", str(tmp_path / "x")]) == 2


def test_module_entry_point():
    r = subprocess.run([sys.executable, "-m", "spectraforge.studio", "presets"], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0 and "mehsi_lab_replica" in r.stdout
