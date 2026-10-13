"""Command line interface of SpectralForge Studio.

::

    spectralforge-studio [serve] [--host H] [--port P] [--open]     # default: web UI + engine
    spectralforge-studio run EXP.json|PRESET --mode preview|full --out DIR [--png] [--seed N]
    spectralforge-studio study STUDY.json --out DIR [--workers N] [--no-resume] [--keep-captures]
    spectralforge-studio sensitivity DIR [--boot N]                  # print + save Sobol'/Morris tables
    spectralforge-studio dataset SPEC.json --out DIR
    spectralforge-studio methods EXP.json|PRESET [--latex] [--study STUDY.json]
    spectralforge-studio presets

``python -m spectraforge.studio`` is the same program.  Exit status is 0 on success, 2 for invalid input, 1 for
runtime failures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SUBCOMMANDS = {"serve", "run", "study", "sensitivity", "dataset", "methods", "presets"}


def _load_experiment(arg: str):
    from . import api
    if Path(arg).suffix.lower() == ".json" or Path(arg).exists():
        return api.load(arg)
    return api.preset(arg)


def _progress_printer(quiet: bool):
    def cb(f: float, msg: str) -> None:
        if not quiet:
            print(f"\r[{f * 100:5.1f}%] {msg:<40}", end="", file=sys.stderr, flush=True)
    return cb


def _cmd_presets(a) -> int:
    from . import api
    from .presets import PRESETS
    for name in api.presets():
        exp = PRESETS[name]()
        print(f"{name:<28} {exp.name}")
    return 0


def _cmd_run(a) -> int:
    from . import api
    exp = _load_experiment(a.experiment)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = api.capture(exp, mode=a.mode, seed=a.seed, progress=lambda f, m: None)
    stem = (exp.name or "capture").replace(" ", "_")
    api.save(res, out / f"{stem}.npz")
    api.dump(exp, out / f"{stem}.experiment.json")
    (out / f"{stem}.provenance.json").write_text(json.dumps(res.meta.get("provenance", {}), indent=2))
    print(f"captured {len(res.excitations_nm)} excitation(s), {res.width}x{res.height} px -> {out / (stem + '.npz')}")
    if a.png:
        from . import render_png
        for ex in res.excitations_nm:
            rgb = render_png.true_color(res.radiance[ex], res.emission_nm[ex], 1.0)
            (out / f"{stem}_ex{ex:g}.png").write_bytes(render_png.encode_png(rgb))
    return 0


def _cmd_study(a) -> int:
    from .study import StudyError, run_study
    from .study_schema import StudySpec
    spec = StudySpec.model_validate_json(Path(a.spec).read_text())
    try:
        r = run_study(spec, a.out, workers=a.workers, resume=not a.no_resume, keep_captures=a.keep_captures,
                      progress=_progress_printer(a.quiet))
    except StudyError as exc:
        print(f"invalid study: {exc}", file=sys.stderr)
        return 2
    print(file=sys.stderr)
    m = r.manifest
    print(f"study {m['name']!r}: {m['state']}, {m['n_completed']}/{m['n_runs']} runs "
          f"({m['n_failed']} failed, {m['n_resumed']} resumed) in {m['wall_time_s']:.1f} s -> {a.out}")
    return 0 if r.state == "done" and m["n_failed"] == 0 else 1


def _cmd_sensitivity(a) -> int:
    from .sensitivity import analyse_study, format_table
    try:
        res = analyse_study(a.dir, n_boot=a.boot)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_table(res))
    print(f"\nsaved {Path(a.dir) / 'sensitivity.json'} and sensitivity.csv")
    return 0


def _cmd_dataset(a) -> int:
    from .dataset import generate_dataset
    from .study_schema import DatasetSpec
    spec = DatasetSpec.model_validate_json(Path(a.spec).read_text())
    idx = generate_dataset(spec, a.out, progress=_progress_printer(a.quiet))
    print(file=sys.stderr)
    print(f"dataset {idx['name']!r}: {idx['n']} samples, classes {idx['class_names']} -> {a.out}")
    return 0


def _cmd_methods(a) -> int:
    from . import methods_text
    study = None
    if a.study:
        from .study_schema import StudySpec
        study = StudySpec.model_validate_json(Path(a.study).read_text())
    exp = _load_experiment(a.experiment)
    print(methods_text.methods_text(exp, study, style="latex" if a.latex else "markdown"))
    return 0


def _cmd_serve(a) -> int:
    from .server import main as serve_main
    argv = ["--host", a.host, "--port", str(a.port)] + (["--open"] if a.open else [])
    serve_main(argv)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spectralforge-studio",
                                description="SpectralForge Studio: simulator, studies, sensitivity analysis")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="start the engine + web UI (default)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--open", action="store_true")
    s.set_defaults(fn=_cmd_serve)
    r = sub.add_parser("run", help="run one capture")
    r.add_argument("experiment", help="experiment JSON file or preset name")
    r.add_argument("--mode", choices=["preview", "full", "sample", "dark", "white_reference"], default="full")
    r.add_argument("--out", required=True)
    r.add_argument("--png", action="store_true", help="also write true-colour PNGs")
    r.add_argument("--seed", type=int, default=None)
    r.set_defaults(fn=_cmd_run)
    st = sub.add_parser("study", help="run a design-of-experiments study")
    st.add_argument("spec")
    st.add_argument("--out", required=True)
    st.add_argument("--workers", type=int, default=1)
    st.add_argument("--no-resume", action="store_true")
    st.add_argument("--keep-captures", action="store_true")
    st.add_argument("--quiet", action="store_true")
    st.set_defaults(fn=_cmd_study)
    se = sub.add_parser("sensitivity", help="Sobol'/Morris indices of a finished saltelli/morris study")
    se.add_argument("dir")
    se.add_argument("--boot", type=int, default=300, help="bootstrap resamples")
    se.set_defaults(fn=_cmd_sensitivity)
    d = sub.add_parser("dataset", help="generate a synthetic ML dataset")
    d.add_argument("spec")
    d.add_argument("--out", required=True)
    d.add_argument("--quiet", action="store_true")
    d.set_defaults(fn=_cmd_dataset)
    m = sub.add_parser("methods", help="print a Methods paragraph for an experiment")
    m.add_argument("experiment")
    m.add_argument("--latex", action="store_true")
    m.add_argument("--study", help="StudySpec JSON to append a study-design paragraph")
    m.set_defaults(fn=_cmd_methods)
    pr = sub.add_parser("presets", help="list built-in presets")
    pr.set_defaults(fn=_cmd_presets)
    return p


def main(argv: list[str] | None = None) -> int | None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in _SUBCOMMANDS | {"-h", "--help"}:
        argv = ["serve", *argv]                      # legacy: `spectralforge-studio --port 9000`
    parser = build_parser()
    a = parser.parse_args(argv)
    try:
        return a.fn(a)
    except KeyError as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
