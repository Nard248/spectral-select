import json

from spectraforge.studio import api
from spectraforge.studio.provenance import canonical_json, experiment_hash, provenance


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1}) == '{"a":[1,2],"b":1}'


def test_experiment_hash_stable_and_sensitive():
    e = api.preset("generic_macro_bench")
    h = experiment_hash(e)
    assert len(h) == 64 and h == experiment_hash(api.preset("generic_macro_bench"))
    assert h == experiment_hash(e.model_dump(mode="json"))          # dict and model hash identically
    e2 = api.set(e, "/acquisition/seed", 5)
    assert experiment_hash(e2) != h


def test_provenance_fields():
    p = provenance()
    for k in ("package_version", "git", "python", "numpy", "numba", "platform", "cpu_count", "timestamp_utc"):
        assert k in p
    assert set(p["git"]) == {"commit", "branch", "dirty"}
    assert "threading_layer" in p["numba"]
    json.dumps(p)                                                   # JSON-safe
    assert provenance({"x": 1})["x"] == 1
