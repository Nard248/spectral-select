# General Dependency-Aware Channel Selection — Implementation Plan (Foundation + PAMAP2 slice)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a domain-agnostic `channel_select` package whose perturbation+MMR selection engine works on any group-structured multi-channel data, and prove the transfer with a PAMAP2 (HAR) vertical slice producing an accuracy-vs-K curve.

**Architecture:** A shared `engine.py` operates over abstract `GroupStructuredModel` + `GroupedChannelData` protocols (group → channels → arbitrary axis). A `TemporalGroupedAutoencoder` (Conv1d-over-time) is the HAR model; the existing HSI autoencoder is the spatial2d case. The engine math mirrors `spectral_select/analyzer.py` so it can later be regression-tested against the frozen TPAMI core.

**Tech Stack:** Python 3.11, PyTorch 2.6 (MPS), scikit-learn 1.6, numpy 2.1, pytest.

**Scope:** This plan covers P1 (package + engine + temporal model) and P2 (PAMAP2 slice). P3 (full baselines + LOSO + HSI regression), P4 (Opportunity), P5 (ablations), P6 (figures) get a follow-up plan once P2 yields real numbers.

**Conventions:** Run all model code with `PYTORCH_ENABLE_MPS_FALLBACK=1`. Latent tensors are `(batch, *latent_dims)`; per-group reconstructions are `(batch, *axis_dims, channel)` — the **channel axis is always last**. Commit after each task.

---

### Task 1: Package skeleton + protocols

**Files:**
- Create: `channel_select/__init__.py`
- Create: `channel_select/protocols.py`
- Test: `tests/channel_select/test_protocols.py`
- Create: `tests/channel_select/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/channel_select/test_protocols.py
import pytest
from channel_select.protocols import SelectionConfig


def test_selection_config_defaults():
    cfg = SelectionConfig()
    assert cfg.dimension_selection_method == "variance"
    assert cfg.n_important_dimensions == 50
    assert cfg.perturbation_method == "percentile"
    assert cfg.perturbation_magnitudes == [10, 20, 30]
    assert cfg.normalization_method == "variance"
    assert cfg.n_channels_to_select == 10
    assert cfg.diversity_method == "mmr"
    assert cfg.lambda_diversity == 0.5


def test_selection_config_rejects_bad_method():
    with pytest.raises(ValueError):
        SelectionConfig(normalization_method="bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/channel_select/test_protocols.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'channel_select'`

- [ ] **Step 3: Write minimal implementation**

```python
# channel_select/__init__.py
"""Domain-agnostic dependency-aware channel selection.

Generalizes the spectral_select perturbation-autoencoder method to any
group-structured multi-channel data (hyperspectral cubes, sensor time series).
"""
from .protocols import SelectionConfig, GroupStructuredModel, GroupedChannelData

__all__ = ["SelectionConfig", "GroupStructuredModel", "GroupedChannelData"]
```

```python
# channel_select/protocols.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Hashable, Protocol, runtime_checkable

import torch

_VALID_DIM_METHODS = {"variance", "activation", "pca"}
_VALID_PERTURB_METHODS = {"percentile", "standard_deviation", "absolute_range"}
_VALID_NORM_METHODS = {"variance", "max_per_group", "none"}
_VALID_DIVERSITY = {"mmr", "min_distance", "none"}


@dataclass
class SelectionConfig:
    """Engine configuration. Field names mirror spectral_select.Config so the
    shared engine can be regression-tested against the HSI Analyzer."""

    dimension_selection_method: str = "variance"
    n_important_dimensions: int = 50
    perturbation_method: str = "percentile"
    perturbation_magnitudes: list[float] = field(default_factory=lambda: [10, 20, 30])
    perturbation_directions: list[str] = field(default_factory=lambda: ["bidirectional"])
    normalization_method: str = "variance"
    n_channels_to_select: int = 10
    diversity_method: str = "mmr"
    lambda_diversity: float = 0.5
    min_distance: float = 0.0

    def __post_init__(self) -> None:
        if self.dimension_selection_method not in _VALID_DIM_METHODS:
            raise ValueError(f"dimension_selection_method must be one of {_VALID_DIM_METHODS}")
        if self.perturbation_method not in _VALID_PERTURB_METHODS:
            raise ValueError(f"perturbation_method must be one of {_VALID_PERTURB_METHODS}")
        if self.normalization_method not in _VALID_NORM_METHODS:
            raise ValueError(f"normalization_method must be one of {_VALID_NORM_METHODS}")
        if self.diversity_method not in _VALID_DIVERSITY:
            raise ValueError(f"diversity_method must be one of {_VALID_DIVERSITY}")


@runtime_checkable
class GroupStructuredModel(Protocol):
    groups: list[Hashable]
    channels_per_group: dict[Hashable, int]

    def encode(self, batch: dict[Hashable, torch.Tensor]) -> torch.Tensor: ...
    def decode(self, latent: torch.Tensor) -> dict[Hashable, torch.Tensor]: ...


@runtime_checkable
class GroupedChannelData(Protocol):
    groups: list[Hashable]
    channels_per_group: dict[Hashable, int]
    axis_type: str  # "spatial2d" | "temporal1d"

    def get_all_data(self) -> dict[Hashable, torch.Tensor]: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/channel_select/test_protocols.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add channel_select/__init__.py channel_select/protocols.py tests/channel_select/__init__.py tests/channel_select/test_protocols.py
git commit -m "feat(channel_select): package skeleton + protocols/config"
```

---

### Task 2: GroupedChannelDataset container

**Files:**
- Create: `channel_select/data.py`
- Test: `tests/channel_select/test_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/channel_select/test_data.py
import torch
from channel_select.data import GroupedChannelDataset


def _toy():
    data = {
        "hand": torch.zeros(4, 8, 3),   # (window, time, channel)
        "ankle": torch.zeros(4, 8, 2),
    }
    labels = torch.tensor([0, 0, 1, 1])
    subjects = torch.tensor([1, 1, 2, 2])
    return GroupedChannelDataset(data, axis_type="temporal1d", labels=labels, subject_ids=subjects)


def test_properties():
    ds = _toy()
    assert ds.groups == ["hand", "ankle"]
    assert ds.channels_per_group == {"hand": 3, "ankle": 2}
    assert ds.axis_type == "temporal1d"
    assert ds.n_windows == 4


def test_subset_by_index():
    ds = _toy()
    sub = ds.subset([0, 2])
    assert sub.n_windows == 2
    assert sub.data["hand"].shape == (2, 8, 3)
    assert sub.labels.tolist() == [0, 1]
    assert sub.subject_ids.tolist() == [1, 2]


def test_loso_split_indices():
    ds = _toy()
    train_idx, test_idx = ds.loso_split(holdout_subject=2)
    assert train_idx == [0, 1]
    assert test_idx == [2, 3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/channel_select/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'channel_select.data'`

- [ ] **Step 3: Write minimal implementation**

```python
# channel_select/data.py
from __future__ import annotations
from typing import Hashable, Optional

import torch


class GroupedChannelDataset:
    """In-memory container conforming to GroupedChannelData.

    data: group -> Tensor[window, *axis_dims, channel]  (channel axis LAST)
    """

    def __init__(
        self,
        data: dict[Hashable, torch.Tensor],
        axis_type: str,
        labels: Optional[torch.Tensor] = None,
        subject_ids: Optional[torch.Tensor] = None,
    ) -> None:
        if axis_type not in ("spatial2d", "temporal1d"):
            raise ValueError(f"axis_type must be spatial2d|temporal1d, got {axis_type}")
        self.data = data
        self.axis_type = axis_type
        self.labels = labels
        self.subject_ids = subject_ids

    @property
    def groups(self) -> list[Hashable]:
        return list(self.data.keys())

    @property
    def channels_per_group(self) -> dict[Hashable, int]:
        return {g: int(t.shape[-1]) for g, t in self.data.items()}

    @property
    def n_windows(self) -> int:
        return int(next(iter(self.data.values())).shape[0])

    def get_all_data(self) -> dict[Hashable, torch.Tensor]:
        return self.data

    def subset(self, indices: list[int]) -> "GroupedChannelDataset":
        idx = torch.as_tensor(indices, dtype=torch.long)
        return GroupedChannelDataset(
            {g: t.index_select(0, idx) for g, t in self.data.items()},
            axis_type=self.axis_type,
            labels=None if self.labels is None else self.labels.index_select(0, idx),
            subject_ids=None if self.subject_ids is None else self.subject_ids.index_select(0, idx),
        )

    def loso_split(self, holdout_subject) -> tuple[list[int], list[int]]:
        if self.subject_ids is None:
            raise ValueError("loso_split requires subject_ids")
        train, test = [], []
        for i, s in enumerate(self.subject_ids.tolist()):
            (test if s == holdout_subject else train).append(i)
        return train, test
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/channel_select/test_data.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add channel_select/data.py tests/channel_select/test_data.py
git commit -m "feat(channel_select): GroupedChannelDataset container with LOSO split"
```

---

### Task 3: engine — select_important_dimensions (rank-agnostic)

**Files:**
- Create: `channel_select/engine.py`
- Test: `tests/channel_select/test_engine_dims.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/channel_select/test_engine_dims.py
import torch
from channel_select.engine import select_important_dimensions


def test_variance_ranks_high_variance_dims_first():
    # latent (batch=3, d0=2, d1=2). Make coord (1,1) highest variance.
    latent = torch.zeros(3, 2, 2)
    latent[:, 1, 1] = torch.tensor([0.0, 5.0, 10.0])   # high variance
    latent[:, 0, 0] = torch.tensor([1.0, 1.1, 0.9])    # low variance
    top = select_important_dimensions(latent, method="variance", n=1)
    assert len(top) == 1
    score, coord = top[0]
    assert coord == (1, 1)
    assert score > 0


def test_returns_n_sorted_descending():
    latent = torch.randn(5, 3, 4)
    top = select_important_dimensions(latent, method="variance", n=4)
    assert len(top) == 4
    scores = [s for s, _ in top]
    assert scores == sorted(scores, reverse=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/channel_select/test_engine_dims.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'channel_select.engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# channel_select/engine.py
"""Shared, domain-agnostic perturbation + selection engine.

Math mirrors spectral_select/analyzer.py, generalized so the only axis-specific
assumption is that the CHANNEL axis is last in each per-group reconstruction and
the BATCH axis is first in the latent.
"""
from __future__ import annotations
from typing import Hashable

import numpy as np
import torch


def select_important_dimensions(
    latent: torch.Tensor, method: str, n: int
) -> list[tuple[float, tuple[int, ...]]]:
    """Rank latent coordinates by importance. latent: (batch, *latent_dims)."""
    batch = latent.shape[0]
    latent_dims = tuple(latent.shape[1:])
    flat = latent.reshape(batch, -1)

    if method == "variance":
        scores = torch.var(flat, dim=0)
    elif method == "activation":
        scores = torch.mean(torch.abs(flat), dim=0)
    elif method == "pca":
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        scaled = StandardScaler().fit_transform(flat.cpu().numpy())
        n_comp = min(n * 2, scaled.shape[1], scaled.shape[0] - 1)
        pca = PCA(n_components=n_comp).fit(scaled)
        scores = torch.tensor(np.sum(np.abs(pca.components_), axis=0))
    else:
        raise ValueError(f"Unknown dimension selection method: {method}")

    coords: list[tuple[float, tuple[int, ...]]] = []
    for i, s in enumerate(scores):
        coord = tuple(int(c) for c in np.unravel_index(i, latent_dims))
        coords.append((float(s.item()), coord))
    coords.sort(reverse=True, key=lambda x: x[0])
    return coords[:n]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/channel_select/test_engine_dims.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add channel_select/engine.py tests/channel_select/test_engine_dims.py
git commit -m "feat(channel_select): rank-agnostic latent dimension selection"
```

---

### Task 4: engine — measure_channel_influence (generalized reduction)

**Files:**
- Modify: `channel_select/engine.py`
- Test: `tests/channel_select/test_engine_influence.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/channel_select/test_engine_influence.py
import numpy as np
import torch
from channel_select.engine import measure_channel_influence


class FakeModel:
    """decode() returns latent[:, :1] broadcast so influence = |Δlatent| per channel."""
    groups = ["g"]
    channels_per_group = {"g": 2}

    def decode(self, latent):
        # latent (batch, 2). Build recon (batch, time=3, channel=2):
        # channel 0 tracks latent dim 0, channel 1 tracks latent dim 1.
        b = latent.shape[0]
        recon = torch.zeros(b, 3, 2)
        recon[..., 0] = latent[:, 0:1]
        recon[..., 1] = latent[:, 1:2]
        return {"g": recon}


def test_reduces_all_axes_except_channel():
    model = FakeModel()
    base_latent = torch.zeros(1, 2)
    baseline_recon = model.decode(base_latent)
    pert = base_latent.clone()
    pert[:, 1] += 4.0  # perturb only channel-1's driver
    infl = measure_channel_influence(model, pert, baseline_recon, weight=1.0)
    assert infl["g"].shape == (2,)
    assert np.isclose(infl["g"][0], 0.0)
    assert np.isclose(infl["g"][1], 4.0)  # mean|Δ| over batch×time = 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/channel_select/test_engine_influence.py -v`
Expected: FAIL with `ImportError: cannot import name 'measure_channel_influence'`

- [ ] **Step 3: Write minimal implementation (append to engine.py)**

```python
def measure_channel_influence(
    model, perturbed_latent: torch.Tensor,
    baseline_recon: dict[Hashable, torch.Tensor], weight: float = 1.0,
) -> dict[Hashable, np.ndarray]:
    """Per-channel influence = mean |perturbed - baseline| over every axis
    except the last (channel) axis. Mirrors Analyzer._measure_band_influence
    where the HSI reduction dim=(0,1,2) leaves the band axis."""
    influence: dict[Hashable, np.ndarray] = {}
    with torch.no_grad():
        recon = model.decode(perturbed_latent)
        for g in model.groups:
            if g not in baseline_recon:
                continue
            base, pert = baseline_recon[g], recon[g]
            reduce_dims = tuple(range(pert.ndim - 1))  # all but channel (last)
            diff = torch.mean(torch.abs(pert - base), dim=reduce_dims)
            influence[g] = diff.cpu().numpy() * weight
    return influence
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/channel_select/test_engine_influence.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add channel_select/engine.py tests/channel_select/test_engine_influence.py
git commit -m "feat(channel_select): generalized per-channel influence reduction"
```

---

### Task 5: engine — perturbation amount + statistics (flat-index generalized)

**Files:**
- Modify: `channel_select/engine.py`
- Test: `tests/channel_select/test_engine_perturb.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/channel_select/test_engine_perturb.py
import numpy as np
import torch
from channel_select.engine import latent_statistics, perturbation_amount


def test_standard_deviation_amount():
    latent = torch.zeros(4, 2, 2)
    latent[:, 0, 1] = torch.tensor([0.0, 2.0, 4.0, 6.0])  # std ~2.236
    stats = latent_statistics(latent.reshape(4, -1))
    amt = perturbation_amount(
        coord=(0, 1), latent_dims=(2, 2), magnitude=100.0, sign=1,
        stats=stats, baseline_latent=latent, method="standard_deviation",
    )
    expected = 1.0 * np.std(latent[:, 0, 1].numpy())  # magnitude/100 * std
    assert np.isclose(amt, expected, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/channel_select/test_engine_perturb.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation (append to engine.py)**

```python
def latent_statistics(latent_flat: torch.Tensor) -> dict:
    pcts = [5, 10, 25, 50, 75, 90, 95]
    return {
        "std": torch.std(latent_flat, dim=0),
        "min": torch.min(latent_flat, dim=0)[0],
        "max": torch.max(latent_flat, dim=0)[0],
        "percentiles": {p: torch.quantile(latent_flat, p / 100.0, dim=0) for p in pcts},
    }


def _flat_index(coord: tuple[int, ...], latent_dims: tuple[int, ...]) -> int:
    return int(np.ravel_multi_index(coord, latent_dims))


def perturbation_amount(
    coord, latent_dims, magnitude, sign, stats, baseline_latent, method,
) -> float:
    idx = _flat_index(coord, latent_dims)
    sel = (slice(None),) + coord
    if method == "percentile":
        target_pct = 50 + sign * magnitude / 2
        closest = min(stats["percentiles"].keys(), key=lambda x: abs(x - target_pct))
        target = stats["percentiles"][closest][idx]
        current_mean = torch.mean(baseline_latent[sel])
        return float((target - current_mean).item())
    if method == "standard_deviation":
        return float(sign * (magnitude / 100.0) * stats["std"][idx].item())
    if method == "absolute_range":
        rng = stats["max"][idx] - stats["min"][idx]
        return float(sign * (magnitude / 100.0) * rng.item())
    raise ValueError(f"Unknown perturbation method: {method}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/channel_select/test_engine_perturb.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add channel_select/engine.py tests/channel_select/test_engine_perturb.py
git commit -m "feat(channel_select): generalized perturbation amount + latent stats"
```

---

### Task 6: engine — normalize_influence

**Files:**
- Modify: `channel_select/engine.py`
- Test: `tests/channel_select/test_engine_normalize.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/channel_select/test_engine_normalize.py
import numpy as np
import torch
from channel_select.engine import normalize_influence


def test_max_per_group():
    infl = {"g": np.array([2.0, 4.0])}
    out = normalize_influence(infl, data={}, method="max_per_group")
    assert np.allclose(out["g"], [0.5, 1.0])


def test_variance_divides_by_per_channel_variance():
    # data (window=3, time=1, channel=2); channel 0 has var 0, channel 1 var>0
    data = {"g": torch.tensor([[[1.0, 0.0]], [[1.0, 3.0]], [[1.0, 6.0]]])}
    infl = {"g": np.array([1.0, 9.0])}
    out = normalize_influence(infl, data=data, method="variance")
    var1 = np.var(np.array([0.0, 3.0, 6.0]))  # = 6.0
    assert out["g"][0] == 1.0 / 1e-10          # zero variance clamped
    assert np.isclose(out["g"][1], 9.0 / var1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/channel_select/test_engine_normalize.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation (append to engine.py)**

```python
def normalize_influence(
    influence: dict[Hashable, np.ndarray], data: dict[Hashable, torch.Tensor], method: str,
) -> dict[Hashable, np.ndarray]:
    out = {g: v.copy() for g, v in influence.items()}
    if method == "none":
        return out
    if method == "max_per_group":
        for g in out:
            m = float(np.max(out[g]))
            if m > 1e-10:
                out[g] = out[g] / m
        return out
    if method == "variance":
        for g in out:
            if g not in data:
                continue
            arr = data[g].cpu().numpy()
            var = np.var(arr, axis=tuple(range(arr.ndim - 1)))  # over all but channel
            var[var < 1e-10] = 1e-10
            out[g] = out[g] / var
        return out
    raise ValueError(f"Unknown normalization method: {method}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/channel_select/test_engine_normalize.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add channel_select/engine.py tests/channel_select/test_engine_normalize.py
git commit -m "feat(channel_select): per-group influence normalization (variance/max)"
```

---

### Task 7: engine — select_channels (MMR / min_distance / topk)

**Files:**
- Modify: `channel_select/engine.py`
- Test: `tests/channel_select/test_engine_select.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/channel_select/test_engine_select.py
import numpy as np
import torch
from channel_select.engine import select_channels


def test_topk_no_diversity():
    infl = {"g": np.array([0.1, 0.9, 0.5])}
    sel = select_channels(infl, data={}, K=2, method="none", lambda_diversity=0.5)
    assert sel == [("g", 1), ("g", 2)]  # by influence desc


def test_mmr_penalizes_redundant_channel():
    # channels 0 and 1 identical profile (redundant); channel 2 distinct.
    data = {"g": torch.tensor([
        [[1.0, 1.0, 0.0]],
        [[2.0, 2.0, 9.0]],
        [[3.0, 3.0, 1.0]],
    ])}  # (window=3, time=1, channel=3)
    infl = {"g": np.array([1.0, 0.95, 0.6])}
    sel = select_channels(data=data, influence=infl, K=2, method="mmr", lambda_diversity=0.9)
    # picks ch0 first (highest), then ch2 (distinct) over ch1 (redundant)
    assert sel[0] == ("g", 0)
    assert sel[1] == ("g", 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/channel_select/test_engine_select.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation (append to engine.py)**

```python
def _channel_catalog(influence):
    combos = []
    for g, vec in influence.items():
        for c, val in enumerate(vec):
            combos.append({"group": g, "channel": int(c), "influence": float(val)})
    combos.sort(key=lambda x: x["influence"], reverse=True)
    return combos


def _profiles(data):
    from sklearn.preprocessing import normalize
    prof = {}
    for g, t in data.items():
        arr = t.cpu().numpy()
        n_ch = arr.shape[-1]
        for c in range(n_ch):
            v = arr[..., c].reshape(1, -1)
            prof[(g, c)] = normalize(v, axis=1).flatten()
    return prof


def select_channels(
    influence, data, K, method="mmr", lambda_diversity=0.5, min_distance=0.0,
):
    combos = _channel_catalog(influence)
    if method == "none" or len(combos) <= K:
        return [(c["group"], c["channel"]) for c in combos[:K]]

    if method == "mmr":
        prof = _profiles(data)
        max_inf = combos[0]["influence"] or 1.0
        selected = [combos[0]]
        keys = [(combos[0]["group"], combos[0]["channel"])]
        while len(selected) < K:
            best, best_key, best_score = None, None, -np.inf
            for combo in combos:
                key = (combo["group"], combo["channel"])
                if key in keys or key not in prof:
                    continue
                rel = combo["influence"] / max_inf
                max_sim = max(
                    (abs(float(np.dot(prof[key], prof[sk]))) for sk in keys if sk in prof),
                    default=0.0,
                )
                score = rel - lambda_diversity * max_sim
                if score > best_score:
                    best, best_key, best_score = combo, key, score
            if best is None:
                break
            selected.append(best)
            keys.append(best_key)
        return [(c["group"], c["channel"]) for c in selected]

    if method == "min_distance":
        selected = []
        for combo in combos:
            key = (combo["group"], combo["channel"])
            ok = all(
                not (combo["group"] == s["group"]
                     and abs(combo["channel"] - s["channel"]) < min_distance)
                for s in selected
            )
            if ok:
                selected.append(combo)
            if len(selected) >= K:
                break
        return [(c["group"], c["channel"]) for c in selected]

    raise ValueError(f"Unknown diversity method: {method}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/channel_select/test_engine_select.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add channel_select/engine.py tests/channel_select/test_engine_select.py
git commit -m "feat(channel_select): MMR/min_distance/topk channel selection"
```

---

### Task 8: engine — run_selection orchestrator

**Files:**
- Modify: `channel_select/engine.py`
- Test: `tests/channel_select/test_engine_end_to_end.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/channel_select/test_engine_end_to_end.py
import torch
from channel_select.engine import run_selection
from channel_select.protocols import SelectionConfig


class TinyModel:
    groups = ["g"]
    channels_per_group = {"g": 4}

    def __init__(self):
        self._latent = torch.randn(6, 3)  # (batch, latent dims)

    def encode(self, batch):
        return self._latent

    def decode(self, latent):
        # channel c reconstructed from latent dim (c % 3), so perturbing a dim
        # influences a deterministic channel.
        b = latent.shape[0]
        recon = torch.zeros(b, 5, 4)  # (batch, time, channel)
        for c in range(4):
            recon[..., c] = latent[:, c % 3:c % 3 + 1]
        return {"g": recon}


def test_run_selection_returns_k_channels():
    model = TinyModel()
    data = {"g": torch.randn(6, 5, 4)}
    cfg = SelectionConfig(
        n_important_dimensions=3, n_channels_to_select=2,
        normalization_method="max_per_group", diversity_method="none",
        perturbation_method="standard_deviation", perturbation_magnitudes=[50],
    )
    result = run_selection(model, data, cfg)
    assert len(result.selected) == 2
    assert all(g == "g" for g, _ in result.selected)
    assert set(result.influence.keys()) == {"g"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/channel_select/test_engine_end_to_end.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_selection'`

- [ ] **Step 3: Write minimal implementation (append to engine.py)**

```python
from dataclasses import dataclass


@dataclass
class SelectionResult:
    selected: list[tuple]                 # [(group, channel), ...]
    influence: dict                       # group -> normalized influence vector
    important_dims: list                  # [(score, coord), ...]


def run_selection(model, data: dict, config) -> "SelectionResult":
    """Full engine pipeline on an already-trained model + its training data."""
    with torch.no_grad():
        latent = model.encode(data if isinstance(data, dict) else data.get_all_data())
        baseline_recon = model.decode(latent)

    important = select_important_dimensions(
        latent, config.dimension_selection_method, config.n_important_dimensions
    )
    latent_dims = tuple(latent.shape[1:])
    stats = latent_statistics(latent.reshape(latent.shape[0], -1))

    influence = {g: np.zeros(model.channels_per_group[g]) for g in model.groups}
    for score, coord in important:
        for mag in config.perturbation_magnitudes:
            for direction in config.perturbation_directions:
                signs = {"bidirectional": [-1, 1], "positive": [1], "negative": [-1]}[direction]
                weight = 0.5 if len(signs) == 2 else 1.0
                for sign in signs:
                    amt = perturbation_amount(
                        coord, latent_dims, mag, sign, stats, latent,
                        config.perturbation_method,
                    )
                    pert = latent.clone()
                    pert[(slice(None),) + coord] += amt
                    contrib = measure_channel_influence(model, pert, baseline_recon, score)
                    for g in contrib:
                        influence[g] += contrib[g] * weight

    influence = normalize_influence(influence, data, config.normalization_method)
    selected = select_channels(
        influence=influence, data=data, K=config.n_channels_to_select,
        method=config.diversity_method, lambda_diversity=config.lambda_diversity,
        min_distance=config.min_distance,
    )
    return SelectionResult(selected=selected, influence=influence, important_dims=important)
```

Note: `important_dims` weighting uses the dimension's importance score as `weight` in
`measure_channel_influence`, matching `Analyzer._compute_influence_scores`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/channel_select/test_engine_end_to_end.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add channel_select/engine.py tests/channel_select/test_engine_end_to_end.py
git commit -m "feat(channel_select): run_selection end-to-end orchestrator"
```

---

### Task 9: Temporal grouped autoencoder

**Files:**
- Create: `channel_select/models/__init__.py`
- Create: `channel_select/models/temporal.py`
- Test: `tests/channel_select/test_temporal_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/channel_select/test_temporal_model.py
import torch
from channel_select.models.temporal import TemporalGroupedAutoencoder
from channel_select.protocols import GroupStructuredModel


def test_encode_decode_shapes_roundtrip():
    channels = {"hand": 3, "ankle": 2}
    model = TemporalGroupedAutoencoder(channels_per_group=channels, time_len=32, latent_dim=8)
    batch = {"hand": torch.randn(4, 32, 3), "ankle": torch.randn(4, 32, 2)}
    latent = model.encode(batch)
    assert latent.shape[0] == 4
    recon = model.decode(latent)
    assert recon["hand"].shape == (4, 32, 3)
    assert recon["ankle"].shape == (4, 32, 2)


def test_conforms_to_protocol():
    model = TemporalGroupedAutoencoder({"hand": 3}, time_len=16, latent_dim=4)
    assert isinstance(model, GroupStructuredModel)
    assert model.groups == ["hand"]
    assert model.channels_per_group == {"hand": 3}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/channel_select/test_temporal_model.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# channel_select/models/__init__.py
from .temporal import TemporalGroupedAutoencoder
__all__ = ["TemporalGroupedAutoencoder"]
```

```python
# channel_select/models/temporal.py
"""Conv1d-over-time group-structured autoencoder.

Mirrors the HSI HyperspectralCAEWithMasking structure: per-group encoders ->
mean fusion -> shared latent -> per-group decoders. The spatial conv is replaced
by a temporal conv; the group axis plays the role of excitation.
"""
from __future__ import annotations
from typing import Hashable

import torch
import torch.nn as nn


def _key(g: Hashable) -> str:
    return str(g).replace(".", "_").replace(" ", "_")


class TemporalGroupedAutoencoder(nn.Module):
    def __init__(self, channels_per_group: dict[Hashable, int], time_len: int,
                 latent_dim: int = 16, hidden: int = 32) -> None:
        super().__init__()
        self.groups = list(channels_per_group.keys())
        self.channels_per_group = dict(channels_per_group)
        self.time_len = time_len
        self.latent_dim = latent_dim

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for g in self.groups:
            cin = channels_per_group[g]
            self.encoders[_key(g)] = nn.Sequential(
                nn.Conv1d(cin, hidden, kernel_size=5, padding=2), nn.ReLU(),
                nn.Conv1d(hidden, latent_dim, kernel_size=5, padding=2), nn.ReLU(),
            )
            self.decoders[_key(g)] = nn.Sequential(
                nn.Conv1d(latent_dim, hidden, kernel_size=5, padding=2), nn.ReLU(),
                nn.Conv1d(hidden, cin, kernel_size=5, padding=2),
            )

    def encode(self, batch: dict[Hashable, torch.Tensor]) -> torch.Tensor:
        feats = []
        for g in self.groups:
            x = batch[g].permute(0, 2, 1)            # (B, time, ch) -> (B, ch, time)
            feats.append(self.encoders[_key(g)](x))  # (B, latent_dim, time)
        stacked = torch.stack(feats, dim=1)          # (B, n_groups, latent_dim, time)
        return torch.mean(stacked, dim=1)            # (B, latent_dim, time) fusion

    def decode(self, latent: torch.Tensor) -> dict[Hashable, torch.Tensor]:
        out = {}
        for g in self.groups:
            y = self.decoders[_key(g)](latent)       # (B, ch, time)
            out[g] = y.permute(0, 2, 1)              # (B, time, ch)
        return out

    def forward(self, batch):
        return self.decode(self.encode(batch))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/channel_select/test_temporal_model.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add channel_select/models/__init__.py channel_select/models/temporal.py tests/channel_select/test_temporal_model.py
git commit -m "feat(channel_select): Conv1d grouped temporal autoencoder"
```

---

### Task 10: Minimal training loop

**Files:**
- Create: `channel_select/models/training.py`
- Test: `tests/channel_select/test_training.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/channel_select/test_training.py
import torch
from channel_select.models.temporal import TemporalGroupedAutoencoder
from channel_select.models.training import train_autoencoder


def test_loss_decreases_on_toy_data():
    torch.manual_seed(0)
    channels = {"hand": 3}
    model = TemporalGroupedAutoencoder(channels, time_len=16, latent_dim=4)
    data = {"hand": torch.randn(20, 16, 3)}
    history = train_autoencoder(model, data, epochs=15, lr=1e-2, batch_size=8, device="cpu")
    assert history[-1] < history[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/channel_select/test_training.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# channel_select/models/training.py
from __future__ import annotations
from typing import Hashable

import torch
import torch.nn.functional as F


def train_autoencoder(model, data: dict[Hashable, torch.Tensor], epochs: int = 25,
                      lr: float = 1e-3, batch_size: int = 32, device: str = "cpu") -> list[float]:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = next(iter(data.values())).shape[0]
    history: list[float] = []
    for _ in range(epochs):
        perm = torch.randperm(n)
        epoch_loss, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = {g: t.index_select(0, idx).to(device) for g, t in data.items()}
            opt.zero_grad()
            recon = model(batch)
            loss = sum(F.mse_loss(recon[g], batch[g]) for g in batch)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            nb += 1
        history.append(epoch_loss / max(nb, 1))
    return history
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/channel_select/test_training.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add channel_select/models/training.py tests/channel_select/test_training.py
git commit -m "feat(channel_select): minimal AE training loop"
```

---

### Task 11: PAMAP2 adapter

**Files:**
- Create: `channel_select/adapters/__init__.py`
- Create: `channel_select/adapters/pamap2.py`
- Test: `tests/channel_select/test_pamap2_adapter.py`

PAMAP2 `.dat` rows: col 0 = timestamp, col 1 = activity_id, col 2 = heart_rate, then
3 IMUs × 17 cols (hand, chest, ankle). Per IMU we use the 6 acc + 3 gyro + 3 mag = cols
offsets {1..3 (temp skip), 4..6 acc16g... }. For v1 use, per IMU, the 3-axis acc(±16g) at
local offsets 4,5,6 and gyro 10,11,12 and mag 13,14,15 → 9 channels/IMU. Activity 0
(transient) is dropped.

- [ ] **Step 1: Write the failing test (parser on a synthetic in-memory frame)**

```python
# tests/channel_select/test_pamap2_adapter.py
import numpy as np
from channel_select.adapters.pamap2 import windows_from_array, IMU_LOCAL_CHANNELS, GROUP_COL_OFFSETS


def test_windowing_groups_and_labels():
    # Build 100 rows, 54 cols. activity col=1. constant activity 4.
    arr = np.zeros((100, 54), dtype=float)
    arr[:, 1] = 4
    # put a ramp in hand-acc-x to verify it lands in the window
    hand_acc_x = GROUP_COL_OFFSETS["hand"] + IMU_LOCAL_CHANNELS[0]
    arr[:, hand_acc_x] = np.arange(100)
    ds = windows_from_array(arr, window=32, step=16, sampling_drop_transient=True)
    assert ds.axis_type == "temporal1d"
    assert ds.channels_per_group["hand"] == len(IMU_LOCAL_CHANNELS)
    assert "ankle" in ds.groups and "chest" in ds.groups
    assert ds.labels is not None
    assert int(ds.labels[0]) == 4
    # first window's hand-acc-x channel should be 0..31
    w0 = ds.data["hand"][0, :, 0].numpy()
    assert np.allclose(w0, np.arange(32))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/channel_select/test_pamap2_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# channel_select/adapters/__init__.py
```

```python
# channel_select/adapters/pamap2.py
"""PAMAP2 -> GroupedChannelDataset.

Dataset: UCI PAMAP2 Physical Activity Monitoring.
Manual download (if auto-fetch unavailable):
  https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip
Unzip to Data/Raw/PAMAP2/Protocol/*.dat (subject101.dat ... subject109.dat).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from ..data import GroupedChannelDataset

# Column 0 timestamp, 1 activity, 2 heart-rate; each IMU block is 17 cols.
GROUP_COL_OFFSETS = {"hand": 3, "chest": 20, "ankle": 37}
# Local offsets within an IMU block: acc16g (1,2,3), gyro (7,8,9), mag (10,11,12).
IMU_LOCAL_CHANNELS = [1, 2, 3, 7, 8, 9, 10, 11, 12]


def windows_from_array(arr: np.ndarray, window: int = 256, step: int = 128,
                       sampling_drop_transient: bool = True,
                       subject_id: int | None = None) -> GroupedChannelDataset:
    activity = arr[:, 1].astype(int)
    per_group_windows: dict[str, list[np.ndarray]] = {g: [] for g in GROUP_COL_OFFSETS}
    labels: list[int] = []
    subjects: list[int] = []
    for start in range(0, len(arr) - window + 1, step):
        sl = slice(start, start + window)
        win_act = activity[sl]
        # majority label; require homogeneous-enough window
        vals, counts = np.unique(win_act, return_counts=True)
        lab = int(vals[np.argmax(counts)])
        if sampling_drop_transient and lab == 0:
            continue
        for g, off in GROUP_COL_OFFSETS.items():
            cols = [off + c for c in IMU_LOCAL_CHANNELS]
            block = arr[sl][:, cols]                       # (window, n_ch)
            block = np.nan_to_num(block, nan=0.0)
            per_group_windows[g].append(block)
        labels.append(lab)
        subjects.append(subject_id if subject_id is not None else -1)

    data = {g: torch.tensor(np.stack(w), dtype=torch.float32)
            for g, w in per_group_windows.items()}
    return GroupedChannelDataset(
        data, axis_type="temporal1d",
        labels=torch.tensor(labels, dtype=torch.long),
        subject_ids=torch.tensor(subjects, dtype=torch.long),
    )


def load_pamap2(protocol_dir: str | Path, window: int = 256, step: int = 128,
                subjects: list[int] | None = None) -> GroupedChannelDataset:
    protocol_dir = Path(protocol_dir)
    files = sorted(protocol_dir.glob("subject1*.dat"))
    if not files:
        raise FileNotFoundError(
            f"No PAMAP2 .dat files in {protocol_dir}. See module docstring for download URL."
        )
    parts: list[GroupedChannelDataset] = []
    for f in files:
        sid = int(f.stem.replace("subject", ""))
        if subjects is not None and sid not in subjects:
            continue
        arr = np.loadtxt(f)
        parts.append(windows_from_array(arr, window, step, subject_id=sid))
    # concatenate
    groups = parts[0].groups
    data = {g: torch.cat([p.data[g] for p in parts], dim=0) for g in groups}
    labels = torch.cat([p.labels for p in parts])
    subjects_t = torch.cat([p.subject_ids for p in parts])
    return GroupedChannelDataset(data, "temporal1d", labels=labels, subject_ids=subjects_t)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/channel_select/test_pamap2_adapter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add channel_select/adapters/__init__.py channel_select/adapters/pamap2.py tests/channel_select/test_pamap2_adapter.py
git commit -m "feat(channel_select): PAMAP2 adapter (windowing + grouping)"
```

---

### Task 12: PAMAP2 vertical-slice experiment

**Files:**
- Create: `experiments/general_pamap2_slice.py`
- Create (output): `generalization/` workspace dirs

This is a runnable research script (not unit-tested). It downloads PAMAP2 if missing,
trains the temporal AE unsupervised, runs `run_selection`, evaluates accuracy-vs-K with
KNN under LOSO, and runs the grouped-vs-ungrouped sanity check.

- [ ] **Step 1: Create workspace + download helper**

```bash
mkdir -p generalization/{baselines,figures,reports} Data/Raw/PAMAP2
```

- [ ] **Step 2: Write the slice script**

```python
# experiments/general_pamap2_slice.py
"""PAMAP2 vertical slice: prove the engine transfers to HAR.

Run: PYTORCH_ENABLE_MPS_FALLBACK=1 python experiments/general_pamap2_slice.py
"""
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from channel_select.adapters.pamap2 import load_pamap2, GROUP_COL_OFFSETS, IMU_LOCAL_CHANNELS
from channel_select.models.temporal import TemporalGroupedAutoencoder
from channel_select.models.training import train_autoencoder
from channel_select.engine import run_selection
from channel_select.protocols import SelectionConfig

PROTOCOL = Path("Data/Raw/PAMAP2/Protocol")
WINDOW, STEP = 256, 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def flatten_features(ds, selected=None):
    """Per-window feature vector = per-channel mean+std over time, optionally
    restricted to selected (group, channel) pairs."""
    feats = []
    for g in ds.groups:
        arr = ds.data[g].numpy()  # (win, time, ch)
        for c in range(arr.shape[-1]):
            if selected is not None and (g, c) not in selected:
                continue
            feats.append(arr[:, :, c].mean(axis=1))
            feats.append(arr[:, :, c].std(axis=1))
    return np.stack(feats, axis=1)


def knn_loso(ds, selected, holdout=5):
    tr, te = ds.loso_split(holdout)
    Xtr = flatten_features(ds.subset(tr), selected)
    Xte = flatten_features(ds.subset(te), selected)
    ytr = ds.labels[tr].numpy(); yte = ds.labels[te].numpy()
    clf = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr)
    return accuracy_score(yte, clf.predict(Xte))


def main():
    ds = load_pamap2(PROTOCOL, window=WINDOW, step=STEP)
    print(f"Loaded {ds.n_windows} windows, groups={ds.groups}, "
          f"channels/group={ds.channels_per_group}")

    model = TemporalGroupedAutoencoder(ds.channels_per_group, time_len=WINDOW, latent_dim=16)
    hist = train_autoencoder(model, ds.get_all_data(), epochs=25, lr=1e-3,
                             batch_size=64, device=DEVICE)
    print(f"AE train loss {hist[0]:.4f} -> {hist[-1]:.4f}")
    model = model.to("cpu")

    rows = []
    for K in [3, 5, 7, 10, 15]:
        cfg = SelectionConfig(n_channels_to_select=K, normalization_method="max_per_group",
                              diversity_method="mmr")
        res = run_selection(model, {g: ds.data[g] for g in ds.groups}, cfg)
        acc = knn_loso(ds, set(res.selected))
        rows.append((K, acc, res.selected))
        print(f"K={K:2d}  LOSO-KNN acc={acc:.3f}  picks={res.selected}")

    # grouped-vs-ungrouped sanity: random-K baseline
    total = sum(ds.channels_per_group.values())
    rng = np.random.default_rng(0)
    for K in [3, 5, 10]:
        flat = [(g, c) for g in ds.groups for c in range(ds.channels_per_group[g])]
        rand = set(flat[i] for i in rng.choice(len(flat), K, replace=False))
        print(f"random K={K}: {knn_loso(ds, rand):.3f}")

    out = Path("generalization/reports/pamap2_slice.txt")
    out.write_text("\n".join(f"K={k}\tacc={a:.4f}\t{s}" for k, a, s in rows))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the slice (requires PAMAP2 data present)**

Run: `PYTORCH_ENABLE_MPS_FALLBACK=1 python experiments/general_pamap2_slice.py`
Expected: prints loaded-window count, decreasing AE loss, an accuracy-vs-K table where
AE-perturb selections beat random-K, and writes `generalization/reports/pamap2_slice.txt`.

- [ ] **Step 4: Record findings**

Append the K-vs-accuracy numbers and the grouped-vs-random gap to
`generalization/RESEARCH_LOG.md` (create it). Note whether the engine transferred and
whether selection beats random — the P2 gate.

- [ ] **Step 5: Commit**

```bash
git add experiments/general_pamap2_slice.py generalization/
git commit -m "feat(channel_select): PAMAP2 vertical slice + first HAR results"
```

---

## Self-Review

**Spec coverage:** §3.1 layout → Tasks 1,2,9,11; §3.2 engine → Tasks 3-8; §3.3 protocols → Task 1; §3.5 data → Task 2; §3.6 temporal model → Tasks 9-10; §4 PAMAP2 → Tasks 11-12; §5 benchmark (KNN+LOSO) → Task 12. **Deferred to follow-up plan (noted in spec §10):** §3.4 HSI regression (needs trained HSI model), §6 full baselines, §7 ablations beyond grouped-vs-random, §8 figures, Opportunity. The grouped-vs-ungrouped ablation appears in lightweight form (random-K comparison) in Task 12; the full ablation is P5.

**Placeholder scan:** No TBD/TODO; every code step has complete code.

**Type consistency:** `SelectionConfig` fields used in Tasks 1/8/12 match. `run_selection` returns `SelectionResult.selected` as `[(group, channel)]`, consumed consistently in Task 12. `measure_channel_influence(model, perturbed_latent, baseline_recon, weight)` signature consistent between Tasks 4 and 8. `GroupedChannelDataset(data, axis_type, labels, subject_ids)` constructor consistent across Tasks 2/11.

**Known follow-ups for the next plan:** HSI regression test; mRMR/JMI/CMIM/ReliefF baselines; Opportunity adapter; full ablation grid; F1–F5 figures; per-window feature design may move from mean+std to learned-embedding if KNN underperforms.
