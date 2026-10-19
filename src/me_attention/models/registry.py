"""Model registry: spectral baselines + our attention family + spatial baselines."""
from __future__ import annotations

from .attention import ATTENTION_MODELS
from .attention_v2 import ATTENTION_V2_MODELS
from .baselines_spatial import PATCH_SIZES, SPATIAL_BASELINES
from .baselines_spectral import SPECTRAL_BASELINES
from ..types import ModelSpec


def _with_patch(spec: ModelSpec) -> ModelSpec:
    """Every spatial model carries the patch size P in its search space (harness builds patches per P)."""
    if spec.track == "spatial" and "P" not in spec.space:
        spec.space = {"P": list(PATCH_SIZES), **spec.space}
        spec.default = {"P": 5, **spec.default}
    return spec


MODELS: dict[str, ModelSpec] = {n: _with_patch(s) for n, s in
                                {**SPECTRAL_BASELINES, **ATTENTION_MODELS, **ATTENTION_V2_MODELS, **SPATIAL_BASELINES}.items()}


def models_for(track: str, family: str | None = None) -> list[str]:
    return [n for n, s in MODELS.items() if s.track == track and (family is None or s.family == family)]


# Frozen model lists of the final sweep (docs/me_attention/FROZEN_SPACES.md).
SWEEP_MODELS: dict[str, list[str]] = {
    "spectral_baselines": ["lr", "rbfsvm", "rf", "hgb", "knn", "mlp", "cnn1d", "spectralformer"],
    "spectral_ours": ["smartattention_v0", "ex_tokens_v2", "hybrid_conv_attn", "meas_tokens"],
    "spatial_baselines": ["patchmean_lr", "patchmean_rf", "cnn2d", "spectralformer_patch", "ssftt"],
    "spatial_ours": ["spatial_attn"],
}
SWEEP_ALL: list[str] = [m for v in SWEEP_MODELS.values() for m in v]
for _n in SWEEP_ALL:
    assert _n in MODELS, _n
