"""Spatial-track baselines (pixel -> class from a P x P patch around it; X = [n, P, P, F]).

Patch-mean classical models (centre spectrum + mean of the valid patch pixels' spectra), 2D/3D CNNs, HybridSN
(Roy et al. 2020), SpectralFormer patch-wise (Hong et al. 2021) and SSFTT (Sun et al. 2022), all scaled down to
our F (~150-210 bands) and P in {5, 7}. Every model has the patch size ``P`` in its search space (the harness builds
patches per P), shares the preprocessing and training space with ours. Features are always ordered by (ex, em).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from ..preprocess import PREPROC_DEFAULT, PREPROC_SPACE
from ..train import TRAIN_DEFAULT, TRAIN_SPACE, SklearnEstimator, TorchEstimator
from ..types import ModelSpec
from .baselines_spectral import _Block, _feature_order, _lr, _rbfsvm, _rf

PATCH_SIZES = [5, 7]


# ------------------------------------------------------------------ patch-mean classical
class PatchMeanEstimator(SklearnEstimator):
    """Features = [centre pixel spectrum, mean spectrum of the valid (non-zero) patch pixels] on preprocessed patches."""

    def _flat(self, A: np.ndarray) -> np.ndarray:
        if A.ndim == 2:
            return A
        n, P = A.shape[0], A.shape[1]
        c = P // 2
        valid = np.any(A != 0, axis=-1)
        cnt = np.maximum(valid.sum((1, 2)), 1)[:, None].astype(np.float32)
        return np.concatenate([A[:, c, c, :], A.sum((1, 2)) / cnt], axis=1)


# ------------------------------------------------------------------ neural modules (input [B,P,P,F], preprocessed)
class _Sorted(nn.Module):
    """Mixin base: reorder features by (ex, em) and move to channel-first [B,F,P,P]."""

    def __init__(self, meta):
        super().__init__()
        self.register_buffer("order", torch.as_tensor(_feature_order(meta), dtype=torch.long))

    def chw(self, x):
        return x[..., self.order].permute(0, 3, 1, 2).contiguous()


class CNN2D(_Sorted):
    def __init__(self, meta, cfg):
        super().__init__(meta)
        F, C, ch, L, d = meta["n_features"], meta["n_classes"], cfg["channels"], cfg["n_layers"], cfg["dropout"]
        layers, i = [], F
        for _ in range(L):
            layers += [nn.Conv2d(i, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.GELU()]
            i = ch
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(d), nn.Linear(2 * ch, C))

    def forward(self, x):
        h = self.conv(self.chw(x))
        return self.head(torch.cat([h.mean((2, 3)), h[:, :, h.shape[2] // 2, h.shape[3] // 2]], 1))


class CNN3D(_Sorted):
    """Conv3d over (F-sorted-by-(ex,em), P, P), spectral stride 3 per layer; global average pooling."""

    def __init__(self, meta, cfg):
        super().__init__(meta)
        C, ch, L, d = meta["n_classes"], cfg["channels"], cfg["n_layers"], cfg["dropout"]
        layers, i = [], 1
        for l in range(L):
            # spectral stride 3 instead of max-pool (aten::max_pool3d is not implemented on MPS)
            layers += [nn.Conv3d(i, ch, (5, 3, 3), stride=(3, 1, 1), padding=(2, 1, 1)), nn.BatchNorm3d(ch), nn.GELU()]
            i = ch
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(d), nn.Linear(ch, C))

    def forward(self, x):
        h = self.conv(self.chw(x).unsqueeze(1))
        return self.head(h.mean((2, 3, 4)))


class HybridSN(_Sorted):
    """Roy et al. 2020: 3 x Conv3d (spectral kernels 7/5/3, here with spectral stride 2 to fit our F), reshape to 2-D
    maps, Conv2d, FC-dropout-FC. Spatial dims are padded so P is preserved."""

    def __init__(self, meta, cfg):
        super().__init__(meta)
        F, C, P, c, d = meta["n_features"], meta["n_classes"], int(cfg["P"]), cfg["channels"], cfg["dropout"]
        s = (2, 1, 1)
        self.c3 = nn.Sequential(
            nn.Conv3d(1, c, (7, 3, 3), stride=(4, 1, 1), padding=(3, 1, 1)), nn.BatchNorm3d(c), nn.ReLU(),
            nn.Conv3d(c, 2 * c, (5, 3, 3), stride=s, padding=(2, 1, 1)), nn.BatchNorm3d(2 * c), nn.ReLU(),
            nn.Conv3d(2 * c, 4 * c, (3, 3, 3), stride=s, padding=(1, 1, 1)), nn.BatchNorm3d(4 * c), nn.ReLU())
        with torch.no_grad():
            f = self.c3(torch.zeros(1, 1, F, P, P)).shape
        self.c2 = nn.Sequential(nn.Conv2d(f[1] * f[2], 4 * c, 3, padding=1), nn.BatchNorm2d(4 * c), nn.ReLU())
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(4 * c * P * P, 128), nn.ReLU(), nn.Dropout(d),
                                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(d), nn.Linear(64, C))

    def forward(self, x):
        h = self.c3(self.chw(x).unsqueeze(1))
        return self.fc(self.c2(h.flatten(1, 2)))


class SpectralFormerPatch(_Sorted):
    """SpectralFormer patch-wise (Hong et al., TGRS 2021): band i's token embeds the flattened P x P neighbourhood of the
    `group` bands centred on i (GSE); CLS + learned positions; pre-norm blocks with CAF cross-layer fusion."""

    def __init__(self, meta, cfg, group: int = 3):
        super().__init__(meta)
        F, C, P, dim, depth, drop = meta["n_features"], meta["n_classes"], int(cfg["P"]), cfg["dim"], cfg["depth"], cfg["dropout"]
        self.group = group
        self.embed = nn.Linear(group * P * P, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, F + 1, dim) * 0.02)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([_Block(dim, cfg.get("heads", 4), 2, drop) for _ in range(depth)])
        self.skipcat = nn.ModuleList([nn.Conv2d(F + 1, F + 1, (1, 2), 1, 0) for _ in range(depth - 1)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, C)

    def forward(self, x):
        B = len(x)
        z = x[..., self.order].flatten(1, 2)                               # [B,P*P,F]
        g = nn.functional.pad(z, (self.group // 2, self.group // 2)).unfold(2, self.group, 1)  # [B,P*P,F,g]
        g = g.permute(0, 2, 3, 1).reshape(B, g.shape[2], -1)               # [B,F,g*P*P]
        t = self.drop(torch.cat([self.cls.expand(B, -1, -1), self.embed(g)], 1) + self.pos)
        prev = []
        for l, blk in enumerate(self.blocks):
            prev.append(t)
            if l > 0:
                t = self.skipcat[l - 1](torch.stack([t, prev[l - 1]], dim=3)).squeeze(3)
            t = blk(t)
        return self.head(self.norm(t)[:, 0])


class SSFTT(_Sorted):
    """Sun et al., TGRS 2022: Conv3d -> Conv2d spectral-spatial features -> Gaussian-weighted (softmax) feature
    tokenizer -> CLS + positions -> transformer -> CLS head. Spatial dims padded; Conv3d spectral stride 2."""

    def __init__(self, meta, cfg):
        super().__init__(meta)
        F, C, P, dim, drop = meta["n_features"], meta["n_classes"], int(cfg["P"]), cfg["dim"], cfg["dropout"]
        c3, L = cfg["channels"], cfg["n_tokens"]
        self.c3 = nn.Sequential(nn.Conv3d(1, c3, (3, 3, 3), stride=(2, 1, 1), padding=1), nn.BatchNorm3d(c3), nn.ReLU())
        with torch.no_grad():
            f = self.c3(torch.zeros(1, 1, F, P, P)).shape
        self.c2 = nn.Sequential(nn.Conv2d(f[1] * f[2], dim, 3, padding=1), nn.BatchNorm2d(dim), nn.ReLU())
        self.wa, self.wv = nn.Linear(dim, L, bias=False), nn.Linear(dim, dim, bias=False)
        nn.init.xavier_normal_(self.wa.weight)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, L + 1, dim) * 0.02)
        self.blocks = nn.Sequential(*[_Block(dim, cfg.get("heads", 4), 2, drop) for _ in range(cfg["depth"])])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, C)

    def forward(self, x):
        h = self.c2(self.c3(self.chw(x).unsqueeze(1)).flatten(1, 2)).flatten(2).transpose(1, 2)  # [B,P*P,dim]
        a = torch.softmax(self.wa(h), dim=1)                                  # [B,N,L] weights over positions
        tok = a.transpose(1, 2) @ self.wv(h)                                  # [B,L,dim]
        t = torch.cat([self.cls.expand(len(x), -1, -1), tok], 1) + self.pos
        return self.head(self.norm(self.blocks(t))[:, 0])


# ------------------------------------------------------------------ registry
def _sk(make, sample_weight=True):
    return lambda cfg, meta, seed, device: PatchMeanEstimator(make, cfg, meta, seed, sample_weight=sample_weight)


def _nn(module_cls):
    return lambda cfg, meta, seed, device: TorchEstimator(lambda m, c: module_cls(m, c), cfg, meta, seed, device,
                                                          input_kind="spatial")


def _spec(name, build, model_space, model_default, desc, neural=False) -> ModelSpec:
    space = {**PREPROC_SPACE, "P": list(PATCH_SIZES), **model_space}
    default = {**PREPROC_DEFAULT, "P": 5, **model_default}
    if neural:
        space = {**space, **TRAIN_SPACE}
        default = {**default, **TRAIN_DEFAULT}
    return ModelSpec(name=name, track="spatial", family="baseline", build=build, space=space, default=default,
                     description=desc)


_TREE = dict(max_depth=[None, 20, 40], max_features=["sqrt", 0.2, 0.5])

SPATIAL_BASELINES: dict[str, ModelSpec] = {s.name: s for s in [
    _spec("patchmean_lr", _sk(_lr), dict(C=[0.01, 0.1, 1.0, 10.0, 100.0]), dict(C=1.0),
          "LR on [centre, patch-mean] spectra"),
    _spec("patchmean_rbfsvm", _sk(_rbfsvm), dict(C=[0.3, 1.0, 3.0, 10.0, 30.0], gamma=["scale", 0.1, 0.03, 0.01],
                                                 max_train=[20000]),
          dict(C=3.0, gamma="scale", max_train=20000), "RBF SVM on [centre, patch-mean] spectra"),
    _spec("patchmean_rf", _sk(_rf), dict(n_estimators=[300], **_TREE),
          dict(n_estimators=300, max_depth=None, max_features="sqrt"), "RF on [centre, patch-mean] spectra"),
    _spec("cnn2d", _nn(CNN2D), dict(channels=[32, 64], n_layers=[2, 3], dropout=[0.0, 0.2]),
          dict(channels=32, n_layers=2, dropout=0.1), "2-D CNN, bands as channels, avg+centre pooling", neural=True),
    _spec("cnn3d", _nn(CNN3D), dict(channels=[4, 8, 16], n_layers=[2, 3], dropout=[0.0, 0.2]),
          dict(channels=8, n_layers=2, dropout=0.1), "3-D CNN over (F sorted by ex,em) x P x P", neural=True),
    _spec("hybridsn", _nn(HybridSN), dict(channels=[2, 4], dropout=[0.2, 0.4]),
          dict(channels=2, dropout=0.3), "HybridSN (Roy 2020), scaled down", neural=True),
    _spec("spectralformer_patch", _nn(SpectralFormerPatch), dict(dim=[32, 64], depth=[2, 3], dropout=[0.0, 0.1]),
          dict(dim=32, depth=2, heads=4, dropout=0.1), "SpectralFormer patch-wise GSE + CAF (Hong 2021), scaled down",
          neural=True),
    _spec("ssftt", _nn(SSFTT), dict(channels=[4, 8], dim=[32, 64], depth=[1, 2], n_tokens=[4, 8]),
          dict(channels=4, dim=32, depth=1, heads=4, n_tokens=4, dropout=0.1),
          "SSFTT (Sun 2022): 3D+2D conv, Gaussian tokenizer, transformer; scaled down", neural=True),
]}
