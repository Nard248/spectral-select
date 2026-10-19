"""Spectral-track baselines (pixel -> class from its own (ex, em) measurements).

Classical: lr, linsvm, rbfsvm, rf, et, hgb, knn (SklearnEstimator, specimen-balanced sample weights where supported).
Neural: mlp, cnn1d, spectralformer (Hong et al. 2021, pixel-wise: GSE + CAF), bandvit (one token per feature).
Neural baselines share ``TorchEstimator`` and ``TRAIN_SPACE`` with our models.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from ..preprocess import PREPROC_DEFAULT, PREPROC_SPACE
from ..train import TRAIN_DEFAULT, TRAIN_SPACE, SklearnEstimator, TorchEstimator
from ..types import ModelSpec

# ------------------------------------------------------------------ classical
def _lr(cfg, meta, seed):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=cfg["C"], max_iter=500, random_state=seed)

def _linsvm(cfg, meta, seed):
    from sklearn.svm import LinearSVC
    return LinearSVC(C=cfg["C"], dual="auto", max_iter=3000, random_state=seed)

def _rbfsvm(cfg, meta, seed):
    from sklearn.svm import SVC
    return SVC(C=cfg["C"], gamma=cfg["gamma"], kernel="rbf", decision_function_shape="ovr", random_state=seed)

def _rf(cfg, meta, seed):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                                  max_features=cfg["max_features"], n_jobs=cfg.get("n_jobs", 4), random_state=seed)

def _et(cfg, meta, seed):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                                max_features=cfg["max_features"], n_jobs=cfg.get("n_jobs", 4), random_state=seed)

def _hgb(cfg, meta, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(learning_rate=cfg["learning_rate"], max_leaf_nodes=cfg["max_leaf_nodes"],
                                          l2_regularization=cfg["l2_regularization"], max_iter=cfg["max_iter"],
                                          early_stopping=False, random_state=seed)

def _knn(cfg, meta, seed):
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=cfg["k"], weights=cfg["weights"], n_jobs=cfg.get("n_jobs", 4))

# ------------------------------------------------------------------ neural modules
def _feature_order(meta) -> np.ndarray:
    return np.lexsort((np.asarray(meta["em_nm"]), np.asarray(meta["ex_nm"])))  # sort by (ex, em)


class MLP(nn.Module):
    def __init__(self, meta, cfg):
        super().__init__()
        F, C, w, L, d = meta["n_features"], meta["n_classes"], cfg["width"], cfg["n_layers"], cfg["dropout"]
        layers, i = [], F
        for _ in range(L):
            layers += [nn.Linear(i, w), nn.LayerNorm(w), nn.GELU(), nn.Dropout(d)]
            i = w
        self.net = nn.Sequential(*layers, nn.Linear(i, C))

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    """Conv1d stack over the feature axis sorted by (ex, em); global average + max pooling head."""

    def __init__(self, meta, cfg):
        super().__init__()
        C, ch, L, d = meta["n_classes"], cfg["channels"], cfg["n_layers"], cfg["dropout"]
        self.register_buffer("order", torch.as_tensor(_feature_order(meta), dtype=torch.long))
        layers, i = [], 1
        for l in range(L):
            layers += [nn.Conv1d(i, ch, cfg["kernel"], padding=cfg["kernel"] // 2), nn.BatchNorm1d(ch), nn.GELU()]
            if l < L - 1:
                layers.append(nn.MaxPool1d(2))
            i = ch
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(d), nn.Linear(2 * ch, C))

    def forward(self, x):
        h = self.conv(x[:, self.order].unsqueeze(1))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))


class _Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, drop):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(drop),
                                nn.Linear(dim * mlp_ratio, dim), nn.Dropout(drop))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class SpectralFormer(nn.Module):
    """Pixel-wise SpectralFormer (Hong et al., TGRS 2021).

    GSE: token i embeds the group of `group`=3 neighbouring bands centred on band i (zero padded) with a shared
    linear map. CLS token, learned positional embedding, pre-norm transformer blocks, CAF: before block l>=2 the
    current input is fused with the input of block l-1 via Conv2d(T,T,[1,2]) over the stacked pair (as in the
    reference code). Bands are taken in (ex, em) order.
    """

    def __init__(self, meta, cfg, group: int = 3):
        super().__init__()
        F, C, dim, depth, drop = meta["n_features"], meta["n_classes"], cfg["dim"], cfg["depth"], cfg["dropout"]
        self.group = group
        self.register_buffer("order", torch.as_tensor(_feature_order(meta), dtype=torch.long))
        self.embed = nn.Linear(group, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, F + 1, dim) * 0.02)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([_Block(dim, cfg.get("heads", 4), 2, drop) for _ in range(depth)])
        self.skipcat = nn.ModuleList([nn.Conv2d(F + 1, F + 1, (1, 2), 1, 0) for _ in range(depth - 1)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, C)

    def forward(self, x):
        x = x[:, self.order]
        g = nn.functional.pad(x, (self.group // 2, self.group // 2)).unfold(1, self.group, 1)  # [B,F,g]
        t = torch.cat([self.cls.expand(len(x), -1, -1), self.embed(g)], 1) + self.pos
        t = self.drop(t)
        prev = []
        for l, blk in enumerate(self.blocks):
            prev.append(t)
            if l > 0:
                t = self.skipcat[l - 1](torch.stack([t, prev[l - 1]], dim=3)).squeeze(3)
            t = blk(t)
        return self.head(self.norm(t)[:, 0])


class BandViT(nn.Module):
    """One token per feature: value * w_v + b_v + learned per-feature embedding; CLS; pre-norm encoder."""

    def __init__(self, meta, cfg):
        super().__init__()
        F, C, dim, depth, drop = meta["n_features"], meta["n_classes"], cfg["dim"], cfg["depth"], cfg["dropout"]
        self.wv = nn.Parameter(torch.randn(dim) * 0.1)
        self.bv = nn.Parameter(torch.zeros(dim))
        self.feat = nn.Parameter(torch.randn(1, F, dim) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, cfg.get("heads", 4), dim * 2, drop, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, C)

    def forward(self, x):
        t = x.unsqueeze(-1) * self.wv + self.bv + self.feat
        t = torch.cat([self.cls.expand(len(x), -1, -1), t], 1)
        return self.head(self.norm(self.enc(t))[:, 0])


# ------------------------------------------------------------------ registry
def _sk(make, sample_weight=True):
    return lambda cfg, meta, seed, device: SklearnEstimator(make, cfg, meta, seed, sample_weight=sample_weight)

def _nn(module_cls):
    return lambda cfg, meta, seed, device: TorchEstimator(lambda m, c: module_cls(m, c), cfg, meta, seed, device)

def _spec(name, build, model_space, model_default, desc, neural=False) -> ModelSpec:
    space = {**PREPROC_SPACE, **model_space}
    default = {**PREPROC_DEFAULT, **model_default}
    if neural:
        space = {**space, **TRAIN_SPACE}
        default = {**default, **TRAIN_DEFAULT}
    return ModelSpec(name=name, track="spectral", family="baseline", build=build, space=space, default=default,
                     description=desc)

_TREE = dict(max_depth=[None, 20, 40], max_features=["sqrt", 0.2, 0.5])
_TREE_D = dict(n_estimators=300, max_depth=None, max_features="sqrt")

SPECTRAL_BASELINES: dict[str, ModelSpec] = {s.name: s for s in [
    _spec("lr", _sk(_lr), dict(C=[0.01, 0.1, 1.0, 10.0, 100.0]), dict(C=1.0), "Logistic regression"),
    _spec("linsvm", _sk(_linsvm), dict(C=[0.001, 0.01, 0.1, 1.0, 10.0]), dict(C=0.1),
          "Linear SVM (proba = softmax of decision function)"),
    _spec("rbfsvm", _sk(_rbfsvm), dict(C=[0.3, 1.0, 3.0, 10.0, 30.0], gamma=["scale", 0.1, 0.03, 0.01, 0.003],
                                       max_train=[20000]),
          dict(C=3.0, gamma="scale", max_train=20000), "RBF SVM (proba = softmax of decision function)"),
    _spec("rf", _sk(_rf), dict(n_estimators=[300], **_TREE), dict(_TREE_D), "Random forest"),
    _spec("et", _sk(_et), dict(n_estimators=[300], **_TREE), dict(_TREE_D), "Extra trees"),
    _spec("hgb", _sk(_hgb), dict(learning_rate=[0.03, 0.1, 0.3], max_leaf_nodes=[15, 31, 63],
                                 l2_regularization=[0.0, 1.0, 10.0], max_iter=[100]),
          dict(learning_rate=0.1, max_leaf_nodes=31, l2_regularization=0.0, max_iter=100), "HistGradientBoosting"),
    _spec("knn", _sk(_knn, sample_weight=False), dict(k=[1, 3, 5, 9, 15, 25], weights=["uniform", "distance"],
                                                      max_train=[20000]),
          dict(k=5, weights="uniform", max_train=20000), "k-nearest neighbours (specimen-balanced subsample)"),
    _spec("mlp", _nn(MLP), dict(width=[64, 128, 256], n_layers=[2, 3], dropout=[0.0, 0.1, 0.3]),
          dict(width=128, n_layers=2, dropout=0.1), "MLP", neural=True),
    _spec("cnn1d", _nn(CNN1D), dict(channels=[32, 64], n_layers=[2, 3, 4], kernel=[3, 5], dropout=[0.0, 0.2]),
          dict(channels=32, n_layers=3, kernel=5, dropout=0.1), "1D CNN over (ex, em)-sorted features", neural=True),
    _spec("spectralformer", _nn(SpectralFormer), dict(dim=[32, 64], depth=[3, 5]),
          dict(dim=64, depth=5, heads=4, dropout=0.1), "SpectralFormer pixel-wise (GSE+CAF)", neural=True),
    _spec("bandvit", _nn(BandViT), dict(dim=[32, 64], depth=[2, 3, 4], dropout=[0.0, 0.1]),
          dict(dim=64, depth=3, heads=4, dropout=0.1), "Vanilla band-token transformer", neural=True),
]}
