"""Mechanism development (engineer E): specimen-invariance oriented additions to the ME-attention family.

Contents
--------
* ``Centered``          -- generic wrapper: per-pixel and/or per-excitation centring of (log-)intensity features, then
                           non-affine BatchNorm, then any inner module.  Makes models invariant to multiplicative
                           brightness / lamp-power drift when the input is log-intensity (``log1p=True,
                           standardize=False``).
* ``HybridConvAttn``    -- shared 1-D conv over emission per excitation -> (excitation, emission-chunk) tokens ->
                           transformer encoder (cross-excitation + cross-chunk attention).
* ``TorchEstimatorV2``  -- ``TorchEstimator`` subclass with extra augmentations and objectives that need group ids:
                           per-excitation intensity scale, spectral smoothing jitter, excitation dropout,
                           same-class cross-specimen mixup, supervised contrastive loss (cross-specimen positives),
                           conditional specimen-adversarial head (gradient reversal).
* ``ATTENTION_V2_MODELS`` -- registered ``ModelSpec`` s (filled at the bottom of this file).

Everything here works on the PREPROCESSED ``[B, F]`` tensor (spectral track); the estimator re-implements the shared
fit loop so the lead can port the useful pieces (marked ``PORT``) into ``train.py``.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as Fn
from torch import nn

from ..preprocess import PREPROC_DEFAULT, PREPROC_SPACE, make_preprocessor
from ..train import TRAIN_DEFAULT, TRAIN_SPACE, TorchEstimator, balanced_accuracy
from ..types import ModelSpec
from .attention import (NEG, Encoder, WaveEnc, _Base, _grid_gather, _head, _ranges, _to_grid, masked_mean)

V2_TRAIN_DEFAULT: dict[str, Any] = dict(
    aug_ex_scale=0.0,     # per-excitation log-uniform scale range: s ~ exp(U(-a, a)) per (sample, excitation)
    aug_smooth=0.0,       # max spectral smoothing blend a ~ U(0, aug_smooth) per sample (neighbour averaging)
    ex_drop=0.0,          # probability to zero a whole excitation block (preprocessed space) per (sample, ex)
    mixup=0.0,            # Beta(a, a) same-class cross-specimen raw mixup (0 = off)
    supcon_w=0.0, supcon_temp=0.1,
    adv_w=0.0,            # max lambda of the conditional specimen-adversarial head (0 = off)
    center="none",        # none | pixel | ex | both   (wrapper Centered)
)


# ============================================================================================ modules
class Centered(_Base):
    """Centre log-intensity features per pixel (``pixel``), per excitation block (``ex``) or both, then BatchNorm."""

    def __init__(self, inner: nn.Module, meta: dict, mode: str = "pixel"):
        super().__init__()
        if mode not in ("pixel", "ex", "both"):
            raise ValueError(mode)
        self.mode = mode
        F = int(meta["n_features"])
        ex = np.asarray(meta["ex_nm"], float)
        ids = np.unique(ex, return_inverse=True)[1]
        oh = np.zeros((F, ids.max() + 1), np.float32)
        oh[np.arange(F), ids] = 1
        self.register_buffer("oh", torch.from_numpy(oh), persistent=False)
        self.register_buffer("cnt", torch.from_numpy(oh.sum(0)), persistent=False)
        self.bn = nn.BatchNorm1d(F, affine=False)
        self.inner = inner

    def forward(self, x):
        if self.mode in ("pixel",):
            x = x - x.mean(1, keepdim=True)
        else:  # per-excitation (ex or both: block centring already removes the global mean)
            m = (x @ self.oh) / self.cnt
            x = x - m @ self.oh.T
        return self.inner(self.bn(x))

    def extra_loss(self):
        return self.inner.extra_loss() if hasattr(self.inner, "extra_loss") else super().extra_loss()


class HybridConvAttn(_Base):
    """Shared conv over emission (per excitation) -> pooled chunk tokens -> transformer (CLS)."""

    def __init__(self, meta, n_classes, dim=64, depth=2, heads=4, mlp_ratio=2, dropout=0.1, drop_path=0.0, n_freq=8,
                 conv_ch=24, kernel=5, n_chunks=6, token_drop=0.1, pool="cls"):
        super().__init__()
        g, valid, E, M = _grid_gather(meta)
        self.E, self.M, self.K, self.token_drop, self.pool_kind = E, M, int(n_chunks), float(token_drop), pool
        self.register_buffer("g", g, persistent=False)
        self.register_buffer("valid", valid.float(), persistent=False)
        r = _ranges(meta)
        emg = np.asarray(meta["emissions"], float)
        self.register_buffer("emchan", torch.as_tensor((emg - r["em"][0]) / r["em"][1], dtype=torch.float32), persistent=False)
        self.conv = nn.Sequential(
            nn.Conv1d(3, conv_ch, kernel, padding=kernel // 2), nn.GELU(),
            nn.Conv1d(conv_ch, conv_ch, kernel, padding=kernel // 2), nn.GELU())
        pm_ = np.zeros((M, self.K), np.float32)
        for k, ch in enumerate(np.array_split(np.arange(M), self.K)):
            pm_[ch, k] = 1.0 / len(ch)
        self.register_buffer("poolm", torch.from_numpy(pm_), persistent=False)
        pool = lambda a: a @ self.poolm   # noqa: E731  (mps lacks non-divisible adaptive pooling)
        self.proj = nn.Linear(conv_ch, dim)
        pm = pool(valid.float())  # [E, K]
        self.register_buffer("tok_valid", (pm > 0).reshape(E * self.K), persistent=False)
        centre = pool(torch.as_tensor(emg, dtype=torch.float32)[None]).view(self.K).numpy()
        self.wave = WaveEnc(np.repeat(np.asarray(meta["excitations"], float), self.K), np.tile(centre, E), r, dim, n_freq)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02) if pool == "cls" else None
        self.enc = Encoder(dim, depth, heads, mlp_ratio, dropout, drop_path)
        self.head = _head(dim, n_classes, dropout)

    def encode(self, x):
        B, E, M, K = x.shape[0], self.E, self.M, self.K
        grid = _to_grid(x, self.g).view(B, E, M) * self.valid
        inp = torch.stack([grid, self.valid.expand(B, E, M), self.emchan.expand(B, E, M)], 2).reshape(B * E, 3, M)
        h = self.conv(inp) @ self.poolm                              # [B*E, C, K]
        tok = self.proj(h.permute(0, 2, 1).reshape(B, E * K, -1)) + self.wave()
        kpm = (~self.tok_valid).unsqueeze(0).expand(B, E * K)
        if self.training and self.token_drop > 0:
            kpm = kpm | (torch.rand(B, E * K, device=x.device) < self.token_drop)
        if self.cls is not None:
            tok = torch.cat([self.cls.expand(B, -1, -1), tok], 1)
            kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), kpm], 1)
        h, _ = self.enc(tok, None, kpm, False)
        return h[:, 0] if self.cls is not None else masked_mean(h, (~kpm).float())[0]

    def forward(self, x):
        return self.head(self.encode(x))


_HYB_DEFAULT = dict(dim=64, depth=2, heads=4, mlp_ratio=2, dropout=0.1, drop_path=0.0, n_freq=8, conv_ch=24, kernel=5,
                    n_chunks=6, token_drop=0.1, pool="cls")


def hybrid_factory(meta, cfg):
    kw = {k: cfg[k] for k in _HYB_DEFAULT if k in cfg}
    return HybridConvAttn(meta, int(meta["n_classes"]), **{**_HYB_DEFAULT, **kw})


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


# ============================================================================================ estimator
def _ordered_blocks(meta) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ex = np.asarray(meta["ex_nm"], float)
    ids = np.unique(ex, return_inverse=True)[1]
    F = len(ex)
    order = np.lexsort((np.asarray(meta["em_nm"], float), ex))
    pos = np.empty(F, int)
    pos[order] = np.arange(F)
    prev, nxt = np.arange(F), np.arange(F)
    for i in range(F):  # neighbours within the same excitation, by emission order
        p = pos[i]
        if p > 0 and ids[order[p - 1]] == ids[i]:
            prev[i] = order[p - 1]
        if p < F - 1 and ids[order[p + 1]] == ids[i]:
            nxt[i] = order[p + 1]
    return ids, prev, nxt


class TorchEstimatorV2(TorchEstimator):
    """TorchEstimator + specimen-invariance augmentations/objectives.  ``cfg['fixed_epochs']=True`` disables early
    stopping (used when training on a full outer-train set without a validation set)."""

    def __init__(self, module_factory, cfg, meta, seed, device, input_kind="spectral"):
        super().__init__(module_factory, {**V2_TRAIN_DEFAULT, **cfg}, meta, seed, device, input_kind)
        self.ex_ids, self.nb_prev, self.nb_next = _ordered_blocks(meta)
        self.n_ex = int(self.ex_ids.max() + 1)

    # -- augmentations on raw ----------------------------------------------------------------------------------
    def _aug_raw_v2(self, xr, rng):  # PORT
        c = self.cfg
        if "scale" in c["aug"]:
            s = np.exp(rng.uniform(math.log(0.7), math.log(1.4), size=(len(xr), 1)))
            xr = xr * s.astype(np.float32)
        if c["aug_ex_scale"] > 0:
            u = rng.uniform(-c["aug_ex_scale"], c["aug_ex_scale"], size=(len(xr), self.n_ex))
            xr = xr * np.exp(u)[:, self.ex_ids].astype(np.float32)
        if c["aug_smooth"] > 0:
            a = rng.uniform(0, c["aug_smooth"], size=(len(xr), 1)).astype(np.float32)
            xr = (1 - a) * xr + a * 0.5 * (xr[:, self.nb_prev] + xr[:, self.nb_next])
        return xr.astype(np.float32, copy=False)

    def _aug_pre_v2(self, xt, rng):  # PORT
        xt = self._augment_noise(xt, rng)
        p = self.cfg["ex_drop"]
        if p > 0:
            drop = rng.random((len(xt), self.n_ex)) < p
            drop[drop.all(1), 0] = False
            xt = xt * (~drop[:, self.ex_ids]).astype(np.float32)
        return xt

    # -- helpers -----------------------------------------------------------------------------------------------
    def _embed_hook(self, C):
        lin = [m for m in self.module.modules() if isinstance(m, nn.Linear) and m.out_features == C][-1]
        self._emb = None

        def pre(_m, inp):
            self._emb = inp[0]
        lin.register_forward_pre_hook(pre)

    @staticmethod
    def _supcon(z, y, g, temp, cross):
        z = Fn.normalize(z, dim=1)
        sim = (z @ z.T) / temp
        n = len(z)
        eye = torch.eye(n, dtype=torch.bool, device=z.device)
        same = (y[:, None] == y[None, :]) & ~eye
        pos = same & (g[:, None] != g[None, :]) if cross else same
        if cross:  # fall back to any same-class positive for anchors without a cross-specimen one
            none = pos.sum(1) == 0
            pos = torch.where(none[:, None], same, pos)
        sim = sim.masked_fill(eye, -1e4)
        logp = sim - torch.logsumexp(sim, 1, keepdim=True)
        npos = pos.sum(1)
        ok = npos > 0
        if not ok.any():
            return z.sum() * 0
        return -((logp * pos).sum(1)[ok] / npos[ok]).mean()

    def fit(self, X, y, groups, X_val=None, y_val=None):
        cfg = self.cfg
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        groups = np.asarray(groups)
        C = int(self.meta["n_classes"])
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.pre = make_preprocessor(cfg, self.meta).fit(X)
        mod = self.module_factory(self.meta, cfg)
        if cfg["center"] != "none":
            mod = Centered(mod, self.meta, cfg["center"])
        self.module = mod.to(self.device)
        uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
        S = len(uniq)
        spec_cls = np.array([np.bincount(y[inv == s]).argmax() for s in range(S)])
        p = (1.0 / counts[inv]).astype(np.float64)
        p /= p.sum()
        epoch_size = int(cfg["samples_per_specimen"]) * S
        bs = int(cfg["batch_size"])
        steps_per_epoch = max(1, math.ceil(epoch_size / bs))
        E = int(cfg["epochs"])
        total, warm = E * steps_per_epoch, int(cfg["warmup_epochs"]) * steps_per_epoch

        use_adv = cfg["adv_w"] > 0 and (np.bincount(spec_cls, minlength=C) >= 2).any()
        use_sc = cfg["supcon_w"] > 0
        self.degenerate_ = bool((cfg["adv_w"] > 0 and not use_adv) or
                                (cfg["mixup"] > 0 and (np.bincount(spec_cls, minlength=C) < 2).all()))
        if use_adv or use_sc:
            self._embed_hook(C)
        params = list(self.module.parameters())
        if use_adv:
            with torch.no_grad():
                self.module.eval()
                emb_dim = None
                self.module(torch.zeros(2, int(self.meta["n_features"]), device=self.device))
                emb_dim = self._emb.shape[1]
            self.adv = nn.Sequential(nn.Linear(emb_dim, 64), nn.GELU(), nn.Linear(64, S)).to(self.device)
            params += list(self.adv.parameters())
            spec_cls_t = torch.from_numpy(spec_cls).to(self.device)
            adv_mask = torch.zeros(C, S, device=self.device)
            adv_mask[spec_cls_t[None, :] != torch.arange(C, device=self.device)[:, None]] = NEG
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

        def lr_lambda(step):
            if step < warm:
                return (step + 1) / max(1, warm)
            return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        crit = nn.CrossEntropyLoss(label_smoothing=float(cfg["label_smoothing"]))

        any_aug = (cfg["aug"] != "none" or cfg["aug_ex_scale"] > 0 or cfg["aug_smooth"] > 0 or cfg["ex_drop"] > 0
                   or cfg["mixup"] > 0)
        pre_train = self._to_dev(self._transform_batched(X)) if not any_aug else None
        yt_all = torch.from_numpy(y).to(self.device)
        inv_t = torch.from_numpy(inv).to(self.device)
        by_spec = [np.flatnonzero(inv == s) for s in range(S)]
        cls_specs = [np.flatnonzero(spec_cls == c) for c in range(C)]

        have_val = (X_val is not None and y_val is not None and len(X_val) > 0 and not cfg.get("fixed_epochs"))
        Xv = self._to_dev(self._transform_batched(np.asarray(X_val, np.float32))) if have_val else None
        y_val = np.asarray(y_val, dtype=np.int64) if have_val else None

        best_score, best_state, self.best_epoch_, bad = -1.0, None, E, 0
        self.history_ = []
        step = 0
        for ep in range(1, E + 1):
            self.module.train()
            idx_all = rng.choice(len(X), size=epoch_size, replace=True, p=p)
            tl, tc, tn = 0.0, 0, 0
            for s in range(0, epoch_size, bs):
                idx = idx_all[s:s + bs]
                it = torch.from_numpy(idx).to(self.device)
                if pre_train is not None:
                    xb = pre_train[it]
                else:
                    xr = X[idx]
                    if cfg["mixup"] > 0:  # PORT: same-class, different-specimen raw mixup
                        part = np.empty_like(idx)
                        for sid in np.unique(inv[idx]):
                            m = inv[idx] == sid
                            others = [t for t in cls_specs[spec_cls[sid]] if t != sid] or [sid]
                            ts = rng.choice(others, size=int(m.sum()))
                            part[m] = [rng.choice(by_spec[t]) for t in ts]
                        lam = rng.beta(cfg["mixup"], cfg["mixup"], size=(len(idx), 1)).astype(np.float32)
                        xr = lam * xr + (1 - lam) * X[part]
                    xr = self._aug_raw_v2(xr, rng)
                    xb = self._to_dev(self._aug_pre_v2(self.pre.transform(xr), rng))
                yb = yt_all[it]
                logits = self.module(xb)
                loss = crit(logits, yb)
                if hasattr(self.module, "extra_loss"):
                    loss = loss + self.module.extra_loss()
                if use_sc:
                    loss = loss + cfg["supcon_w"] * self._supcon(self._emb, yb, inv_t[it], cfg["supcon_temp"], True)
                if use_adv:
                    prog = step / max(1, total)
                    lam_a = cfg["adv_w"] * (2.0 / (1.0 + math.exp(-10 * prog)) - 1.0)
                    al = self.adv(_GRL.apply(self._emb, lam_a)) + adv_mask[yb]
                    loss = loss + Fn.cross_entropy(al, inv_t[it])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg["grad_clip"]:
                    nn.utils.clip_grad_norm_(params, float(cfg["grad_clip"]))
                opt.step()
                sched.step()
                step += 1
                tl += float(loss.detach()) * len(idx)
                tc += int((logits.argmax(1) == yb).sum())
                tn += len(idx)
            rec: dict[str, Any] = dict(epoch=ep, train_loss=tl / tn, train_acc=tc / tn)
            if have_val:
                pred = self._predict_logits(Xv).argmax(1)
                rec["val_bacc"] = balanced_accuracy(y_val, pred, C)
                if rec["val_bacc"] > best_score:
                    best_score, self.best_epoch_, bad = rec["val_bacc"], ep, 0
                    best_state = copy.deepcopy(self.module.state_dict())
                else:
                    bad += 1
            self.history_.append(rec)
            if have_val and bad >= int(cfg["patience"]):
                break
        if best_state is not None:
            self.module.load_state_dict(best_state)
        return self


# ============================================================================================ registry
def _build(factory):
    """All sweep models use the ONE shared training loop (train.TorchEstimator); TorchEstimatorV2 is dev-only."""
    return lambda cfg, meta, seed, device: TorchEstimator(factory, cfg, meta, seed, device)


def _spec_v2(name, factory, model_space, model_default, desc) -> ModelSpec:
    # frozen shared keys (identical for every neural model): lr, aug, ex_drop, norm, log1p
    space = {**model_space, **TRAIN_SPACE, "norm": PREPROC_SPACE["norm"], "log1p": PREPROC_SPACE["log1p"],
             "norm_scope": PREPROC_SPACE["norm_scope"]}
    default = {**PREPROC_DEFAULT, **TRAIN_DEFAULT, **model_default}
    return ModelSpec(name=name, track="spectral", family="ours", build=_build(factory), space=space, default=default,
                     description=desc)


def _ex_tokens_factory(meta, cfg):
    from .attention import ex_tokens_factory
    return ex_tokens_factory(meta, cfg)


class _LinearShortcut(nn.Module):
    """logits = inner(x) + Linear(F, C)(x); shortcut zero-initialised so training starts identical to inner."""

    def __init__(self, inner: nn.Module, n_features: int, n_classes: int):
        super().__init__()
        self.inner = inner
        self.shortcut = nn.Linear(n_features, n_classes)
        nn.init.zeros_(self.shortcut.weight)
        nn.init.zeros_(self.shortcut.bias)

    def forward(self, x):
        return self.inner(x) + self.shortcut(x)


def _ex_tokens_lin_factory(meta, cfg):
    return _LinearShortcut(_ex_tokens_factory(meta, cfg), int(meta["n_features"]), int(meta["n_classes"]))


_EX_TOKENS_V2_SPACE = dict(dim=[32, 64], depth=[2, 3], sub_tokens=[1, 2], dropout=[0.1, 0.3])
_EX_TOKENS_V2_DEFAULT = dict(dim=64, depth=3, heads=4, mlp_ratio=2, dropout=0.1, drop_path=0.0, n_freq=8,
                             ex_embed="linear", sub_tokens=1, pool="cls", token_drop=0.1)

ATTENTION_V2_MODELS: dict[str, ModelSpec] = {s.name: s for s in [
    _spec_v2("hybrid_conv_attn", hybrid_factory,
             dict(dim=[32, 64], depth=[1, 2, 3], n_chunks=[4, 6, 8], dropout=[0.1, 0.3]),
             dict(_HYB_DEFAULT, dim=64, depth=2, n_chunks=6),
             "shared emission conv per excitation -> (excitation, chunk) tokens -> transformer (CLS)"),
    _spec_v2("ex_tokens_v2", _ex_tokens_factory,
             _EX_TOKENS_V2_SPACE, _EX_TOKENS_V2_DEFAULT,
             "excitation tokens (dev: best existing attention variant) trained with the dev-selected preprocessing/aug"),
    _spec_v2("ex_tokens_lin", _ex_tokens_lin_factory, _EX_TOKENS_V2_SPACE, _EX_TOKENS_V2_DEFAULT,
             "ex_tokens_v2 plus a zero-initialised linear shortcut on the flat input added to the logits"),
]}
