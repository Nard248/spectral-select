"""Shared training code for all ME-attention models (baselines and ours).

``TorchEstimator``: AdamW + linear warmup / cosine decay, specimen-balanced sampling with replacement, label
smoothing, intensity-scale / noise augmentation, early stopping on inner-val balanced accuracy (best weights
restored). ``SklearnEstimator``: preprocessing + sklearn model with specimen-balanced sample weights.

Data flow (torch, ALL on-device after a one-off setup; identical semantics for every neural model):
  * per-pixel intensity-scale aug (``aug`` contains "scale", s ~ logU(0.7, 1.4)) is applied to the RAW input before
    preprocessing.  Every ``norm`` in {l2, area, max} (global or per-excitation) is invariant to a per-pixel scalar, and
    log1p/standardisation come after the norm, so for norm != "none" the scale aug is an exact no-op and is skipped; the
    training array is then preprocessed ONCE (numpy) and kept as a tensor.  For norm == "none" the raw array is kept as a
    tensor and preprocessing (log1p + standardise, fitted on train) is applied on-device per batch after the scale.
  * noise aug ("scale+noise"): Gaussian, ``noise_std`` x feature std (preprocessed space is standardised), on-device.
  * ``ex_drop`` p: per (sample, excitation) zero the whole excitation block of features with prob p (after preprocessing;
    at least one excitation is always kept), spectral and spatial inputs alike.
``best_epoch_`` is the 1-indexed number of epochs to train for to reproduce the restored weights (when no
validation set is given, it equals ``epochs``).
"""
from __future__ import annotations

import copy
import math
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from .preprocess import PREPROC_DEFAULT, PREPROC_SPACE, make_preprocessor

# Frozen shared search space for ALL neural models (baselines and ours). samples_per_specimen (500), weight decay,
# label smoothing, epochs (30) and patience (6) are fixed in TRAIN_DEFAULT.
TRAIN_SPACE: dict[str, list[Any]] = dict(
    lr=[3e-4, 1e-3, 3e-3],
    aug=["none", "scale+noise"],
    ex_drop=[0.0, 0.25],
)
TRAIN_DEFAULT: dict[str, Any] = dict(
    lr=1e-3, weight_decay=1e-4, epochs=30, warmup_epochs=3, batch_size=256, samples_per_specimen=500,
    label_smoothing=0.05, aug="scale+noise", noise_std=0.01, patience=6, grad_clip=1.0, ex_drop=0.0)


def balanced_accuracy(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    rec = []
    for c in range(n_classes):
        m = y == c
        if m.any():
            rec.append(float((pred[m] == c).mean()))
    return float(np.mean(rec)) if rec else 0.0


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class TorchEstimator:
    """Implements ``types.Estimator``. ``module_factory(meta, cfg) -> nn.Module`` mapping the preprocessed
    float32 tensor ([B,F] spectral or [B,P,P,F] spatial) to logits [B,C]."""

    def __init__(self, module_factory: Callable[[dict, dict], nn.Module], cfg: dict, meta: dict, seed: int,
                 device: str, input_kind: str = "spectral"):
        self.module_factory, self.meta, self.seed, self.device = module_factory, meta, int(seed), device
        self.input_kind = input_kind
        self.cfg = {**TRAIN_DEFAULT, **PREPROC_DEFAULT, **cfg}
        self.best_epoch_: int = 0
        self.history_: list[dict] = []
        self.module: nn.Module | None = None
        self.pre = None

    # -- helpers --
    def _augment_noise(self, xt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """numpy noise aug (kept for TorchEstimatorV2 and the per_ex fallback)."""
        if self.cfg["aug"] != "scale+noise" or self.cfg["noise_std"] <= 0:
            return xt
        noise = rng.standard_normal(xt.shape, dtype=np.float32) * np.float32(self.cfg["noise_std"])
        if self.input_kind == "spatial":
            noise *= np.any(xt != 0, axis=-1, keepdims=True)
        return xt + noise

    def _tpre_none(self, x: torch.Tensor) -> torch.Tensor:
        """On-device equivalent of ``Preprocessor.transform`` for norm == 'none' (log1p + standardise, zero pixels stay 0)."""
        pre = self.pre
        valid = (x != 0).any(-1, keepdim=True)
        if pre.log1p:
            x = torch.log1p(x.clamp_min(0.0) / pre.scale_)
        x = (x - self._t_mean) / self._t_std
        return x * valid

    def _t_noise(self, xb: torch.Tensor) -> torch.Tensor:
        n = torch.randn_like(xb) * float(self.cfg["noise_std"])
        if self.input_kind == "spatial":
            n = n * (xb != 0).any(-1, keepdim=True)
        return xb + n

    def _t_ex_drop(self, xb: torch.Tensor, p: float) -> torch.Tensor:
        B = xb.shape[0]
        drop = torch.rand(B, self._n_ex, device=xb.device) < p
        all_d = drop.all(1)
        drop[:, 0] = drop[:, 0] & ~all_d  # always keep at least one excitation
        keep = (~drop)[:, self._ex_ids_t].to(xb.dtype)  # [B, F]
        if xb.dim() == 4:
            keep = keep[:, None, None, :]
        return xb * keep

    def _to_dev(self, a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(self.device)

    def _predict_logits(self, Xt: torch.Tensor, bs: int | None = None) -> np.ndarray:
        bs = bs or 512  # large inference batches blow up mps memory for token models (O(B*T^2) attention)
        self.module.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xt), bs):
                out.append(self.module(Xt[i:i + bs]).float().cpu().numpy())
        return np.concatenate(out)

    def _transform_batched(self, X: np.ndarray, bs: int = 4096) -> np.ndarray:
        return np.concatenate([self.pre.transform(X[i:i + bs]) for i in range(0, len(X), bs)])

    # -- API --
    def fit(self, X, y, groups, X_val=None, y_val=None) -> "TorchEstimator":
        cfg = self.cfg
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        groups = np.asarray(groups)
        C = int(self.meta["n_classes"])
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        self.pre = make_preprocessor(cfg, self.meta).fit(X)
        self.module = self.module_factory(self.meta, cfg).to(self.device)

        # specimen-balanced sampling distribution (with replacement)
        uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
        p = (1.0 / counts[inv]).astype(np.float64)
        p /= p.sum()
        epoch_size = int(cfg["samples_per_specimen"]) * len(uniq)
        bs = int(cfg["batch_size"])
        steps_per_epoch = max(1, math.ceil(epoch_size / bs))
        E = int(cfg["epochs"])
        total, warm = E * steps_per_epoch, int(cfg["warmup_epochs"]) * steps_per_epoch

        opt = torch.optim.AdamW(self.module.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        def lr_lambda(step):
            if step < warm:
                return (step + 1) / max(1, warm)
            return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        crit = nn.CrossEntropyLoss(label_smoothing=float(cfg["label_smoothing"]))

        # one-off setup of the training tensor (see module docstring for the scale-aug semantics)
        spatial = self.input_kind == "spatial"
        scale_aug = "scale" in cfg["aug"] and cfg["norm"] == "none"  # scale aug is an exact no-op for any other norm
        noise_aug = cfg["aug"] == "scale+noise" and float(cfg["noise_std"]) > 0
        p_drop = float(cfg.get("ex_drop", 0.0) or 0.0)
        ex_ids = np.unique(np.asarray(self.meta["ex_nm"], float), return_inverse=True)[1]
        self._n_ex = int(ex_ids.max() + 1)
        self._ex_ids_t = torch.from_numpy(ex_ids.astype(np.int64)).to(self.device)
        self._t_mean = torch.from_numpy(self.pre.mean_).to(self.device)
        self._t_std = torch.from_numpy(self.pre.std_).to(self.device)
        keep_dev = not spatial  # spatial training arrays stay in host memory, batches are moved per step
        if scale_aug:
            train_src = torch.from_numpy(np.ascontiguousarray(X))
        else:
            train_src = torch.from_numpy(self._transform_batched(X))
        if keep_dev:
            train_src = train_src.to(self.device)
        yt_all = torch.from_numpy(y).to(self.device)
        log_lo, log_hi = math.log(0.7), math.log(1.4)

        have_val = X_val is not None and y_val is not None and len(X_val) > 0
        Xv = self._to_dev(self._transform_batched(np.asarray(X_val, np.float32))) if have_val else None
        y_val = np.asarray(y_val, dtype=np.int64) if have_val else None

        best_score, best_state, self.best_epoch_, bad = -1.0, None, E, 0
        self.history_ = []
        for ep in range(1, E + 1):
            self.module.train()
            idx_all = rng.choice(len(X), size=epoch_size, replace=True, p=p)
            tot_loss, tot_correct, tot_n = 0.0, 0, 0
            for s in range(0, epoch_size, bs):
                idx = idx_all[s:s + bs]
                it = torch.from_numpy(idx)
                xb = train_src[it.to(self.device) if keep_dev else it]
                if not keep_dev:
                    xb = xb.to(self.device)
                yb = yt_all[it.to(self.device)]
                if scale_aug:
                    sc = torch.exp(torch.empty((len(idx),) + (1,) * (xb.dim() - 1), device=self.device)
                                   .uniform_(log_lo, log_hi))
                    xb = self._tpre_none(xb * sc)
                if noise_aug:
                    xb = self._t_noise(xb)
                if p_drop > 0:
                    xb = self._t_ex_drop(xb, p_drop)
                logits = self.module(xb)
                loss = crit(logits, yb)
                if hasattr(self.module, "extra_loss"):
                    loss = loss + self.module.extra_loss()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg["grad_clip"]:
                    nn.utils.clip_grad_norm_(self.module.parameters(), float(cfg["grad_clip"]))
                opt.step()
                sched.step()
                tot_loss += float(loss.detach()) * len(idx)
                tot_correct += int((logits.argmax(1) == yb).sum())
                tot_n += len(idx)
            rec: dict[str, Any] = dict(epoch=ep, train_loss=tot_loss / tot_n, train_acc=tot_correct / tot_n)
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

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        out = []
        step = 2048 if self.input_kind == "spectral" else 512  # host-side chunks; device batches are 512
        for i in range(0, len(X), step):
            xt = self._to_dev(self.pre.transform(X[i:i + step]))
            out.append(_softmax(self._predict_logits(xt)))
        return np.concatenate(out)


# --------------------------------------------------------------------------------------------------------------
class SklearnEstimator:
    """Preprocess (fit on train) + sklearn model.

    ``make_model(cfg, meta, seed)`` returns the sklearn estimator. ``sample_weight``: if True, fit with weight
    1/specimen-size (normalised to mean 1). ``cfg['max_train']`` (optional): specimen-balanced seeded subsample
    (without replacement, p proportional to 1/specimen-size) before fitting.
    ``predict_proba``: uses the model's ``predict_proba`` if present, otherwise a softmax of ``decision_function``
    (LinearSVC / SVC; documented choice: cheap, monotone, avoids the internal CV of ``probability=True``).
    Columns are always aligned to ``meta['n_classes']`` (absent classes get probability 0).
    """

    def __init__(self, make_model: Callable[[dict, dict, int], Any], cfg: dict, meta: dict, seed: int,
                 sample_weight: bool = True):
        self.make_model, self.cfg, self.meta, self.seed = make_model, {**PREPROC_DEFAULT, **cfg}, meta, int(seed)
        self.use_sw = sample_weight
        self.pre = None
        self.model = None

    def _flat(self, A: np.ndarray) -> np.ndarray:
        return A.reshape(len(A), -1)

    def fit(self, X, y, groups, X_val=None, y_val=None) -> "SklearnEstimator":
        X = np.asarray(X, dtype=np.float32)
        y, groups = np.asarray(y), np.asarray(groups)
        mt = self.cfg.get("max_train")
        if mt and len(X) > mt:
            _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
            p = 1.0 / counts[inv]
            p /= p.sum()
            sel = np.random.default_rng(self.seed).choice(len(X), size=int(mt), replace=False, p=p)
            sel.sort()
            X, y, groups = X[sel], y[sel], groups[sel]
        self.pre = make_preprocessor(self.cfg, self.meta).fit(X)
        Xt = self._flat(self.pre.transform(X))
        self.model = self.make_model(self.cfg, self.meta, self.seed)
        if self.use_sw:
            _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
            sw = 1.0 / counts[inv]
            sw = sw / sw.mean()
            self.model.fit(Xt, y, sample_weight=sw)
        else:
            self.model.fit(Xt, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        Xt = self._flat(self.pre.transform(np.asarray(X, dtype=np.float32)))
        C = int(self.meta["n_classes"])
        if hasattr(self.model, "predict_proba"):
            P = self.model.predict_proba(Xt)
        else:
            d = self.model.decision_function(Xt)
            if d.ndim == 1:
                d = np.stack([-d, d], 1)
            P = _softmax(d)
        out = np.zeros((len(Xt), C), np.float64)
        out[:, np.asarray(self.model.classes_, dtype=int)] = P
        return out
