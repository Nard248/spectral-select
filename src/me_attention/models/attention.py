"""Our attention family for multi-excitation fluorescence pixels (spectral + spatial tracks).

Models (all map PREPROCESSED float32 ``[B, F]`` -> logits ``[B, C]``; spatial: ``[B, P, P, F]``):

* ``smartattention_v0`` -- Gevorg's MESmartAttention on the aligned E x M grid built from ``grid_index``.
* ``meas_tokens``       -- one token per valid (ex, em) measurement (+ wavelength encoding), optional relative
                           position bias (``relbias``) and learned per-feature gates (``gated``).
* ``axial``             -- tokens on the E x M grid, alternating emission-row / excitation-column attention.
* ``ex_tokens``         -- one token per excitation (optionally ``sub_tokens`` emission chunks).
* ``spatial_attn``      -- shared spectral encoder per patch pixel -> attention over pixel tokens.

Conventions
-----------
* Grids are irregular: nothing is assumed regular except through ``meta['grid_index']``.
* Key masking uses a finite additive value (``-1e4``) rather than ``-inf`` so fully masked rows give a uniform
  softmax instead of NaN. Outputs of masked query cells are zeroed / ignored by the pooling.
* Every model exposes ``extra_loss()`` (0 unless ``gated``), ``feature_importance()`` (gate values or ``None``)
  and ``attention_importance(x, mode)`` (rollout / last-layer attention readout mapped to per-feature scores).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as Fn
from torch import nn

from ..types import ModelSpec

NEG = -1e4


# ============================================================================================ small helpers
def _t(a, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a), dtype=dtype)


def _ranges(meta) -> dict[str, tuple[float, float]]:
    ex, em = np.asarray(meta["ex_nm"], float), np.asarray(meta["em_nm"], float)
    st = em - ex
    out = {}
    for k, a in (("ex", ex), ("em", em), ("st", st)):
        lo, hi = float(a.min()), float(a.max())
        out[k] = (lo, max(hi - lo, 1.0))
    return out


def _grid_gather(meta) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Gather index [E*M] into [x, 0] (F -> zero token) and validity mask [E, M]."""
    gi = np.asarray(meta["grid_index"], dtype=np.int64)
    E, M = len(meta["excitations"]), len(meta["emissions"])
    g = np.full(E * M, len(gi), dtype=np.int64)
    g[gi[:, 0] * M + gi[:, 1]] = np.arange(len(gi))
    valid = np.zeros((E, M), dtype=bool)
    valid[gi[:, 0], gi[:, 1]] = True
    return torch.from_numpy(g), torch.from_numpy(valid), E, M


def _to_grid(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """x [B, F, ...] -> [B, E*M, ...] with zeros at missing cells."""
    pad = x.new_zeros((x.shape[0], 1) + tuple(x.shape[2:]))
    return torch.cat([x, pad], 1)[:, g]


class WaveEnc(nn.Module):
    """Fourier features of (ex, em, em-ex) scaled to [0, 1] -> Linear -> [N, dim]."""

    def __init__(self, ex_nm, em_nm, rng, dim, n_freq=8):
        super().__init__()
        ex, em = np.asarray(ex_nm, float), np.asarray(em_nm, float)
        u = np.stack([(ex - rng["ex"][0]) / rng["ex"][1], (em - rng["em"][0]) / rng["em"][1],
                      ((em - ex) - rng["st"][0]) / rng["st"][1]], 1)
        self.register_buffer("u", _t(u), persistent=False)
        self.register_buffer("freq", 2 * math.pi * torch.arange(1, n_freq + 1, dtype=torch.float32), persistent=False)
        self.lin = nn.Linear(3 * (2 * n_freq + 1), dim)

    def forward(self) -> torch.Tensor:
        ang = self.u[:, :, None] * self.freq
        f = torch.cat([self.u, torch.sin(ang).flatten(1), torch.cos(ang).flatten(1)], 1)
        return self.lin(f)


class ValEmbed(nn.Module):
    def __init__(self, dim, kind="linear"):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.w = nn.Parameter(torch.randn(dim) * 0.5)
            self.b = nn.Parameter(torch.zeros(dim))
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        else:
            raise ValueError(f"val_embed {kind!r}")

    def forward(self, x):  # [..., ] -> [..., dim]
        if self.kind == "linear":
            return x.unsqueeze(-1) * self.w + self.b
        return self.net(x.unsqueeze(-1))


class PairBias(nn.Module):
    """Relative position bias: MLP(pairwise |delta| features) -> per-head additive bias [1, H, N, N]."""

    def __init__(self, feats: torch.Tensor, heads: int, hidden: int = 32):
        super().__init__()
        self.register_buffer("feats", feats.float(), persistent=False)  # [N, N, k]
        self.mlp = nn.Sequential(nn.Linear(feats.shape[-1], hidden), nn.GELU(), nn.Linear(hidden, heads))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self) -> torch.Tensor:
        return self.mlp(self.feats).permute(2, 0, 1).unsqueeze(0)


def _meas_pair_feats(meta) -> torch.Tensor:
    ex, em = np.asarray(meta["ex_nm"], float), np.asarray(meta["em_nm"], float)
    st, r = em - ex, _ranges(meta)
    d = [np.abs(a[:, None] - a[None, :]) / r[k][1] for a, k in ((ex, "ex"), (em, "em"), (st, "st"))]
    return _t(np.stack(d, -1))


# ============================================================================================ attention core
class MHA(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.h, self.p = heads, dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias=None, kpm=None, need_attn=False):
        """x [B,N,D]; bias additive float broadcastable to [B,H,N,N]; kpm bool [B,N] (True = ignore key)."""
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        mask = bias
        if kpm is not None:
            km = kpm[:, None, None, :].to(x.dtype) * NEG
            mask = km if mask is None else mask + km
        if need_attn:
            a = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
            if mask is not None:
                a = a + mask
            a = a.softmax(-1)
            out = self.drop(a) @ v
        else:
            a = None
            if mask is not None:
                mask = mask.expand(B, self.h, N, N) if mask.shape != (B, self.h, N, N) else mask
            out = Fn.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                  dropout_p=self.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out), a


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = MHA(dim, heads, dropout)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, dim), nn.Dropout(dropout))
        self.dp = float(drop_path)

    def _dp(self, y):
        if not self.training or self.dp <= 0:
            return y
        keep = (torch.rand(y.shape[0], *([1] * (y.dim() - 1)), device=y.device) >= self.dp).to(y.dtype)
        return y * keep / (1 - self.dp)

    def forward(self, x, bias=None, kpm=None, need_attn=False):
        a, w = self.attn(self.n1(x), bias, kpm, need_attn)
        x = x + self._dp(a)
        x = x + self._dp(self.mlp(self.n2(x)))
        return x, w


class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_ratio=2.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        dps = np.linspace(0, drop_path, max(depth, 1))
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, dropout, float(dps[i])) for i in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, bias=None, kpm=None, need_attn=False):
        attns = []
        for b in self.blocks:
            x, w = b(x, bias, kpm, need_attn)
            attns.append(w)
        return self.norm(x), attns


class AttnPool(nn.Module):
    """Learned-query attention pooling (single query, softmax over unmasked tokens)."""

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x, kpm=None, weight=None):
        s = self.score(x).squeeze(-1)
        if kpm is not None:
            s = s + kpm.to(s.dtype) * NEG
        if weight is not None:  # multiplicative per-token weight (gates)
            s = s + torch.log(weight + 1e-6)
        w = s.softmax(-1)
        return (w.unsqueeze(-1) * x).sum(1), w


def masked_mean(x, keep, weight=None):
    """x [B,N,D]; keep [B,N] float 0/1."""
    w = keep if weight is None else keep * weight
    return (x * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-6), w


def rollout(attns: list[torch.Tensor], readout: torch.Tensor, mode: str = "rollout") -> torch.Tensor:
    """attns: list of [B,H,N,N] (rows = queries). readout [B,N] weights over output tokens. -> [B,N] token scores."""
    As = [a.mean(1) for a in attns]
    if mode == "last":
        return (readout.unsqueeze(1) @ As[-1]).squeeze(1)
    N = As[0].shape[-1]
    I = torch.eye(N, device=As[0].device, dtype=As[0].dtype)
    R = I.expand_as(As[0])
    for A in As:
        R = (0.5 * A + 0.5 * I) @ R
    return (readout.unsqueeze(1) @ R).squeeze(1)


class _Base(nn.Module):
    """Common analysis hooks."""

    def extra_loss(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)

    def feature_importance(self):
        return None

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def attention_importance(self, x: torch.Tensor, mode: str = "rollout") -> np.ndarray:
        """Per-feature attention readout averaged over the batch ``x`` ([F] numpy; spatial: [P, P])."""
        was = self.training
        self.eval()
        try:
            return self._importance(x, mode).float().cpu().numpy()
        finally:
            self.train(was)

    def _importance(self, x, mode):  # pragma: no cover
        raise NotImplementedError


def _head(dim, n_classes, dropout):
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_classes)) if n_classes else nn.Identity()


# ============================================================================================ 1. v0
class _V0Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner = heads * dim_head
        self.h, self.scale = heads, dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
        self.last = None

    def forward(self, x, need_attn=False):
        B, N, _ = x.shape
        q, k, v = [t.view(B, N, self.h, -1).transpose(1, 2) for t in self.to_qkv(x).chunk(3, -1)]
        a = ((q @ k.transpose(-1, -2)) * self.scale).softmax(-1)
        self.last = a if need_attn else None
        return self.to_out((a @ v).transpose(1, 2).reshape(B, N, -1))


class _V0Layer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = _V0Attention(dim, heads, dim_head, dropout)
        self.ff = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_dim, dim), nn.Dropout(dropout))

    def forward(self, x, need_attn=False):
        x = x + self.attn(self.n1(x), need_attn)
        return x + self.ff(self.n2(x))


class SmartAttentionV0(_Base):
    """Gevorg's MESmartAttention (einops removed) on the aligned [B, E, M] grid. Edge modes: zero | discard."""

    def __init__(self, meta, n_classes, dim=32, depth=3, heads=2, dim_head=16, mlp_ratio=2, dropout=0.1,
                 embedding_dropout=0.1, edge_mode="zero"):
        super().__init__()
        if edge_mode not in ("zero", "discard"):
            raise ValueError("edge_mode must be 'zero' or 'discard'")
        g, valid, E, M = _grid_gather(meta)
        if M < 3:
            raise ValueError("need >= 3 emission positions")
        self.register_buffer("g", g, persistent=False)
        self.E, self.M, self.edge_mode = E, M, edge_mode
        self.n_tok = M - 2 if edge_mode == "discard" else M
        self.token_embedding = nn.Linear(3 * E, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, M + 1, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.drop = nn.Dropout(embedding_dropout)
        self.layers = nn.ModuleList([_V0Layer(dim, heads, dim_head, int(dim * mlp_ratio), dropout) for _ in range(depth)])
        # cross-layer feature fusion (CAF skipcat)
        self.skipcat = nn.ModuleList([nn.Conv2d(self.n_tok + 1, self.n_tok + 1, (1, 2)) for _ in range(max(0, depth - 2))])
        self.classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_classes))
        gi = np.asarray(meta["grid_index"])
        self.register_buffer("feat_m", torch.from_numpy(gi[:, 1].astype(np.int64)), persistent=False)

    def make_tokens(self, grid):  # [B,E,M]
        x = grid.transpose(1, 2)  # [B, M, E]
        if self.edge_mode == "zero":
            z = torch.zeros_like(x[:, :1])
            prev, nxt = torch.cat([z, x[:, :-1]], 1), torch.cat([x[:, 1:], z], 1)
            return torch.cat([prev, x, nxt], -1)
        return torch.cat([x[:, :-2], x[:, 1:-1], x[:, 2:]], -1)

    def _pos(self):
        if self.edge_mode == "discard":
            return torch.cat([self.pos_embedding[:, :1], self.pos_embedding[:, 2:self.M]], 1)
        return self.pos_embedding

    def _run(self, x, need_attn=False):
        grid = _to_grid(x, self.g).view(x.shape[0], self.E, self.M)
        t = self.token_embedding(self.make_tokens(grid))
        t = torch.cat([self.cls_token.expand(t.shape[0], -1, -1), t], 1) + self._pos()
        t = self.drop(t)
        prev = []
        for i, layer in enumerate(self.layers):
            prev.append(t)
            if i > 1:
                fusion = torch.cat([t.unsqueeze(-1), prev[i - 2].unsqueeze(-1)], -1)
                t = self.skipcat[i - 2](fusion).squeeze(-1)
            t = layer(t, need_attn)
        return t

    def forward(self, x):
        return self.classifier(self._run(x)[:, 0])

    def _importance(self, x, mode):
        self._run(x, need_attn=True)
        attns = [l.attn.last for l in self.layers]
        B, N = x.shape[0], attns[0].shape[-1]
        r = torch.zeros(B, N, device=x.device)
        r[:, 0] = 1
        tok = rollout(attns, r, mode).mean(0)[1:]  # [n_tok]
        full = torch.zeros(self.M, device=x.device)
        if self.edge_mode == "discard":
            full[1:self.M - 1] = tok
        else:
            full = tok
        return full[self.feat_m]


def _v0_factory(meta, cfg):
    d = {**V0_DEFAULT, **{k: cfg[k] for k in V0_DEFAULT if k in cfg}}
    return SmartAttentionV0(meta, int(meta["n_classes"]), **d)


# ============================================================================================ 2. meas_tokens
class MeasTokens(_Base):
    def __init__(self, meta, n_classes, dim=64, depth=3, heads=4, mlp_ratio=2, dropout=0.1, drop_path=0.0,
                 pos="fourier", n_freq=8, val_embed="linear", pool="cls", token_drop=0.0, relbias=False,
                 gated=False, gate_l1=0.0, head_dropout=None):
        super().__init__()
        if pos not in ("fourier", "learned", "both"):
            raise ValueError(f"pos {pos!r}")
        if pool not in ("cls", "mean", "attn"):
            raise ValueError(f"pool {pool!r}")
        F = int(meta["n_features"])
        self.F, self.dim, self.pool_kind, self.token_drop = F, dim, pool, float(token_drop)
        self.pos_kind, self.gated, self.gate_l1 = pos, bool(gated), float(gate_l1)
        self.val = ValEmbed(dim, val_embed)
        if pos in ("fourier", "both"):
            self.wave = WaveEnc(meta["ex_nm"], meta["em_nm"], _ranges(meta), dim, n_freq)
        if pos in ("learned", "both"):
            self.pos_emb = nn.Embedding(F, dim)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
            self.register_buffer("ar", torch.arange(F), persistent=False)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02) if pool == "cls" else None
        self.relbias = PairBias(_meas_pair_feats(meta), heads) if relbias else None
        self.gate_logit = nn.Parameter(torch.full((F,), 2.0)) if gated else None
        self.enc = Encoder(dim, depth, heads, mlp_ratio, dropout, drop_path)
        self.apool = AttnPool(dim) if pool == "attn" else None
        self.head = _head(dim, n_classes, dropout if head_dropout is None else head_dropout)
        self._heads = heads

    def _pos(self):
        p = 0
        if self.pos_kind in ("fourier", "both"):
            p = p + self.wave()
        if self.pos_kind in ("learned", "both"):
            p = p + self.pos_emb(self.ar)
        return p

    def gates(self):
        return torch.sigmoid(self.gate_logit) if self.gated else None

    def extra_loss(self):
        if not self.gated or self.gate_l1 <= 0:
            return super().extra_loss()
        return self.gate_l1 * self.gates().mean()

    def feature_importance(self):
        return None if not self.gated else self.gates().detach().cpu().numpy()

    def encode(self, x, need_attn=False):
        B, F = x.shape
        t = self.val(x) + self._pos()
        g = self.gates()
        if g is not None:
            t = t * g[None, :, None]
        kpm = None
        if self.training and self.token_drop > 0:
            kpm = torch.rand(B, F, device=x.device) < self.token_drop
        bias = self.relbias() if self.relbias is not None else None
        kb = None if g is None else torch.log(g + 1e-6)
        if self.cls is not None:
            t = torch.cat([self.cls.expand(B, -1, -1), t], 1)
            if kpm is not None:
                kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), kpm], 1)
            if bias is not None:
                bias = Fn.pad(bias, (1, 0, 1, 0))
            if kb is not None:
                kb = Fn.pad(kb, (1, 0))
        if kb is not None:
            kbias = kb[None, None, None, :]
            bias = kbias if bias is None else bias + kbias
        h, attns = self.enc(t, bias, kpm, need_attn)
        if self.pool_kind == "cls":
            z, w = h[:, 0], None
        elif self.pool_kind == "mean":
            keep = torch.ones(B, F, device=x.device) if kpm is None else (~kpm).float()
            z, w = masked_mean(h, keep, g[None] if g is not None else None)
        else:
            z, w = self.apool(h, kpm, g[None] if g is not None else None)
        return z, attns, w

    def forward(self, x):
        return self.head(self.encode(x)[0])

    def _importance(self, x, mode):
        _, attns, w = self.encode(x, need_attn=True)
        B, N = x.shape[0], attns[0].shape[-1]
        if self.pool_kind == "cls":
            r = torch.zeros(B, N, device=x.device)
            r[:, 0] = 1
        else:
            r = w / w.sum(1, keepdim=True)
        s = rollout(attns, r, mode).mean(0)
        return s[1:] if self.pool_kind == "cls" else s


# ============================================================================================ 3. axial
class Axial(_Base):
    def __init__(self, meta, n_classes, dim=64, depth=2, heads=4, mlp_ratio=2, dropout=0.1, drop_path=0.0,
                 n_freq=8, val_embed="linear", pool="mean", token_drop=0.0, relbias=False):
        super().__init__()
        g, valid, E, M = _grid_gather(meta)
        self.register_buffer("g", g, persistent=False)
        self.register_buffer("valid", valid, persistent=False)
        self.E, self.M, self.dim, self.token_drop = E, M, dim, float(token_drop)
        self.pool_kind = "mean" if pool == "mean" else "attn"  # 'cls' == learned-query attention pooling
        self.val = ValEmbed(dim, val_embed)
        self.wave = WaveEnc(meta["ex_nm"], meta["em_nm"], _ranges(meta), dim, n_freq)
        self.row_emb, self.col_emb = nn.Embedding(E, dim), nn.Embedding(M, dim)
        nn.init.normal_(self.row_emb.weight, std=0.02)
        nn.init.normal_(self.col_emb.weight, std=0.02)
        dps = np.linspace(0, drop_path, max(2 * depth, 1))
        self.row_blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, dropout, float(dps[2 * i])) for i in range(depth)])
        self.col_blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, dropout, float(dps[2 * i + 1])) for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.apool = AttnPool(dim) if self.pool_kind == "attn" else None
        self.head = _head(dim, n_classes, dropout)
        if relbias:
            emg, exg = np.asarray(meta["emissions"], float), np.asarray(meta["excitations"], float)
            fm = np.abs(emg[:, None] - emg[None, :]) / max(np.ptp(emg), 1.0)
            fe = np.abs(exg[:, None] - exg[None, :]) / max(np.ptp(exg), 1.0)
            self.row_bias, self.col_bias = PairBias(_t(fm)[..., None], heads), PairBias(_t(fe)[..., None], heads)
        else:
            self.row_bias = self.col_bias = None
        gi = np.asarray(meta["grid_index"])
        self.register_buffer("feat_flat", torch.from_numpy(gi[:, 0] * M + gi[:, 1]), persistent=False)

    def encode(self, x, need_attn=False):
        B, F = x.shape
        E, M = self.E, self.M
        t = self.val(x) + self.wave()
        grid = _to_grid(t, self.g).view(B, E, M, self.dim)
        grid = grid + self.row_emb.weight[None, :, None] + self.col_emb.weight[None, None, :]
        valid = self.valid
        grid = grid * valid[None, :, :, None]
        drop = valid.unsqueeze(0).expand(B, E, M)
        if self.training and self.token_drop > 0:
            drop = drop & (torch.rand(B, E, M, device=x.device) >= self.token_drop)
        kpm = ~drop  # True = ignore as key
        vmask = valid[None, :, :, None].to(grid.dtype)
        rb = self.row_bias() if self.row_bias is not None else None
        cb = self.col_bias() if self.col_bias is not None else None
        attns = []
        for rblk, cblk in zip(self.row_blocks, self.col_blocks):
            h, a = rblk(grid.reshape(B * E, M, self.dim), rb, kpm.reshape(B * E, M), need_attn)
            grid = h.view(B, E, M, self.dim) * vmask
            if need_attn:
                attns.append(("row", a))
            h, a = cblk(grid.transpose(1, 2).reshape(B * M, E, self.dim), cb, kpm.transpose(1, 2).reshape(B * M, E), need_attn)
            grid = h.view(B, M, E, self.dim).transpose(1, 2) * vmask
            if need_attn:
                attns.append(("col", a))
        grid = self.norm(grid).reshape(B, E * M, self.dim)
        keep = drop.reshape(B, E * M).float()
        if self.pool_kind == "mean":
            z, w = masked_mean(grid, keep)
        else:
            z, w = self.apool(grid, ~drop.reshape(B, E * M))
        return z, attns, w

    def forward(self, x):
        return self.head(self.encode(x)[0])

    def _importance(self, x, mode):
        """Mean received attention per cell (rollout is not defined for axial factorisation), averaged over
        layers/heads/queries/batch."""
        B = x.shape[0]
        _, attns, _ = self.encode(x, need_attn=True)
        E, M = self.E, self.M
        cells = []
        for kind, a in attns:
            recv = a.mean(1).mean(1)  # [B*, keys]
            cells.append(recv.view(B, E, M) if kind == "row" else recv.view(B, M, E).transpose(1, 2))
        s = torch.stack(cells).mean(0).mean(0).reshape(-1)
        return s[self.feat_flat]


# ============================================================================================ 4. ex_tokens
class ExTokens(_Base):
    def __init__(self, meta, n_classes, dim=64, depth=3, heads=4, mlp_ratio=2, dropout=0.1, drop_path=0.0,
                 n_freq=8, ex_embed="linear", sub_tokens=1, pool="cls", token_drop=0.0, conv_ch=16):
        super().__init__()
        if ex_embed not in ("linear", "conv"):
            raise ValueError(f"ex_embed {ex_embed!r}")
        g, valid, E, M = _grid_gather(meta)
        S = int(sub_tokens)
        Mp = int(math.ceil(M / (2 * S)) * 2 * S)
        self.E, self.M, self.Mp, self.S, self.dim = E, M, Mp, S, dim
        self.ex_embed, self.pool_kind, self.token_drop = ex_embed, pool, float(token_drop)
        self.register_buffer("g", g, persistent=False)
        vp = torch.zeros(E, Mp)
        vp[:, :M] = valid.float()
        self.register_buffer("maskp", vp, persistent=False)
        emg = np.asarray(meta["emissions"], float)
        r = _ranges(meta)
        emp = np.concatenate([emg, np.full(Mp - M, emg[-1])])
        self.register_buffer("emchan", _t((emp - r["em"][0]) / r["em"][1]), persistent=False)
        Mc = Mp // S
        if ex_embed == "linear":
            self.embed = nn.Linear(2 * Mc, dim)
        else:
            self.conv = nn.Sequential(nn.Conv1d(3, conv_ch, 5, padding=2), nn.GELU(),
                                      nn.Conv1d(conv_ch, conv_ch, 5, stride=2, padding=2), nn.GELU())
            self.embed = nn.Linear(conv_ch * (Mc // 2), dim)
        # token wavelength encoding: (ex, chunk-centre em)
        exn = np.repeat(np.asarray(meta["excitations"], float), S)
        centres = np.array([emp[c * Mc:(c + 1) * Mc][: max(1, min(Mc, M - c * Mc))].mean() if c * Mc < M else emp[M - 1]
                            for c in range(S)])
        emn = np.tile(centres, E)
        self.wave = WaveEnc(exn, emn, r, dim, n_freq)
        self.chunk_emb = nn.Embedding(S, dim)
        nn.init.normal_(self.chunk_emb.weight, std=0.02)
        self.register_buffer("chunk_ids", torch.arange(S).repeat(E), persistent=False)
        tv = vp.view(E, S, Mc).sum(-1) > 0
        self.register_buffer("tok_valid", tv.reshape(E * S), persistent=False)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02) if pool == "cls" else None
        self.enc = Encoder(dim, depth, heads, mlp_ratio, dropout, drop_path)
        self.head = _head(dim, n_classes, dropout)
        gi = np.asarray(meta["grid_index"])
        chunk = np.minimum(gi[:, 1] // Mc, S - 1)
        self.register_buffer("feat_tok", torch.from_numpy(gi[:, 0] * S + chunk), persistent=False)

    def encode(self, x, need_attn=False):
        B = x.shape[0]
        E, M, Mp, S = self.E, self.M, self.Mp, self.S
        grid = _to_grid(x, self.g).view(B, E, M)
        grid = Fn.pad(grid, (0, Mp - M))
        mask = self.maskp.unsqueeze(0).expand(B, E, Mp)
        Mc = Mp // S
        if self.ex_embed == "linear":
            inp = torch.stack([grid, mask], 2).view(B, E, 2, S, Mc).permute(0, 1, 3, 2, 4).reshape(B, E * S, 2 * Mc)
            tok = self.embed(inp)
        else:
            inp = torch.stack([grid, mask, self.emchan.expand(B, E, Mp)], 2).reshape(B * E, 3, Mp)
            h = self.conv(inp)  # [B*E, C, Mp/2]
            C = h.shape[1]
            h = h.view(B * E, C, S, Mc // 2).permute(0, 2, 1, 3).reshape(B, E * S, C * (Mc // 2))
            tok = self.embed(h)
        tok = tok + self.wave() + self.chunk_emb(self.chunk_ids)
        kpm = (~self.tok_valid).unsqueeze(0).expand(B, E * S)
        if self.training and self.token_drop > 0:
            kpm = kpm | (torch.rand(B, E * S, device=x.device) < self.token_drop)
        if self.cls is not None:
            tok = torch.cat([self.cls.expand(B, -1, -1), tok], 1)
            kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), kpm], 1)
        h, attns = self.enc(tok, None, kpm, need_attn)
        if self.pool_kind == "cls":
            z, w = h[:, 0], None
        else:
            z, w = masked_mean(h, (~kpm).float())
        return z, attns, w

    def forward(self, x):
        return self.head(self.encode(x)[0])

    def _importance(self, x, mode):
        _, attns, w = self.encode(x, need_attn=True)
        B, N = x.shape[0], attns[0].shape[-1]
        if self.pool_kind == "cls":
            r = torch.zeros(B, N, device=x.device)
            r[:, 0] = 1
        else:
            r = w / w.sum(1, keepdim=True)
        s = rollout(attns, r, mode).mean(0)
        if self.pool_kind == "cls":
            s = s[1:]
        return s[self.feat_tok]


# ============================================================================================ 5. spatial
class _MLPEnc(_Base):
    def __init__(self, meta, dim, dropout):
        super().__init__()
        F = int(meta["n_features"])
        self.net = nn.Sequential(nn.Linear(F, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim), nn.LayerNorm(dim))

    def encode(self, x):
        return self.net(x), [], None


class SpatialAttn(_Base):
    def __init__(self, meta, n_classes, spec_encoder="ex", dim=32, enc_depth=1, depth=2, heads=2, mlp_ratio=2,
                 dropout=0.1, drop_path=0.0, token_drop=0.0, enc_token_drop=0.0, readout="cls",
                 centre_weight=False, max_patch=9, enc_heads=None, sub_tokens=1, ex_embed="linear"):
        super().__init__()
        eh = enc_heads or heads
        if spec_encoder == "meas":
            self.senc = MeasTokens(meta, 0, dim=dim, depth=enc_depth, heads=eh, mlp_ratio=mlp_ratio, dropout=dropout,
                                   pool="mean", token_drop=enc_token_drop)
        elif spec_encoder == "ex":
            self.senc = ExTokens(meta, 0, dim=dim, depth=enc_depth, heads=eh, mlp_ratio=mlp_ratio, dropout=dropout,
                                 pool="cls", token_drop=enc_token_drop, sub_tokens=sub_tokens, ex_embed=ex_embed)
        elif spec_encoder == "mlp":
            self.senc = _MLPEnc(meta, dim, dropout)
        else:
            raise ValueError(f"spec_encoder {spec_encoder!r}")
        if readout not in ("cls", "centre"):
            raise ValueError(f"readout {readout!r}")
        self.dim, self.readout, self.centre_weight, self.token_drop = dim, readout, bool(centre_weight), float(token_drop)
        self.max_patch = max_patch
        self.row_emb, self.col_emb = nn.Embedding(max_patch, dim), nn.Embedding(max_patch, dim)
        nn.init.normal_(self.row_emb.weight, std=0.02)
        nn.init.normal_(self.col_emb.weight, std=0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02) if readout == "cls" else None
        self.enc = Encoder(dim, depth, heads, mlp_ratio, dropout, drop_path)
        self.cnorm = nn.LayerNorm(dim) if centre_weight else None
        self.head = _head(dim * (2 if centre_weight else 1), n_classes, dropout)

    def encode(self, x, need_attn=False):
        B, P, P2, F = x.shape
        assert P == P2 and P <= self.max_patch and P % 2 == 1, "odd square patches up to max_patch"
        n, c = P * P, (P * P) // 2
        flat = x.reshape(B * n, F)
        valid = flat.abs().sum(-1) > 0
        valid = valid.view(B, n)
        valid[:, c] = True  # centre pixel is always valid
        vflat = valid.reshape(-1)
        idx = vflat.nonzero().squeeze(1)
        z = self.senc.encode(flat[idx])[0]
        emb = x.new_zeros(B * n, self.dim).index_put((idx,), z).view(B, n, self.dim)
        r = torch.arange(P, device=x.device) + (self.max_patch - P) // 2
        pos = (self.row_emb(r)[:, None] + self.col_emb(r)[None, :]).reshape(1, n, self.dim)
        tok = (emb + pos) * valid.unsqueeze(-1)
        kpm = ~valid
        if self.training and self.token_drop > 0:
            d = torch.rand(B, n, device=x.device) < self.token_drop
            d[:, c] = False
            kpm = kpm | d
        ci = 0
        if self.cls is not None:
            tok = torch.cat([self.cls.expand(B, -1, -1), tok], 1)
            kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), kpm], 1)
            ci = 1
        h, attns = self.enc(tok, None, kpm, need_attn)
        out = h[:, 0] if self.cls is not None else h[:, c]
        if self.centre_weight:
            out = torch.cat([out, self.cnorm(emb[:, c])], -1)
        return out, attns, (ci, c, kpm)

    def forward(self, x):
        return self.head(self.encode(x)[0])

    def _importance(self, x, mode):
        """Pixel-token attention readout map [P, P] (batch mean); spectral importance lives in ``self.senc``."""
        _, attns, (ci, c, kpm) = self.encode(x, need_attn=True)
        B, N = x.shape[0], attns[0].shape[-1]
        r = torch.zeros(B, N, device=x.device)
        r[:, 0 if ci else c] = 1
        s = rollout(attns, r, mode).mean(0)[ci:]
        P = x.shape[1]
        return s.view(P, P)


# ============================================================================================ factories + registry
V0_DEFAULT = dict(dim=32, depth=3, heads=2, dim_head=16, mlp_ratio=2, dropout=0.1, embedding_dropout=0.1, edge_mode="zero")
MEAS_DEFAULT = dict(dim=32, depth=2, heads=4, mlp_ratio=2, dropout=0.1, drop_path=0.0, pos="fourier", n_freq=8,
                    val_embed="linear", pool="cls", token_drop=0.1, relbias=False, gated=False, gate_l1=0.03)
AXIAL_DEFAULT = dict(dim=64, depth=2, heads=4, mlp_ratio=2, dropout=0.1, drop_path=0.0, n_freq=8, val_embed="linear",
                     pool="mean", token_drop=0.1, relbias=False)
EX_DEFAULT = dict(dim=64, depth=3, heads=4, mlp_ratio=2, dropout=0.1, drop_path=0.0, n_freq=8, ex_embed="linear",
                  sub_tokens=1, pool="cls", token_drop=0.1)
SPATIAL_DEFAULT = dict(spec_encoder="ex", dim=32, enc_depth=1, depth=2, heads=2, mlp_ratio=2, dropout=0.1,
                       token_drop=0.1, enc_token_drop=0.0, readout="cls", centre_weight=False)


def _mk(cls, defaults):
    def factory(meta, cfg):
        kw = {k: cfg[k] for k in defaults if k in cfg}
        return cls(meta, int(meta["n_classes"]), **{**defaults, **kw})
    return factory


meas_tokens_factory = _mk(MeasTokens, MEAS_DEFAULT)
axial_factory = _mk(Axial, AXIAL_DEFAULT)
ex_tokens_factory = _mk(ExTokens, EX_DEFAULT)
spatial_factory = _mk(SpatialAttn, SPATIAL_DEFAULT)
smartattention_v0_factory = _v0_factory

_DIM = [32, 64, 128]
# FROZEN sweep spaces (docs/me_attention/FROZEN_SPACES.md): <= 4 model keys each; shared keys {lr, aug, ex_drop, norm, log1p}
# are added in build_registry. Non-sweep models (axial, ex_tokens) keep a small legacy space.
MODEL_SPACES: dict[str, dict[str, list[Any]]] = dict(
    smartattention_v0=dict(dim=[16, 32, 64], depth=[2, 3, 4], dropout=[0.0, 0.1, 0.3], edge_mode=["zero", "discard"]),
    meas_tokens=dict(dim=[32, 64], depth=[2], heads=[4], pos=["fourier", "both"], relbias=[False, True],
                     pool=["cls", "attn"]),
    axial=dict(dim=[32, 64], depth=[1, 2], dropout=[0.1, 0.3], pool=["mean", "attn"]),
    ex_tokens=dict(dim=[32, 64], depth=[2, 3], sub_tokens=[1, 2], dropout=[0.1, 0.3]),
    spatial_attn=dict(spec_encoder=["ex", "mlp"], dim=[32, 64], depth=[1, 2]),  # + P in {5,7} (registry)
)
_DEFAULTS = dict(smartattention_v0=V0_DEFAULT, meas_tokens=MEAS_DEFAULT, axial=AXIAL_DEFAULT, ex_tokens=EX_DEFAULT,
                 spatial_attn=SPATIAL_DEFAULT)
_FACTORIES = dict(smartattention_v0=_v0_factory, meas_tokens=meas_tokens_factory, axial=axial_factory,
                  ex_tokens=ex_tokens_factory, spatial_attn=spatial_factory)
_DESC = dict(
    smartattention_v0="MESmartAttention (Gevorg) on the aligned E x M grid: 3-neighbour emission tokens, CLS, CAF skipcat",
    meas_tokens="one token per (ex, em) measurement + Fourier/learned wavelength encoding; relbias/gated options",
    axial="E x M grid with alternating emission-row / excitation-column attention and key-padding for missing cells",
    ex_tokens="one token per excitation (masked linear/conv embedding of its emission spectrum), optional sub-tokens",
    spatial_attn="shared spectral encoder per patch pixel -> attention over pixel tokens with 2-D position embedding")


def _train_parts():
    """Shared frozen spaces/defaults (preprocessing + training) -- identical for every neural model."""
    from ..preprocess import PREPROC_DEFAULT, PREPROC_SPACE
    from ..train import TRAIN_DEFAULT, TRAIN_SPACE
    return {**TRAIN_SPACE, **PREPROC_SPACE}, {**TRAIN_DEFAULT, **PREPROC_DEFAULT}


def _make_build(factory, input_kind):
    def build(cfg, meta, seed, device):
        from ..train import TorchEstimator
        return TorchEstimator(factory, cfg, meta, seed, device, input_kind=input_kind)
    return build


def build_registry() -> dict[str, ModelSpec]:
    tspace, tdef = _train_parts()
    reg = {}
    for name, space in MODEL_SPACES.items():
        track = "spatial" if name == "spatial_attn" else "spectral"
        reg[name] = ModelSpec(name=name, track=track, family="ours",
                              build=_make_build(_FACTORIES[name], "spatial" if track == "spatial" else "spectral"),
                              space={**space, **tspace}, default={**_DEFAULTS[name], **tdef}, description=_DESC[name])
    return reg


ATTENTION_MODELS: dict[str, ModelSpec] = build_registry()
SPECTRAL_FACTORIES = {k: v for k, v in _FACTORIES.items() if k != "spatial_attn"}


# ============================================================================================ synthetic meta
def synthetic_meta(n_features: int = 60, n_ex: int = 6, n_classes: int = 3, seed: int = 0, step: float = 10.0,
                   em_lo: float = 400.0, em_hi: float = 750.0, offset_ex: bool = True) -> dict[str, Any]:
    """Irregular ME grid: gaps inside each excitation; one excitation sampled on a 5-nm-offset emission grid."""
    rng = np.random.default_rng(seed)
    exs = np.linspace(300.0, 300.0 + 20.0 * (n_ex - 1), n_ex)
    base = np.arange(em_lo, em_hi + 1e-6, step)
    cand = []
    for e, ex in enumerate(exs):
        grid = base + (step / 2 if (offset_ex and e == n_ex // 2) else 0.0)
        for em in grid[grid > ex + 30]:
            cand.append((e, em))
    cand = np.array(cand)
    must = set()
    for e in range(n_ex):  # every excitation keeps >= 4 measurements
        must.update(np.where(cand[:, 0] == e)[0][:4].tolist())
    rest = [i for i in rng.permutation(len(cand)).tolist() if i not in must]
    sel = must | set(rest[:max(0, min(n_features, len(cand)) - len(must))])
    sel = sorted(sel, key=lambda i: (cand[i, 0], cand[i, 1]))
    ex_nm = exs[cand[sel, 0].astype(int)]
    em_nm = cand[sel, 1]
    emissions = np.unique(em_nm)
    grid_index = np.stack([cand[sel, 0].astype(np.int64), np.searchsorted(emissions, em_nm)], 1)
    return dict(ex_nm=ex_nm, em_nm=em_nm, grid_index=grid_index, excitations=exs, emissions=emissions,
                n_features=len(sel), n_classes=n_classes)
