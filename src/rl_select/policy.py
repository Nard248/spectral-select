"""Set-based actor-critic for subset selection (shared per-candidate encoder, masked categorical)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .env import Obs

NEG = -1e9


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class SelectPolicy(nn.Module):
    """phi(c, S) -> h; ctx = MLP([mean_S h, max_S h, global]); add/remove heads on [h_c, ctx]; STOP and value on ctx.

    Works with padded candidate axes: padded entries are masked out of pooling and of the action
    distribution, so tasks with different F can share a batch.
    """

    def __init__(self, d: int, g: int, hidden: int = 64, meta: dict | None = None):
        super().__init__()
        self.d, self.g, self.hidden = d, g, hidden
        self.meta = dict(meta or {})
        self.enc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.ctx = nn.Sequential(nn.Linear(2 * hidden + g, hidden), nn.ReLU())
        self.add_head = _mlp(2 * hidden, hidden, 1)
        self.rem_head = _mlp(2 * hidden, hidden, 1)
        self.stop_head = _mlp(hidden, 32, 1)
        self.value_head = _mlp(hidden, 64, 1)

    def forward(self, cand, glob, sel, add_mask, rem_mask, stop_ok, phase):
        """Returns (masked logits [A, Fmax+1] with STOP last, value [A])."""
        h = self.enc(cand)                                    # [A,F,H]
        selk = sel.unsqueeze(-1)
        cnt = sel.sum(1, keepdim=True).clamp(min=1).to(h.dtype)
        mean = (h * selk).sum(1) / cnt
        mx = h.masked_fill(~selk, -1e9).max(1).values
        mx = torch.where(sel.any(1, keepdim=True), mx, torch.zeros_like(mx))
        ctx = self.ctx(torch.cat([mean, mx, glob], -1))      # [A,H]
        hc = torch.cat([h, ctx.unsqueeze(1).expand(-1, h.shape[1], -1)], -1)
        la = self.add_head(hc).squeeze(-1)
        lr = self.rem_head(hc).squeeze(-1)
        is_add = (phase == 0).unsqueeze(-1)
        cl = torch.where(is_add, la, lr)
        mask = torch.where(is_add, add_mask, rem_mask)
        cl = cl.masked_fill(~mask, NEG)
        sl = self.stop_head(ctx).masked_fill(~stop_ok.unsqueeze(-1), NEG)
        return torch.cat([cl, sl], -1), self.value_head(ctx).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, out_dir: str | Path, extra: dict | None = None) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out / "policy.pt")
        cfg = dict(d=self.d, g=self.g, hidden=self.hidden, meta=self.meta, **(extra or {}))
        (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))

    @classmethod
    def load(cls, out_dir: str | Path) -> "SelectPolicy":
        out = Path(out_dir)
        cfg = json.loads((out / "config.json").read_text())
        p = cls(cfg["d"], cfg["g"], cfg["hidden"], cfg.get("meta"))
        p.load_state_dict(torch.load(out / "policy.pt", weights_only=True))
        return p


def obs_to_tensors(obs: Obs) -> tuple:
    return (torch.from_numpy(obs.cand), torch.from_numpy(obs.glob), torch.from_numpy(obs.sel),
            torch.from_numpy(obs.add_mask), torch.from_numpy(obs.rem_mask), torch.from_numpy(obs.stop_ok),
            torch.from_numpy(obs.phase))


@torch.no_grad()
def action_probs(policy: SelectPolicy, obs: Obs) -> np.ndarray:
    logits, _ = policy(*obs_to_tensors(obs))
    return torch.softmax(logits, -1).numpy()
