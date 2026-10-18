"""PPO and REINFORCE trainers for the subset-selection policy."""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.distributions import Categorical

from .env import BudgetExhausted, VecSelectEnv
from .policy import SelectPolicy, obs_to_tensors


@dataclass
class PPOConfig:
    algo: str = "ppo"            # ppo | reinforce
    n_envs: int = 64
    clip: float = 0.2
    lam: float = 0.95
    gamma: float = 1.0
    ent_start: float = 0.01
    ent_end: float = 0.001
    lr: float = 3e-4
    grad_clip: float = 0.5
    epochs: int = 4
    minibatch: int = 256
    vf_coef: float = 0.5


@dataclass
class Rollout:
    obs: tuple            # stacked tensors (cand, glob, sel, add, rem, stop, phase)
    actions: torch.Tensor
    logp: torch.Tensor
    values: torch.Tensor
    rewards: np.ndarray
    env_idx: np.ndarray
    step_in_ep: np.ndarray
    ep_returns: np.ndarray
    ep_final_J: np.ndarray
    ep_len: np.ndarray


def collect_rollout(policy: SelectPolicy, venv: VecSelectEnv, greedy: bool = False) -> Rollout:
    """Run all episodes in ``venv`` (already reset) to completion with the current policy."""
    obs_l, act_l, lp_l, v_l, r_l, e_l, s_l = [], [], [], [], [], [], []
    t = 0
    while len(venv.active()):
        obs = venv.observe()
        tens = obs_to_tensors(obs)
        with torch.no_grad():
            logits, val = policy(*tens)
            dist = Categorical(logits=logits)
            a = logits.argmax(-1) if greedy else dist.sample()
            lp = dist.log_prob(a)
        rew, _ = venv.step(obs.env_idx, a.numpy())
        obs_l.append(tens); act_l.append(a); lp_l.append(lp); v_l.append(val)
        r_l.append(rew); e_l.append(obs.env_idx); s_l.append(np.full(len(rew), t))
        t += 1
    stacked = tuple(torch.cat([o[i] for o in obs_l]) for i in range(7))
    eps = venv.episodes
    return Rollout(stacked, torch.cat(act_l), torch.cat(lp_l), torch.cat(v_l), np.concatenate(r_l),
                   np.concatenate(e_l), np.concatenate(s_l),
                   np.array([e.return_ for e in eps]), np.array([e.final_J for e in eps], dtype=float),
                   np.array([e.n_steps for e in eps]))


def compute_gae(ro: Rollout, gamma: float, lam: float):
    """Per-episode GAE (episodes end terminally, V_next = 0). Returns (advantages, returns)."""
    order = np.argsort(ro.env_idx, kind="stable")
    env_sorted = ro.env_idx[order]
    adv = np.zeros(len(order), np.float32)
    v = ro.values.numpy()[order]
    r = ro.rewards[order]
    starts = np.flatnonzero(np.r_[True, env_sorted[1:] != env_sorted[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        last = 0.0
        for k in range(e - 1, s - 1, -1):
            nv = v[k + 1] if k + 1 < e else 0.0
            delta = r[k] + gamma * nv - v[k]
            last = delta + gamma * lam * last
            adv[k] = last
    A = np.zeros(len(order), np.float32)
    A[order] = adv
    return A, A + ro.values.numpy()


def reinforce_advantages(ro: Rollout, gamma: float):
    """Returns-to-go minus a batch-mean baseline taken over steps with the same within-episode index."""
    order = np.argsort(ro.env_idx, kind="stable")
    env_sorted = ro.env_idx[order]
    r = ro.rewards[order]
    G = np.zeros(len(order), np.float32)
    starts = np.flatnonzero(np.r_[True, env_sorted[1:] != env_sorted[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        acc = 0.0
        for k in range(e - 1, s - 1, -1):
            acc = r[k] + gamma * acc
            G[k] = acc
    Gu = np.zeros(len(order), np.float32)
    Gu[order] = G
    b = np.zeros_like(Gu)
    for t in np.unique(ro.step_in_ep):
        m = ro.step_in_ep == t
        b[m] = Gu[m].mean()
    return Gu - b


def ppo_update(policy, opt, ro: Rollout, cfg: PPOConfig, ent_coef: float, gen: torch.Generator) -> dict:
    N = len(ro.rewards)
    if cfg.algo == "ppo":
        adv_np, ret_np = compute_gae(ro, cfg.gamma, cfg.lam)
        adv = torch.from_numpy(adv_np)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = torch.from_numpy(ret_np)
        epochs, mb = cfg.epochs, cfg.minibatch
    elif cfg.algo == "reinforce":
        adv = torch.from_numpy(reinforce_advantages(ro, cfg.gamma))
        ret = None
        epochs, mb = 1, N
    else:
        raise ValueError(f"unknown algo {cfg.algo}")
    ents, losses = [], []
    for _ in range(epochs):
        perm = torch.randperm(N, generator=gen)
        for s in range(0, N, mb):
            ix = perm[s:s + mb]
            tens = tuple(o[ix] for o in ro.obs)
            logits, val = policy(*tens)
            dist = Categorical(logits=logits)
            lp = dist.log_prob(ro.actions[ix])
            ent = dist.entropy().mean()
            if cfg.algo == "ppo":
                ratio = (lp - ro.logp[ix]).exp()
                a = adv[ix]
                pl = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                vl = ((val - ret[ix]) ** 2).mean()
                loss = pl + cfg.vf_coef * vl - ent_coef * ent
            else:
                loss = -(lp * adv[ix]).mean() - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt.step()
            ents.append(float(ent)); losses.append(float(loss))
    return dict(entropy=float(np.mean(ents)), loss=float(np.mean(losses)))


def train(policy: SelectPolicy, venv: VecSelectEnv, cfg: PPOConfig, seed: int, n_updates: int | None = None,
          max_seconds: float | None = None, out_dir: str | Path | None = None,
          progress_fn: Callable[[], float] | None = None, stop_fn: Callable[[], bool] | None = None,
          log_every_ckpt: int = 0, verbose: bool = False) -> dict:
    """Train ``policy`` on episodes sampled from ``venv``'s pool. Stops at n_updates / max_seconds /
    stop_fn() / BudgetExhausted (raised by trackers). Writes train_log.csv + policy checkpoint."""
    if n_updates is None and max_seconds is None and stop_fn is None:
        raise ValueError("give n_updates, max_seconds or stop_fn")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    venv.rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    t0 = time.time()
    out = Path(out_dir) if out_dir else None
    writer = None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        f = open(out / "train_log.csv", "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["update", "episodes", "steps", "mean_return", "mean_final_J", "entropy", "loss", "seconds"])
    log = []
    steps = eps = 0
    u = 0
    stopped = "n_updates"
    while True:
        if n_updates is not None and u >= n_updates:
            break
        if max_seconds is not None and time.time() - t0 >= max_seconds:
            stopped = "max_seconds"
            break
        if stop_fn is not None and stop_fn():
            stopped = "stop_fn"
            break
        prog = 0.0
        if progress_fn is not None:
            prog = progress_fn()
        elif n_updates is not None:
            prog = u / n_updates
        elif max_seconds is not None:
            prog = (time.time() - t0) / max_seconds
        ent_coef = cfg.ent_start + (cfg.ent_end - cfg.ent_start) * min(max(prog, 0.0), 1.0)
        venv.reset(B=cfg.n_envs)
        try:
            ro = collect_rollout(policy, venv)
        except BudgetExhausted:
            stopped = "budget"
            break
        st = ppo_update(policy, opt, ro, cfg, ent_coef, gen)
        steps += len(ro.rewards); eps += cfg.n_envs
        row = dict(update=u, episodes=eps, steps=steps, mean_return=float(ro.ep_returns.mean()),
                   mean_final_J=float(np.nanmean(ro.ep_final_J)), entropy=st["entropy"], loss=st["loss"],
                   seconds=time.time() - t0)
        log.append(row)
        if writer:
            writer.writerow(list(row.values())); f.flush()
        if verbose and u % 10 == 0:
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        u += 1
        if out and log_every_ckpt and u % log_every_ckpt == 0:
            policy.save(out, extra=dict(ppo=asdict(cfg), seed=seed, updates=u))
    if writer:
        f.close()
    if out:
        policy.save(out, extra=dict(ppo=asdict(cfg), seed=seed, updates=u, stopped=stopped))
    return dict(log=log, updates=u, steps=steps, episodes=eps, seconds=time.time() - t0, stopped=stopped)
