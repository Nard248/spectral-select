"""Tests for the RL selector (features, env, policy, PPO, search API) on synthetic low-rank tasks."""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from rl_select.env import (ADD, REMOVE, BudgetExhausted, EnvConfig, EpSpec, Episode, SelectEnv, Tracker,
                           VecSelectEnv, empty_value)
from rl_select.features import (TaskFeatures, candidate_features, global_features, residual_stats)
from rl_select.policy import SelectPolicy, obs_to_tensors
from rl_select.ppo import PPOConfig, collect_rollout, train
from rl_select.rl_search import (deploy_few_shot, deploy_label_free, rl_per_instance, train_policy)
from rl_select.types import Task

torch.set_num_threads(1)
np.seterr(all="ignore")   # Accelerate raises spurious FP flags in matmul


def make_task(F=60, seed=0, n_px=2000, n_classes=3, n_spec=6, name="syn/r0/f0") -> Task:
    rng = np.random.default_rng(seed)
    K = 8
    W = rng.normal(size=(K, F))
    y = rng.integers(0, n_classes, n_px)
    g = np.array([c * 2 + rng.integers(0, 2) for c in y])
    lat = rng.normal(size=(n_px, K))
    cm = rng.normal(size=(n_classes, K)) * 1.2
    lat += cm[y]
    X = lat @ W + 0.7 * rng.normal(size=(n_px, F))
    X = ((X - X.mean(0)) / X.std(0)).astype(np.float32)
    ex = np.repeat(np.arange(300, 300 + 20 * (F // 6), 20), 6)[:F].astype(float)
    em = ex + 40 + 15 * np.tile(np.arange(6), F // 6 + 1)[:F]
    return Task(name=name, dataset="syn", repeat=0, fold=0, Xtr=X, ytr=y.astype(np.int64), gtr=g.astype(np.int64),
                sub_idx=np.arange(n_px), Xte=X[:100], yte=y[:100].astype(np.int64), gte=g[:100].astype(np.int64),
                ex_nm=ex, em_nm=em, raw_logmean=rng.normal(size=F), raw_logstd=rng.random(F) + 0.1,
                n_classes=n_classes, valid=np.ones(F, bool))


class ToyObj:
    """Cached shrinkage-LDA balanced accuracy on the training pixels (toy stand-in for JSup)."""
    name = "toy"
    uses_labels = True

    def __init__(self, task: Task):
        self.X = task.Xtr.astype(np.float64)
        self.y = task.ytr
        self.C = task.n_classes
        self.n_evals = 0
        self._c: dict = {}
        self.mu = np.stack([self.X[self.y == c].mean(0) for c in range(self.C)])

    def __call__(self, S):
        k = frozenset(int(s) for s in S)
        if k in self._c:
            return self._c[k]
        s = np.array(sorted(k))
        self.n_evals += 1
        if len(s) == 0:
            v = 1.0 / self.C
        else:
            Xs = self.X[:, s]
            mu = self.mu[:, s]
            res = Xs - mu[self.y]
            Sw = res.T @ res / len(Xs)
            Sw = 0.9 * Sw + 0.1 * np.trace(Sw) / len(s) * np.eye(len(s))
            W = np.linalg.solve(Sw, mu.T)                          # [s,C]
            sc = Xs @ W - 0.5 * (mu.T * W).sum(0)
            pred = sc.argmax(1)
            v = float(np.mean([(pred[self.y == c] == c).mean() for c in range(self.C)]))
        self._c[k] = v
        return v

    def reset_counter(self):
        self.n_evals = 0


@pytest.fixture(scope="module")
def tasks():
    return [make_task(60, 0, name="syn/a"), make_task(48, 1, name="syn/b")]


@pytest.fixture(scope="module")
def pool(tasks):
    return [(t, ToyObj(t)) for t in tasks]


def tfs_for(pool, sup="none", tmp=None):
    return [TaskFeatures(t, o, sup=sup, oracle_dir=tmp) for t, o in pool]


def test_features_match_objective_module(tasks):
    from rl_select.objective import JRecon, JReconPC
    t = tasks[0]
    tf = TaskFeatures(t)
    jr, jp = JRecon(t), JReconPC(t)
    rng = np.random.default_rng(0)
    for _ in range(5):
        S = list(rng.choice(tf.F, 6, replace=False))
        assert np.isclose(tf.jrecon(S), jr(S), atol=1e-8)
        assert np.isclose(tf.jpc(S), jp(S), atol=1e-8)
        # gain / loss agree with brute-force differences of the objective
        st = residual_stats(tf.Sigma, tf.C, S)
        c_in, c_out = S[0], [c for c in range(tf.F) if c not in S][0]
        assert np.isclose(st["loss_rec"][c_in], jr(S) - jr([s for s in S if s != c_in]), atol=1e-7)
        assert np.isclose(st["gain_pc"][c_out], jp(S + [c_out]) - jp(S), atol=1e-7)
        assert np.isclose(st["gain_rec"][c_out], jr(S + [c_out]) - jr(S), atol=1e-7)
        from rl_select.objective import residual_stats as theirs
        th = theirs(tf.Sigma, S)
        assert np.allclose(th["resid_var"], st["rvar"], atol=1e-8)
        assert np.allclose(th["gain_add"], st["gain_rec"], atol=1e-8)
        assert np.allclose(th["loss_remove"], st["loss_rec"], atol=1e-8)


def test_feature_shapes_and_finite(tasks, tmp_path):
    t = tasks[0]
    for sup, d in (("none", 20), ("fisher", 21), ("oracle", 23)):
        tf = TaskFeatures(t, ToyObj(t), sup=sup, oracle_dir=tmp_path)
        m = np.zeros(tf.F, bool)
        m[[3, 10, 20]] = True
        f = candidate_features(tf, m, 0)
        assert f.shape == (tf.F, d) and tf.d == d and np.isfinite(f).all()
        g = global_features(tf, m, 1, 5, 2, 5)
        assert g.shape == (7,) and np.isfinite(g).all()
    assert (tmp_path / "syn_a__toy.npz").exists()


def cfg_variants():
    for mode, init in (("add", "empty"), ("addremove", "empty"), ("addremove", "css_greedy")):
        for reward in ("shaped", "terminal"):
            yield EnvConfig(mode=mode, init=init, reward=reward, dropout=0.2)


def run_random(venv, rng, check=None):
    steps = 0
    while len(venv.active()):
        obs = venv.observe()
        acts = []
        for k, i in enumerate(obs.env_idx):
            ep = venv.episodes[i]
            if obs.phase[k] == ADD:
                assert not obs.rem_mask[k].any() and not obs.stop_ok[k]
                legal = np.flatnonzero(obs.add_mask[k])
                assert not (obs.add_mask[k] & obs.sel[k]).any()          # no duplicate adds
                assert not (obs.add_mask[k] & ~ep.avail_padded(venv.Fmax)).any() if hasattr(ep, "avail_padded") else True
            else:
                assert not obs.add_mask[k].any() and obs.stop_ok[k]
                assert not (obs.rem_mask[k] & ~obs.sel[k]).any()         # only present pairs removable
                legal = np.append(np.flatnonzero(obs.rem_mask[k]), venv.Fmax)
            assert len(legal) > 0
            acts.append(rng.choice(legal))
        if check:
            check(venv, obs, acts)
        venv.step(obs.env_idx, np.array(acts))
        steps += 1
    return steps


def test_masks_and_final_sets(pool):
    tfs = tfs_for(pool)
    for cfg in cfg_variants():
        venv = VecSelectEnv([(tf, o) for tf, (_, o) in zip(tfs, pool)], cfg, seed=1)
        rng = np.random.default_rng(0)
        venv.reset(B=24)
        run_random(venv, rng)
        for e in venv.episodes:
            S = e.current_set()
            assert len(S) == e.n == len(set(S)) and min(S) >= 0 and max(S) < e.tf.F
            assert all(e.avail[s] or cfg.init == "css_greedy" for s in S)
            assert np.isfinite(e.final_J)


def test_padding_two_F(pool):
    tfs = tfs_for(pool)
    venv = VecSelectEnv([(tf, o) for tf, (_, o) in zip(tfs, pool)], EnvConfig(mode="addremove"), seed=0)
    assert venv.Fmax == 60
    venv.reset([EpSpec(0, 5), EpSpec(1, 5), EpSpec(1, 3)])
    obs = venv.observe()
    assert obs.cand.shape == (3, 60, 20)
    assert not obs.add_mask[1, 48:].any() and not obs.sel[1, 48:].any() and (obs.cand[1, 48:] == 0).all()
    pol = SelectPolicy(20, 7)
    logits, val = pol(*obs_to_tensors(obs))
    assert logits.shape == (3, 61) and val.shape == (3,)
    assert (logits[1, 48:60] < -1e8).all()
    ro = None
    torch.manual_seed(0)
    ro = collect_rollout(pol, venv)
    for e in venv.episodes:
        assert len(set(e.current_set())) == e.n
    assert pol.n_params() < 60000


def test_tabu_and_no_undo(pool):
    tf = tfs_for(pool)[0]
    obj = pool[0][1]
    cfg = EnvConfig(mode="addremove", dropout=0.0, tabu_L=3, ns=(5,))
    rng = np.random.default_rng(3)
    for trial in range(10):
        ep = Episode(tf, obj, 5, cfg, rng)
        removed_at = {}
        added_at = {}
        while not ep.done:
            if ep.phase == ADD:
                am = ep.add_mask()
                for c, cyc in removed_at.items():
                    if ep.cycle <= cyc + 3:
                        assert not am[c], "tabu violated"
                a = int(rng.choice(np.flatnonzero(am)))
                was_constructing = ep.constructing
                ep.step(a)
                if not was_constructing:
                    added_at[ep.cycle - 1] = a
            else:
                rm = ep.remove_mask()
                if ep.cycle - 1 in added_at:
                    assert not rm[added_at[ep.cycle - 1]], "just-added pair removable"
                a = int(rng.choice(np.flatnonzero(rm)))
                removed_at[a] = ep.cycle
                ep.step(a)
        assert ep.cycle <= 5


def test_reward_telescopes(pool):
    tfs = tfs_for(pool)
    for cfg in cfg_variants():
        venv = VecSelectEnv([(tf, o) for tf, (_, o) in zip(tfs, pool)], cfg, seed=5)
        venv.reset(B=16)
        rng = np.random.default_rng(1)
        run_random(venv, rng)
        for e in venv.episodes:
            Jf = e.obj(e.current_set())
            expected = Jf - e.J0 - cfg.repeat_penalty * e.n_penalties
            assert np.isclose(e.return_, expected, atol=1e-9), (cfg, e.return_, expected)
    # empty value is chance
    assert np.isclose(empty_value(pool[0][1], 3), 1 / 3)


def test_label_free_deploy_no_objective_calls(pool):
    t, o = pool[0]
    pol = SelectPolicy(20, 7, meta=dict(sup="none"))
    n0 = o.n_evals
    for mode, init in (("add", "empty"), ("addremove", "empty"), ("addremove", "css_greedy")):
        r = deploy_label_free(pol, t, 5, mode=mode, init=init)
        assert len(r.selected) == 5 == len(set(r.selected)) and r.n_evals == 0
    r = deploy_label_free(pol, t, 5, mode="addremove", n_samples=8, seed=1)
    assert len(set(r.selected)) == 5 and r.method.endswith("best8")
    assert o.n_evals == n0
    with pytest.raises(ValueError):
        deploy_label_free(SelectPolicy(21, 7, meta=dict(sup="fisher")), t, 5)


def test_few_shot_budget(tasks):
    t = tasks[0]
    for mode, mc in (("add", False), ("addremove", False), ("addremove", True)):
        obj = ToyObj(t)
        pol = SelectPolicy(21, 7, meta=dict(sup="fisher"))
        r = deploy_few_shot(pol, t, 5, obj, budget=60, mode=mode, mc_lookahead=mc)
        assert len(r.selected) == 5 == len(set(r.selected))
        assert r.n_evals <= 60 and obj.n_evals <= 61
        assert np.isclose(r.inner_J, obj(r.selected)) and r.trace[-1][1] == r.inner_J


def test_per_instance_budget(tasks):
    t = tasks[0]
    obj = ToyObj(t)
    r = rl_per_instance(t, 5, obj, mode="addremove", budget=150, seed=0, n_envs=8)
    assert r.n_evals <= 150 and obj.n_evals <= 151
    assert len(set(r.selected)) == 5
    assert all(a[0] <= b[0] and a[1] < b[1] for a, b in zip(r.trace, r.trace[1:]))


def test_ppo_improves_over_random_and_is_deterministic(pool, tmp_path):
    t = pool[0][0]
    n = 4
    t0 = time.time()
    pol = train_policy(pool, mode="add", sup_features=True, seed=0, n_updates=40, n_envs=32, ns=(n,),
                       dropout=0.0, out_dir=tmp_path / "run")
    assert time.time() - t0 < 150
    log = pol.train_info["log"]
    first, last = np.mean([r["mean_final_J"] for r in log[:3]]), np.mean([r["mean_final_J"] for r in log[-3:]])
    assert last > first + 0.02, (first, last)
    # versus random subsets (greedy deployment, objective observable only to score)
    obj = pool[0][1]
    rng = np.random.default_rng(0)
    rand = np.mean([obj(list(rng.choice(60, n, replace=False))) for _ in range(200)])
    r = deploy_few_shot(pol, t, n, ToyObj(t), budget=1, mode="add")
    assert r.inner_J > rand
    assert (tmp_path / "run" / "policy.pt").exists() and (tmp_path / "run" / "train_log.csv").exists()
    pol2 = train_policy(pool, mode="add", sup_features=True, seed=0, n_updates=3, n_envs=32, ns=(n,), dropout=0.0)
    pol3 = train_policy(pool, mode="add", sup_features=True, seed=0, n_updates=3, n_envs=32, ns=(n,), dropout=0.0)
    for a, b in zip(pol2.parameters(), pol3.parameters()):
        assert torch.equal(a, b)
    loaded = SelectPolicy.load(tmp_path / "run")
    for a, b in zip(pol.parameters(), loaded.parameters()):
        assert torch.equal(a, b)


def test_reinforce_and_addremove_train_run(pool):
    pol = train_policy(pool, mode="addremove", algo="reinforce", seed=0, n_updates=3, n_envs=16, ns=(3, 5))
    assert pol.train_info["updates"] == 3
    pol = train_policy(pool, mode="addremove", init="css_greedy", reward="terminal", seed=0, n_updates=3,
                       n_envs=16, ns=(3, 5))
    assert pol.train_info["updates"] == 3
