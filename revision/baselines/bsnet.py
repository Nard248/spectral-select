"""BS-Net-FC - Band Selection Network (fully-connected variant).

Cai, Liu, Chanussot (2020). "BS-Nets: An End-to-End Framework for Band
Selection of Hyperspectral Image." IEEE TGRS.

Architecture (FC variant):
    Input x in R^B  ->  attention subnet outputs per-sample weights w
                        (each in [0,1], same shape as x).
    x_tilde = w * x  ->  reconstruction subnet  ->  x_hat.

Loss: ||x - x_hat||^2 + lam * mean(w)
Selection: top-K bands by population-mean attention weight.

Lightweight implementation, suitable for local MPS / CPU.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


if _HAS_TORCH:
    class _BSNetFC(nn.Module):
        def __init__(self, n_bands: int, hidden: int = 64):
            super().__init__()
            self.attn = nn.Sequential(
                nn.Linear(n_bands, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, n_bands), nn.Sigmoid(),
            )
            self.recon = nn.Sequential(
                nn.Linear(n_bands, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, n_bands),
            )

        def forward(self, x):
            w = self.attn(x)
            x_tilde = w * x
            x_hat = self.recon(x_tilde)
            return x_hat, w


def select_bsnet_fc(features: np.ndarray, K: int, *,
                    seed: int = 0,
                    epochs: int = 30,
                    batch_size: int = 256,
                    lr: float = 1e-3,
                    lam_sparsity: float = 0.01,
                    device: str | None = None,
                    verbose: bool = False, **_):
    """BS-Net-FC selection."""
    if not _HAS_TORCH:
        raise ImportError("BS-Net requires PyTorch")

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    F = features.astype(np.float32)
    # Per-band min-max scaling so attention sees comparable input dynamics
    fmin = F.min(axis=0, keepdims=True)
    fmax = F.max(axis=0, keepdims=True)
    span = fmax - fmin
    span[span < 1e-8] = 1.0
    F_scaled = (F - fmin) / span

    N, B = F_scaled.shape
    model = _BSNetFC(B).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(N)[: min(N, 50000)]
    X = torch.from_numpy(F_scaled[train_idx]).to(device)
    n_train = X.shape[0]

    model.train(True)
    for ep in range(epochs):
        perm = torch.randperm(n_train, device=device)
        total = 0.0
        n_batches = 0
        for i in range(0, n_train, batch_size):
            batch = X[perm[i:i + batch_size]]
            x_hat, w = model(batch)
            loss = mse(x_hat, batch) + lam_sparsity * w.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n_batches += 1
        if verbose and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"  BS-Net ep {ep+1}/{epochs}: loss={total/n_batches:.4f}")

    # Inference mode: average attention weight per band
    model.train(False)
    with torch.no_grad():
        _, w = model(X)
        importance = w.mean(dim=0).cpu().numpy()

    ranking = np.argsort(-importance)
    return np.asarray(ranking[:K], dtype=int)
