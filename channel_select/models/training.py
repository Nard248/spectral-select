from __future__ import annotations
from typing import Hashable

import torch
import torch.nn.functional as F


def train_autoencoder(model, data: dict[Hashable, torch.Tensor], epochs: int = 25,
                      lr: float = 1e-3, batch_size: int = 32, device: str = "cpu") -> list[float]:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = next(iter(data.values())).shape[0]
    history: list[float] = []
    for _ in range(epochs):
        perm = torch.randperm(n)
        epoch_loss, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = {g: t.index_select(0, idx).to(device) for g, t in data.items()}
            opt.zero_grad()
            recon = model(batch)
            loss = sum(F.mse_loss(recon[g], batch[g]) for g in batch)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            nb += 1
        history.append(epoch_loss / max(nb, 1))
    return history
