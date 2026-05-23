"""Conv1d-over-time group-structured autoencoder.

Mirrors the HSI HyperspectralCAEWithMasking structure: per-group encoders ->
mean fusion -> shared latent -> per-group decoders. The spatial conv is replaced
by a temporal conv; the group axis plays the role of excitation.
"""
from __future__ import annotations
from typing import Hashable

import torch
import torch.nn as nn


def _key(g: Hashable) -> str:
    return str(g).replace(".", "_").replace(" ", "_")


class TemporalGroupedAutoencoder(nn.Module):
    def __init__(self, channels_per_group: dict[Hashable, int], time_len: int,
                 latent_dim: int = 16, hidden: int = 32) -> None:
        super().__init__()
        self.groups = list(channels_per_group.keys())
        self.channels_per_group = dict(channels_per_group)
        self.time_len = time_len
        self.latent_dim = latent_dim

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for g in self.groups:
            cin = channels_per_group[g]
            self.encoders[_key(g)] = nn.Sequential(
                nn.Conv1d(cin, hidden, kernel_size=5, padding=2), nn.ReLU(),
                nn.Conv1d(hidden, latent_dim, kernel_size=5, padding=2), nn.ReLU(),
            )
            self.decoders[_key(g)] = nn.Sequential(
                nn.Conv1d(latent_dim, hidden, kernel_size=5, padding=2), nn.ReLU(),
                nn.Conv1d(hidden, cin, kernel_size=5, padding=2),
            )

    def encode(self, batch: dict[Hashable, torch.Tensor]) -> torch.Tensor:
        feats = []
        for g in self.groups:
            x = batch[g].permute(0, 2, 1)            # (B, time, ch) -> (B, ch, time)
            feats.append(self.encoders[_key(g)](x))  # (B, latent_dim, time)
        stacked = torch.stack(feats, dim=1)          # (B, n_groups, latent_dim, time)
        return torch.mean(stacked, dim=1)            # (B, latent_dim, time) fusion

    def decode(self, latent: torch.Tensor) -> dict[Hashable, torch.Tensor]:
        out = {}
        for g in self.groups:
            y = self.decoders[_key(g)](latent)       # (B, ch, time)
            out[g] = y.permute(0, 2, 1)              # (B, time, ch)
        return out

    def forward(self, batch):
        return self.decode(self.encode(batch))
