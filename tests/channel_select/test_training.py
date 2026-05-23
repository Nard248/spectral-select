import torch
from channel_select.models.temporal import TemporalGroupedAutoencoder
from channel_select.models.training import train_autoencoder


def test_loss_decreases_on_toy_data():
    torch.manual_seed(0)
    channels = {"hand": 3}
    model = TemporalGroupedAutoencoder(channels, time_len=16, latent_dim=4)
    data = {"hand": torch.randn(20, 16, 3)}
    history = train_autoencoder(model, data, epochs=15, lr=1e-2, batch_size=8, device="cpu")
    assert history[-1] < history[0]
