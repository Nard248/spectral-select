import torch
from channel_select.models.temporal import TemporalGroupedAutoencoder
from channel_select.protocols import GroupStructuredModel


def test_encode_decode_shapes_roundtrip():
    channels = {"hand": 3, "ankle": 2}
    model = TemporalGroupedAutoencoder(channels_per_group=channels, time_len=32, latent_dim=8)
    batch = {"hand": torch.randn(4, 32, 3), "ankle": torch.randn(4, 32, 2)}
    latent = model.encode(batch)
    assert latent.shape[0] == 4
    recon = model.decode(latent)
    assert recon["hand"].shape == (4, 32, 3)
    assert recon["ankle"].shape == (4, 32, 2)


def test_conforms_to_protocol():
    model = TemporalGroupedAutoencoder({"hand": 3}, time_len=16, latent_dim=4)
    assert isinstance(model, GroupStructuredModel)
    assert model.groups == ["hand"]
    assert model.channels_per_group == {"hand": 3}
