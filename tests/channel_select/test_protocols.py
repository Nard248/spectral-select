import pytest
from channel_select.protocols import SelectionConfig


def test_selection_config_defaults():
    cfg = SelectionConfig()
    assert cfg.dimension_selection_method == "variance"
    assert cfg.n_important_dimensions == 50
    assert cfg.perturbation_method == "percentile"
    assert cfg.perturbation_magnitudes == [10, 20, 30]
    assert cfg.normalization_method == "variance"
    assert cfg.n_channels_to_select == 10
    assert cfg.diversity_method == "mmr"
    assert cfg.lambda_diversity == 0.5


def test_selection_config_rejects_bad_method():
    with pytest.raises(ValueError):
        SelectionConfig(normalization_method="bogus")
