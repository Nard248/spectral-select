"""SOTA hyperspectral band-selection baselines for the spectral-select revision.

Each baseline is a function with the unified signature:

    def select(features: np.ndarray, K: int, *,
               labels: np.ndarray | None = None,
               seed: int = 0,
               **kwargs) -> np.ndarray:
        '''Returns a length-K array of selected band indices.'''

This common interface lets the runner sweep methods x K x seeds with one loop.

Methods provided:
    classical:  variance, pca_loading, sam_greedy, spa, mcuve, random
    deep:       bsnet_fc  (Cai et al. 2020)
    supervised: sparse_lasso
    cluster:    issc
    project:    ae_perturb (used via spectral_select.Analyzer)
"""
from .classical import (
    select_variance,
    select_pca_loading,
    select_sam_greedy,
    select_spa,
    select_mcuve,
    select_random,
)
from .bsnet import select_bsnet_fc
from .sparse_lasso import select_sparse_lasso
from .issc import select_issc
from .ae_perturb_cached import select_ae_perturb

ALL_METHODS = {
    "variance": select_variance,
    "pca_loading": select_pca_loading,
    "sam_greedy": select_sam_greedy,
    "spa": select_spa,
    "mcuve": select_mcuve,
    "issc": select_issc,
    "bsnet_fc": select_bsnet_fc,
    "sparse_lasso": select_sparse_lasso,
    "random": select_random,
    "ae_perturb": select_ae_perturb,
}

METHOD_TYPE = {
    "variance": "filter",
    "pca_loading": "filter",
    "sam_greedy": "wrapper-unsup",
    "spa": "wrapper-unsup",
    "mcuve": "wrapper-unsup",
    "issc": "clustering-unsup",
    "bsnet_fc": "deep-unsup",
    "sparse_lasso": "embedded-sup",
    "random": "baseline",
    "ae_perturb": "ours",
}

SUPERVISED = {"sparse_lasso"}
