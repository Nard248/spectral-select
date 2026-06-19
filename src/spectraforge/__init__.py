"""SpectraForge — chemical-level synthetic ME-HSI dataset generator.

Define fluorophores and materials, paint them onto a scene, and render a physically
grounded multi-excitation hyperspectral cube (spectral_select.SpectraData) plus the
ground truth needed to validate band-selection methods.
"""
from spectraforge.acquisition import AcquisitionConfig
from spectraforge.artifacts import ArtifactConfig
from spectraforge.fluorophore import Fluorophore
from spectraforge.forward import render
from spectraforge.groundtruth import GroundTruth
from spectraforge.library import load_builtin_library
from spectraforge.material import Material
from spectraforge.physics import PhysicsConfig
from spectraforge.scene import Scene

__all__ = [
    "Fluorophore",
    "Material",
    "Scene",
    "AcquisitionConfig",
    "ArtifactConfig",
    "PhysicsConfig",
    "GroundTruth",
    "render",
    "load_builtin_library",
]
