"""SpectraForge — chemical-level synthetic ME-HSI dataset generator.

Define fluorophores and materials, paint them onto a scene, and render a physically
grounded multi-excitation hyperspectral cube (spectral_select.SpectraData) plus the
ground truth needed to validate band-selection methods.
"""
from spectraforge.fluorophore import Fluorophore

__all__ = ["Fluorophore"]
