"""Reduced thermal models with auditable discrete error bounds."""

__version__ = "0.1.0"

from .model import Parameters, ThermalPlate
from .reduction import Prediction, ReducedModel, fit_pod

__all__ = ["Parameters", "Prediction", "ReducedModel", "ThermalPlate", "fit_pod"]
