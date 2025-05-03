"""
English to Gloss Translation Package
===================================

A package for translating English text to sign language gloss notation.
"""

from .model import load_model, translate_to_gloss
from .evaluate import evaluate_model, calculate_metrics

__version__ = "0.1.0"
__all__ = ["load_model", "translate_to_gloss", "evaluate_model", "calculate_metrics"]