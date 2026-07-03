"""
Compositelab: A Python library for composite materials analysis.

This library provides tools for:
- Material property definitions (orthotropic, isotropic)
- Classical Lamination Theory (CLT)
- Finite Element Analysis (FEM)
- Failure criteria (Tsai-Wu, Hashin, etc.)
"""

from . import materials

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    'materials',
]
