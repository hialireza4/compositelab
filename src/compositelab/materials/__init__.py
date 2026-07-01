"""Materials module for compositelab package.

This module provides material models for composite and structural analysis,
including orthotropic, isotropic, and transversely isotropic materials.

Available material classes:
    - Material: Abstract base class for all materials
    - Orthotropic2D: 2D orthotropic material (plane stress)
    - Orthotropic3D: 3D orthotropic material
    - Isotropic: Isotropic material (metals, etc.)
    - TransverselyIsotropic: Transversely isotropic material
"""

from .base import Material
from .orthotropic import Orthotropic2D, Orthotropic3D
from .isotropic import Isotropic, TransverselyIsotropic

__all__ = [
    "Material",
    "Orthotropic2D",
    "Orthotropic3D",
    "Isotropic",
    "TransverselyIsotropic",
]

__version__ = "0.1.0"
