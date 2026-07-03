"""Layer module for defining composite layers."""

import numpy as np
from typing import Optional
from compositelab.materials.base import Material

class Layer:
    """
    Represents a single layer in a composite laminate.
    
    Parameters
    ----------
    material : Material
        Material object (Orthotropic2D, Isotropic, etc.)
    thickness : float
        Layer thickness in meters (m)
    angle : float, optional
        Fiber orientation angle in degrees (default: 0.0)
    name : str, optional
        Layer identifier
    """
    
    _id_counter = 0
    
    def __init__(
        self,
        material: Material,
        thickness: float,
        angle: float = 0.0,
        name: Optional[str] = None
    ):
        if thickness <= 0:
            raise ValueError("Thickness must be positive")
        
        Layer._id_counter += 1
        self.id = Layer._id_counter
        self.material = material
        self.thickness = thickness  # meters
        self.angle = angle  # degrees
        self.name = name or f"Layer_{self.id}"
    
    @property
    def Q_bar(self):
        """Rotated reduced stiffness matrix (3x3) of this layer in laminate axes."""
        mat = self.material
        # Orthotropic (2D or 3D): has an explicit rotation method
        if hasattr(mat, "get_rotated_stiffness_matrix"):
            return mat.get_rotated_stiffness_matrix(self.angle)
        # Isotropic: rotation-invariant, use plane-stress reduced Q directly
        if hasattr(mat, "stiffness_matrix_2d"):
            return mat.stiffness_matrix_2d("plane_stress")
        # Fallback: material already exposes a 3x3 reduced stiffness
        Q = np.asaray(mat.stiffness_matrix)
        if Q.shape == (3, 3):
            return Q
        raise TypeError(
            f"Cannot build 3x3 Q_bar for material {type(mat).__name__}"
        )


    
    def __repr__(self) -> str:
        # Display in mm for readability, store in meters
        return (
            f"Layer(material={self.material.name}, "
            f"thickness={self.thickness*1000:.4f}mm, "
            f"angle={self.angle}°)"
        )
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Layer):
            return False
        return (
            self.material == other.material and
            np.isclose(self.thickness, other.thickness) and
            np.isclose(self.angle, other.angle)
        )
