"""Base material class for compositelab package."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from itertools import count


class Material(ABC):
    """Abstract base class for all material types.
    
    This class defines the common interface that all material models
    must implement. It provides automatic ID generation and handles
    common material properties.
    
    Attributes:
        id: Unique material identifier (auto-generated)
        name: Material name
        density: Material density in kg/m³ (optional)
    """
    
    _id_counter = count(1)
    
    def __init__(
        self,
        name: Optional[str] = None,
        density: Optional[float] = None,
    ):
        """Initialize base material properties.
        
        Args:
            name: Material name (default: "Material_{id}")
            density: Material density in kg/m³ (optional)
        """
        self.id = next(self._id_counter)
        self.name = name or f"Material_{self.id}"
        self.density = density
    
    @property
    @abstractmethod
    def stiffness_matrix(self) -> np.ndarray:
        """Returns the stiffness matrix (Q or C).
        
        Returns:
            Stiffness matrix - shape (3,3) for 2D, (6,6) for 3D
        """
        pass
    
    @property
    @abstractmethod
    def compliance_matrix(self) -> np.ndarray:
        """Returns the compliance matrix (S).
        
        Returns:
            Compliance matrix - shape (3,3) for 2D, (6,6) for 3D
        """
        pass
    
    @abstractmethod
    def __eq__(self, other) -> bool:
        """Check equality between materials."""
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the material."""
        pass
