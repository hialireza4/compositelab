"""Isotropic material models for compositelab package."""

from typing import Optional
import numpy as np
from functools import cached_property

from .base import Material


class Isotropic(Material):
    """Isotropic material with identical properties in all directions.
    
    This class represents materials like metals, where mechanical properties
    are independent of direction. Can be used in both 2D (plane stress/strain)
    and 3D analysis.
    
    Attributes:
        E: Young's modulus (Pa)
        nu: Poisson's ratio
        G: Shear modulus (Pa) - computed automatically if not provided
    """
    
    def __init__(
        self,
        E: float,
        nu: float,
        G: Optional[float] = None,
        name: Optional[str] = None,
        density: Optional[float] = None,
        yield_strength: Optional[float] = None,
        ultimate_strength: Optional[float] = None,
    ):
        """Initialize isotropic material.
        
        Args:
            E: Young's modulus (Pa)
            nu: Poisson's ratio
            G: Shear modulus (Pa). If None, computed as E/(2(1+nu))
            name: Material name
            density: Material density (kg/m³)
            yield_strength: Yield strength (Pa)
            ultimate_strength: Ultimate tensile strength (Pa)
        """
        super().__init__(name=name, density=density)
        
        self.E = E
        self.nu = nu
        self._G = G
        
        # Strength properties (optional)
        self.yield_strength = yield_strength
        self.ultimate_strength = ultimate_strength
    
    @property
    def G(self) -> float:
        """Shear modulus (computed from E and nu if not provided)."""
        if self._G is not None:
            return self._G
        return self.E / (2 * (1 + self.nu))
    
    @property
    def K(self) -> float:
        """Bulk modulus."""
        return self.E / (3 * (1 - 2*self.nu))
    
    @property
    def lambda_(self) -> float:
        """Lamé's first parameter."""
        return self.E * self.nu / ((1 + self.nu) * (1 - 2*self.nu))
    
    @cached_property
    def stiffness_matrix(self) -> np.ndarray:
        """Returns 3D stiffness matrix C (6×6) in Voigt notation.
        
        For 2D plane stress, extract top-left 3×3 submatrix.
        
        Returns:
            Stiffness matrix C (6×6)
        """
        lam = self.lambda_
        G = self.G
        
        C = np.array([
            [lam + 2*G, lam,       lam,       0, 0, 0],
            [lam,       lam + 2*G, lam,       0, 0, 0],
            [lam,       lam,       lam + 2*G, 0, 0, 0],
            [0,         0,         0,         G, 0, 0],
            [0,         0,         0,         0, G, 0],
            [0,         0,         0,         0, 0, G]
        ])
        
        return C
    
    @cached_property
    def compliance_matrix(self) -> np.ndarray:
        """Returns 3D compliance matrix S (6×6).
        
        Returns:
            Compliance matrix S (6×6)
        """
        E_inv = 1 / self.E
        G_inv = 1 / self.G
        
        S = np.array([
            [E_inv,      -self.nu*E_inv, -self.nu*E_inv, 0,     0,     0],
            [-self.nu*E_inv, E_inv,       -self.nu*E_inv, 0,     0,     0],
            [-self.nu*E_inv, -self.nu*E_inv, E_inv,       0,     0,     0],
            [0,          0,               0,              G_inv, 0,     0],
            [0,          0,               0,              0,     G_inv, 0],
            [0,          0,               0,              0,     0,     G_inv]
        ])
        
        return S
    
    def stiffness_matrix_2d(self, state: str = "plane_stress") -> np.ndarray:
        """Returns 2D stiffness matrix (3×3) for plane stress or plane strain.
        
        Args:
            state: Either 'plane_stress' or 'plane_strain'
        
        Returns:
            Reduced stiffness matrix Q (3×3)
        """
        if state == "plane_stress":
            factor = self.E / (1 - self.nu**2)
            Q = factor * np.array([
                [1,       self.nu, 0],
                [self.nu, 1,       0],
                [0,       0,       (1 - self.nu) / 2]
            ])
        elif state == "plane_strain":
            factor = self.E / ((1 + self.nu) * (1 - 2*self.nu))
            Q = factor * np.array([
                [1 - self.nu, self.nu,     0],
                [self.nu,     1 - self.nu, 0],
                [0,           0,           (1 - 2*self.nu) / 2]
            ])
        else:
            raise ValueError(f"Invalid state: {state}. Use 'plane_stress' or 'plane_strain'.")
        
        return Q
    
    def compliance_matrix_2d(self, state: str = "plane_stress") -> np.ndarray:
        """Returns 2D compliance matrix (3×3) for plane stress or plane strain.
        
        Args:
            state: Either 'plane_stress' or 'plane_strain'
        
        Returns:
            Reduced compliance matrix S (3×3)
        """
        E_inv = 1 / self.E
        G_inv = 1 / self.G
        
        if state == "plane_stress":
            S = np.array([
                [E_inv,          -self.nu*E_inv, 0],
                [-self.nu*E_inv, E_inv,          0],
                [0,              0,              G_inv]
            ])
        elif state == "plane_strain":
            factor = (1 + self.nu) / self.E
            S = factor * np.array([
                [1 - self.nu,  -self.nu,      0],
                [-self.nu,     1 - self.nu,   0],
                [0,            0,             2]
            ])
        else:
            raise ValueError(f"Invalid state: {state}. Use 'plane_stress' or 'plane_strain'.")
        
        return S
    
    def __eq__(self, other) -> bool:
        """Check equality based on material properties."""
        if not isinstance(other, Isotropic):
            return False
        return np.allclose([self.E, self.nu], [other.E, other.nu])
    
    def __repr__(self) -> str:
        """String representation of the material."""
        return (
            f"Isotropic(name='{self.name}', "
            f"E={self.E:.2e}, nu={self.nu:.3f}, G={self.G:.2e})"
        )


class TransverselyIsotropic(Material):
    """Transversely isotropic material (special case of orthotropic).
    
    Material with isotropic behavior in one plane (2-3 plane) and 
    different properties in the perpendicular direction (1-axis).
    Common in unidirectional fiber composites.
    
    Attributes:
        E1: Longitudinal Young's modulus (Pa)
        E2: Transverse Young's modulus (= E3) (Pa)
        nu12: Major Poisson's ratio (= nu13)
        nu23: Transverse Poisson's ratio
        G12: Longitudinal shear modulus (= G13) (Pa)
        G23: Transverse shear modulus (Pa) - computed if not provided
    """
    
    def __init__(
        self,
        E1: float,
        E2: float,
        nu12: float,
        nu23: float,
        G12: float,
        G23: Optional[float] = None,
        name: Optional[str] = None,
        density: Optional[float] = None,
    ):
        """Initialize transversely isotropic material.
        
        Args:
            E1: Longitudinal Young's modulus (Pa)
            E2: Transverse Young's modulus (Pa)
            nu12: Longitudinal Poisson's ratio
            nu23: Transverse Poisson's ratio
            G12: Longitudinal shear modulus (Pa)
            G23: Transverse shear modulus (Pa). If None, computed as E2/(2(1+nu23))
            name: Material name
            density: Material density (kg/m³)
        """
        super().__init__(name=name, density=density)
        
        self.E1 = E1
        self.E2 = E2
        self.nu12 = nu12
        self.nu23 = nu23
        self.G12 = G12
        self._G23 = G23
    
    @property
    def E3(self) -> float:
        """Through-thickness Young's modulus (= E2)."""
        return self.E2
    
    @property
    def nu13(self) -> float:
        """Poisson's ratio in 1-3 plane (= nu12)."""
        return self.nu12
    
    @property
    def G13(self) -> float:
        """Shear modulus in 1-3 plane (= G12)."""
        return self.G12
    
    @property
    def G23(self) -> float:
        """Transverse shear modulus (computed if not provided)."""
        if self._G23 is not None:
            return self._G23
        return self.E2 / (2 * (1 + self.nu23))
    
    @property
    def nu21(self) -> float:
        """Minor Poisson's ratio."""
        return self.nu12 * self.E2 / self.E1
    
    @property
    def nu31(self) -> float:
        """Minor Poisson's ratio (= nu21)."""
        return self.nu21
    
    @property
    def nu32(self) -> float:
        """Minor Poisson's ratio (= nu23)."""
        return self.nu23
    
    @cached_property
    def compliance_matrix(self) -> np.ndarray:
        """Returns compliance matrix S (6×6).
        
        Returns:
            Compliance matrix S (6×6)
        """
        S = np.zeros((6, 6))
        
        # Normal strains
        S[0, 0] = 1 / self.E1
        S[0, 1] = -self.nu12 / self.E1
        S[0, 2] = -self.nu12 / self.E1
        
        S[1, 0] = -self.nu21 / self.E2
        S[1, 1] = 1 / self.E2
        S[1, 2] = -self.nu23 / self.E2
        
        S[2, 0] = -self.nu21 / self.E2
        S[2, 1] = -self.nu23 / self.E2
        S[2, 2] = 1 / self.E2
        
        # Shear strains
        S[3, 3] = 1 / self.G23
        S[4, 4] = 1 / self.G12
        S[5, 5] = 1 / self.G12
        
        return S
    
    @cached_property
    def stiffness_matrix(self) -> np.ndarray:
        """Returns stiffness matrix C = S⁻¹ (6×6).
        
        Returns:
            Stiffness matrix C (6×6)
        """
        return np.linalg.inv(self.compliance_matrix)
    
    def __eq__(self, other) -> bool:
        """Check equality based on material properties."""
        if not isinstance(other, TransverselyIsotropic):
            return False
        return np.allclose(
            [self.E1, self.E2, self.nu12, self.nu23, self.G12, self.G23],
            [other.E1, other.E2, other.nu12, other.nu23, other.G12, other.G23]
        )
    
    def __repr__(self) -> str:
        """String representation of the material."""
        return (
            f"TransverselyIsotropic(name='{self.name}', "
            f"E1={self.E1:.2e}, E2={self.E2:.2e}, "
            f"nu12={self.nu12:.3f}, G12={self.G12:.2e})"
        )
