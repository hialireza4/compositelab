"""Orthotropic material models for compositelab package."""

from typing import Optional, Dict
import numpy as np
from functools import cached_property

from .base import Material


class Orthotropic2D(Material):
    """2D orthotropic material under plane stress assumption.
    
    This class represents a material with different properties in two
    perpendicular directions (fiber and transverse). Suitable for
    unidirectional composite laminae in CLT analysis.
    
    Attributes:
        E1: Longitudinal Young's modulus (Pa)
        E2: Transverse Young's modulus (Pa)
        nu12: Major Poisson's ratio
        G12: In-plane shear modulus (Pa)
        thickness: Ply thickness in meters (optional)
        strengths: Material strength properties (optional)
    """
    
    def __init__(
        self,
        E1: float,
        E2: float,
        nu12: float,
        G12: float,
        thickness: Optional[float] = None,
        name: Optional[str] = None,
        density: Optional[float] = None,
        Xt: Optional[float] = None,
        Xc: Optional[float] = None,
        Yt: Optional[float] = None,
        Yc: Optional[float] = None,
        S: Optional[float] = None,
    ):
        """Initialize 2D orthotropic material.
        
        Args:
            E1: Longitudinal Young's modulus (Pa)
            E2: Transverse Young's modulus (Pa)
            nu12: Major Poisson's ratio
            G12: In-plane shear modulus (Pa)
            thickness: Ply thickness (m)
            name: Material name
            density: Material density (kg/m³)
            Xt: Longitudinal tensile strength (Pa)
            Xc: Longitudinal compressive strength (Pa)
            Yt: Transverse tensile strength (Pa)
            Yc: Transverse compressive strength (Pa)
            S: In-plane shear strength (Pa)
        """
        super().__init__(name=name, density=density)
        
        self.E1 = E1
        self.E2 = E2
        self.nu12 = nu12
        self.G12 = G12
        self.thickness = thickness
        
        # Strength properties (optional)
        self.Xt = Xt
        self.Xc = Xc
        self.Yt = Yt
        self.Yc = Yc
        self.S = S
    
    @property
    def nu21(self) -> float:
        """Minor Poisson's ratio (computed from reciprocal relation)."""
        return self.nu12 * self.E2 / self.E1
    
    @cached_property
    def stiffness_matrix(self) -> np.ndarray:
        """Returns reduced stiffness matrix Q (3×3) under plane stress.
        
        Returns:
            Q matrix with components:
                [[Q11, Q12,   0],
                 [Q12, Q22,   0],
                 [  0,   0, Q66]]
        """
        denom = 1.0 - self.nu12 * self.nu21
        
        Q11 = self.E1 / denom
        Q22 = self.E2 / denom
        Q12 = self.nu12 * self.E2 / denom
        Q66 = self.G12
        
        Q = np.array([
            [Q11, Q12, 0.0],
            [Q12, Q22, 0.0],
            [0.0, 0.0, Q66]
        ])
        
        return Q
    
    @cached_property
    def compliance_matrix(self) -> np.ndarray:
        """Returns compliance matrix S = Q⁻¹ (3×3).
        
        Returns:
            S matrix with components:
                [[1/E1,  -nu12/E1,      0],
                 [-nu21/E2,  1/E2,      0],
                 [     0,       0,  1/G12]]
        """
        S = np.array([
            [1/self.E1, -self.nu12/self.E1, 0.0],
            [-self.nu21/self.E2, 1/self.E2, 0.0],
            [0.0, 0.0, 1/self.G12]
        ])
        
        return S
    
    def get_rotated_stiffness_matrix(self, theta: float) -> np.ndarray:
        """Returns transformed stiffness matrix Q̄(θ) for rotated lamina.
        
        Args:
            theta: Rotation angle in degrees (positive counterclockwise)
        
        Returns:
            Transformed stiffness matrix Q̄ (3×3)
        """
        theta_rad = np.radians(theta)
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        
        c2 = c**2
        s2 = s**2
        c3 = c**3
        s3 = s**3
        c4 = c**4
        s4 = s**4
        
        Q = self.stiffness_matrix
        Q11, Q22, Q12, Q66 = Q[0,0], Q[1,1], Q[0,1], Q[2,2]
        
        # Transformed stiffness components
        Q11_bar = Q11*c4 + 2*(Q12 + 2*Q66)*s2*c2 + Q22*s4
        Q22_bar = Q11*s4 + 2*(Q12 + 2*Q66)*s2*c2 + Q22*c4
        Q12_bar = (Q11 + Q22 - 4*Q66)*s2*c2 + Q12*(s4 + c4)
        Q66_bar = (Q11 + Q22 - 2*Q12 - 2*Q66)*s2*c2 + Q66*(s4 + c4)
        Q16_bar = (Q11 - Q12 - 2*Q66)*s*c3 + (Q12 - Q22 + 2*Q66)*s3*c
        Q26_bar = (Q11 - Q12 - 2*Q66)*s3*c + (Q12 - Q22 + 2*Q66)*s*c3
        
        Q_bar = np.array([
            [Q11_bar, Q12_bar, Q16_bar],
            [Q12_bar, Q22_bar, Q26_bar],
            [Q16_bar, Q26_bar, Q66_bar]
        ])
        
        return Q_bar
    
    @cached_property
    def invariants(self) -> np.ndarray:
        """Returns Tsai-Pagano invariants U (5,).
        
        These invariants are useful for laminate analysis as they
        separate material properties from orientation effects.
        
        Returns:
            Array [U1, U2, U3, U4, U5]
        """
        Q = self.stiffness_matrix
        Q11, Q22, Q12, Q66 = Q[0,0], Q[1,1], Q[0,1], Q[2,2]
        
        U1 = (3*Q11 + 3*Q22 + 2*Q12 + 4*Q66) / 8
        U2 = (Q11 - Q22) / 2
        U3 = (Q11 + Q22 - 2*Q12 - 4*Q66) / 8
        U4 = (Q11 + Q22 + 6*Q12 - 4*Q66) / 8
        U5 = (Q11 + Q22 - 2*Q12 + 4*Q66) / 8
        
        return np.array([U1, U2, U3, U4, U5])
    
    def __eq__(self, other) -> bool:
        """Check equality based on material properties."""
        if not isinstance(other, Orthotropic2D):
            return False
        return np.allclose(
            [self.E1, self.E2, self.nu12, self.G12],
            [other.E1, other.E2, other.nu12, other.G12]
        )
    
    def __repr__(self) -> str:
        """String representation of the material."""
        return (
            f"Orthotropic2D(name='{self.name}', "
            f"E1={self.E1:.2e}, E2={self.E2:.2e}, "
            f"nu12={self.nu12:.3f}, G12={self.G12:.2e})"
        )


class Orthotropic3D(Material):
    """3D orthotropic material with full anisotropic stiffness tensor.
    
    This class represents a material with different properties along
    three perpendicular axes. Suitable for 3D FEM analysis or thick
    composite structures.
    
    Attributes:
        E1, E2, E3: Young's moduli in three principal directions (Pa)
        nu12, nu13, nu23: Poisson's ratios
        G12, G13, G23: Shear moduli (Pa)
        strengths: Material strength properties (optional)
    """
    
    def __init__(
        self,
        E1: float,
        E2: float,
        E3: float,
        nu12: float,
        nu13: float,
        nu23: float,
        G12: float,
        G13: float,
        G23: float,
        name: Optional[str] = None,
        density: Optional[float] = None,
        Xt: Optional[float] = None,
        Xc: Optional[float] = None,
        Yt: Optional[float] = None,
        Yc: Optional[float] = None,
        Zt: Optional[float] = None,
        Zc: Optional[float] = None,
        S12: Optional[float] = None,
        S13: Optional[float] = None,
        S23: Optional[float] = None,
    ):
        """Initialize 3D orthotropic material.
        
        Args:
            E1, E2, E3: Young's moduli (Pa)
            nu12, nu13, nu23: Major Poisson's ratios
            G12, G13, G23: Shear moduli (Pa)
            name: Material name
            density: Material density (kg/m³)
            Xt, Xc: Longitudinal tensile/compressive strength (Pa)
            Yt, Yc: Transverse tensile/compressive strength (Pa)
            Zt, Zc: Through-thickness tensile/compressive strength (Pa)
            S12, S13, S23: Shear strengths (Pa)
        """
        super().__init__(name=name, density=density)
        
        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        self.nu12 = nu12
        self.nu13 = nu13
        self.nu23 = nu23
        self.G12 = G12
        self.G13 = G13
        self.G23 = G23
        
        # Strength properties (optional)
        self.Xt = Xt
        self.Xc = Xc
        self.Yt = Yt
        self.Yc = Yc
        self.Zt = Zt
        self.Zc = Zc
        self.S12 = S12
        self.S13 = S13
        self.S23 = S23
    
    @property
    def nu21(self) -> float:
        """Minor Poisson's ratio ν21."""
        return self.nu12 * self.E2 / self.E1
    
    @property
    def nu31(self) -> float:
        """Minor Poisson's ratio ν31."""
        return self.nu13 * self.E3 / self.E1
    
    @property
    def nu32(self) -> float:
        """Minor Poisson's ratio ν32."""
        return self.nu23 * self.E3 / self.E2
    
    @cached_property
    def compliance_matrix(self) -> np.ndarray:
        """Returns compliance matrix S (6×6).
        
        Voigt notation: [ε11, ε22, ε33, γ23, γ13, γ12]ᵀ = S [σ11, σ22, σ33, σ23, σ13, σ12]ᵀ
        
        Returns:
            Compliance matrix S (6×6)
        """
        S = np.zeros((6, 6))
        
        # Normal strains
        S[0, 0] = 1 / self.E1
        S[0, 1] = -self.nu12 / self.E1
        S[0, 2] = -self.nu13 / self.E1
        
        S[1, 0] = -self.nu21 / self.E2
        S[1, 1] = 1 / self.E2
        S[1, 2] = -self.nu23 / self.E2
        
        S[2, 0] = -self.nu31 / self.E3
        S[2, 1] = -self.nu32 / self.E3
        S[2, 2] = 1 / self.E3
        
        # Shear strains
        S[3, 3] = 1 / self.G23
        S[4, 4] = 1 / self.G13
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
        if not isinstance(other, Orthotropic3D):
            return False
        return np.allclose(
            [self.E1, self.E2, self.E3, self.nu12, self.nu13, self.nu23,
             self.G12, self.G13, self.G23],
            [other.E1, other.E2, other.E3, other.nu12, other.nu13, other.nu23,
             other.G12, other.G13, other.G23]
        )
    
    def __repr__(self) -> str:
        """String representation of the material."""
        return (
            f"Orthotropic3D(name='{self.name}', "
            f"E1={self.E1:.2e}, E2={self.E2:.2e}, E3={self.E3:.2e})"
        )
