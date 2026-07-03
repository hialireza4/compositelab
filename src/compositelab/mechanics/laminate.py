"""
Classical Laminate Theory (CLT) implementation.

This module provides the Laminate class for analyzing composite laminates
using Classical Laminate Theory.
"""

import numpy as np
from typing import Optional, Tuple, List
from compositelab.layers.layup import Layup


class Laminate:
    """
    Classical Laminate Theory (CLT) analyzer.
    
    This class computes the ABD stiffness matrices and analyzes the mechanical
    response of composite laminates under in-plane loads (N) and moments (M).
    
    Parameters
    ----------
    layup : Layup
        The layup defining the geometry and stacking sequence.
        
    Attributes
    ----------
    layup : Layup
        Reference to the layup object.
    A : ndarray (3, 3)
        Extensional stiffness matrix [N/m].
    B : ndarray (3, 3)
        Coupling stiffness matrix [N].
    D : ndarray (3, 3)
        Bending stiffness matrix [N·m].
    ABD : ndarray (6, 6)
        Combined laminate stiffness matrix.
    abd : ndarray (6, 6)
        Laminate compliance matrix (inverse of ABD).
        
    Notes
    -----
    - All thicknesses must be in meters (SI).
    - Forces N = [Nx, Ny, Nxy] in [N/m].
    - Moments M = [Mx, My, Mxy] in [N].
    - Z-coordinates follow laminate reference (typically mid-plane at z=0).
    """
    
    def __init__(self, layup: Layup):
        """
        Initialize the laminate analyzer.
        
        Parameters
        ----------
        layup : Layup
            Layup object containing layers and geometry.
        """
        self.layup = layup
        
        # Cache for ABD matrices
        self._A = None
        self._B = None
        self._D = None
        self._ABD = None
        self._abd = None
        
    def _compute_ABD(self):
        """
        Compute A, B, D matrices using Classical Laminate Theory.
        
        The matrices are computed by integrating the transformed reduced
        stiffness matrices through the laminate thickness:
        
        A_ij = Σ (Q̄_ij)_k (z_k - z_{k-1})
        B_ij = (1/2) Σ (Q̄_ij)_k (z_k² - z_{k-1}²)
        D_ij = (1/3) Σ (Q̄_ij)_k (z_k³ - z_{k-1}³)
        
        Notes
        -----
        Results are cached. Subsequent calls return immediately.
        """
        if self._A is not None:
            return  # Already computed
            
        n_layers = len(self.layup.layers)
        z_coords = self.z_coords 
        
        # Initialize matrices
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))
        
        # Integrate through thickness
        for i in range(n_layers):
            layer = self.layup.layers[i]
            z_bottom = z_coords[i]
            z_top = z_coords[i + 1]
            
            # Get transformed reduced stiffness matrix Q̄
            # Note: Layer.Q_bar uses material.get_rotated_stiffness_matrix(angle)
            Q_bar = layer.Q_bar
            
            # CLT integration
            h = z_top - z_bottom
            z_mid = (z_top + z_bottom) / 2
            
            A += Q_bar * h
            B += Q_bar * h * z_mid
            D += Q_bar * (h**3 / 12 + h * z_mid**2)
        
        #Check B matrix for symetric condition
        B[np.abs(B) < 1e-10 * max(np.abs(layer.Q_bar).max() for layer in self.layup.layers) * (z_coords[-1] - z_coords[0])**2] = 0.0
        # Cache results
        self._A = A
        self._B = B
        self._D = D
        
        # Build combined 6×6 matrix
        self._ABD = np.block([
            [A, B],
            [B, D]
        ])
        
        # Compute compliance (inverse)
        try:
            self._abd = np.linalg.inv(self._ABD)
        except np.linalg.LinAlgError:
            raise ValueError(
                "ABD matrix is singular. Check layer definitions and thicknesses."
            )
    
    @property
    def z_coords(self) -> np.ndarray:
        """
        Through-thickness z-coordinates of layer boundaries.
        
        Returns
        -------
        ndarray
            Array of z-coordinates [m], length = n_layers + 1.
        """
        return self.layup.z_coords
    
    @property
    def A(self) -> np.ndarray:
        """
        Extensional stiffness matrix [N/m].
        
        Returns
        -------
        ndarray (3, 3)
            A matrix relating in-plane forces to mid-plane strains.
        """
        self._compute_ABD()
        return self._A
    
    @property
    def B(self) -> np.ndarray:
        """
        Coupling stiffness matrix [N].
        
        Returns
        -------
        ndarray (3, 3)
            B matrix relating forces to curvatures and moments to strains.
            Zero for symmetric laminates.
        """
        self._compute_ABD()
        return self._B
    
    @property
    def D(self) -> np.ndarray:
        """
        Bending stiffness matrix [N·m].
        
        Returns
        -------
        ndarray (3, 3)
            D matrix relating moments to curvatures.
        """
        self._compute_ABD()
        return self._D
    
    @property
    def ABD(self) -> np.ndarray:
        """
        Combined laminate stiffness matrix.
        
        Returns
        -------
        ndarray (6, 6)
            Block matrix [[A, B], [B, D]].
        """
        self._compute_ABD()
        return self._ABD
    
    @property
    def abd(self) -> np.ndarray:
        """
        Laminate compliance matrix (inverse of ABD).
        
        Returns
        -------
        ndarray (6, 6)
            Compliance matrix relating loads to mid-plane strains/curvatures.
        """
        self._compute_ABD()
        return self._abd
    
    def get_midplane_strains(
        self, 
        N: np.ndarray, 
        M: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate mid-plane strains and curvatures from applied loads.
        
        Solves the laminate constitutive equation:
        [N]   [A  B] [ε⁰]
        [M] = [B  D] [κ ]
        
        Parameters
        ----------
        N : array_like (3,)
            In-plane force resultants [Nx, Ny, Nxy] in [N/m].
        M : array_like (3,)
            Moment resultants [Mx, My, Mxy] in [N].
            
        Returns
        -------
        eps0 : ndarray (3,)
            Mid-plane strains [εx⁰, εy⁰, γxy⁰] (dimensionless).
        kappa : ndarray (3,)
            Curvatures [κx, κy, κxy] in [1/m].
            
        Examples
        --------
        >>> N = np.array([1000, 0, 0])  # Nx = 1000 N/m
        >>> M = np.array([0, 0, 0])      # No moments
        >>> eps0, kappa = laminate.get_midplane_strains(N, M)
        """
        N = np.asarray(N, dtype=float).reshape(3, 1)
        M = np.asarray(M, dtype=float).reshape(3, 1)
        
        # Build load vector
        loads = np.vstack([N, M])
        
        # Solve [ε⁰, κ]ᵀ = abd · [N, M]ᵀ
        result = self.abd @ loads
        
        eps0 = result[:3].flatten()
        kappa = result[3:].flatten()
        
        return eps0, kappa
    
    def get_strains_at_z(
        self, 
        z: float, 
        eps0: np.ndarray, 
        kappa: np.ndarray
    ) -> np.ndarray:
        """
        Calculate strains at a given z-coordinate through the laminate thickness.
        
        Uses the kinematic assumption of CLT:
        ε(z) = ε⁰ + z·κ
        
        Parameters
        ----------
        z : float
            Through-thickness coordinate [m].
        eps0 : array_like (3,)
            Mid-plane strains [εx⁰, εy⁰, γxy⁰].
        kappa : array_like (3,)
            Curvatures [κx, κy, κxy] in [1/m].
            
        Returns
        -------
        ndarray (3,)
            Strain vector [εx, εy, γxy] at z.
        """
        eps0 = np.asarray(eps0)
        kappa = np.asarray(kappa)
        return eps0 + z * kappa
    
    def get_layer_strains(
        self,
        layer_index: int,
        eps0: np.ndarray = None,
        kappa: np.ndarray = None,
        surface: str = 'all'
    ) -> Tuple[np.ndarray, ...]:
        """
        Calculate strains at layer surfaces.
        
        Parameters
        ----------
        layer_index : int
            Layer index (0-based).
        eps0 : array_like (3,), optional
            Mid-plane strains. If None, must be provided with loads.
        kappa : array_like (3,), optional
            Curvatures. If None, must be provided with loads.
        surface : {'all', 'bottom', 'mid', 'top'}, default 'all'
            Which surface(s) to return.
            
        Returns
        -------
        tuple of ndarray
            If surface='all': (eps_bottom, eps_mid, eps_top)
            Otherwise: single ndarray for requested surface.
            
        Raises
        ------
        ValueError
            If layer_index is out of range or eps0/kappa not provided.
        """
        if eps0 is None or kappa is None:
            raise ValueError(
                "Must provide eps0 and kappa. "
                "Use get_midplane_strains() first if applying loads."
            )
        
        n_layers = len(self.layup.layers)
        if not (0 <= layer_index < n_layers):
            raise ValueError(
                f"layer_index {layer_index} out of range [0, {n_layers-1}]"
            )
        
        z_bottom = self.z_coords[layer_index]
        z_top = self.z_coords[layer_index + 1]
        z_mid = (z_bottom + z_top) / 2
        
        eps_bottom = self.get_strains_at_z(z_bottom, eps0, kappa)
        eps_mid = self.get_strains_at_z(z_mid, eps0, kappa)
        eps_top = self.get_strains_at_z(z_top, eps0, kappa)
        
        if surface == 'all':
            return eps_bottom, eps_mid, eps_top
        elif surface == 'bottom':
            return eps_bottom
        elif surface == 'mid':
            return eps_mid
        elif surface == 'top':
            return eps_top
        else:
            raise ValueError(
                f"Invalid surface '{surface}'. "
                "Must be 'all', 'bottom', 'mid', or 'top'."
            )
    
    def get_layer_stresses(
        self,
        layer_index: int,
        eps0: np.ndarray = None,
        kappa: np.ndarray = None,
        surface: str = 'all'
    ) -> Tuple[np.ndarray, ...]:
        """
        Calculate stresses at layer surfaces.
        
        Stresses are computed using the layer's transformed stiffness:
        σ(z) = Q̄ · ε(z)
        
        Parameters
        ----------
        layer_index : int
            Layer index (0-based).
        eps0 : array_like (3,), optional
            Mid-plane strains.
        kappa : array_like (3,), optional
            Curvatures.
        surface : {'all', 'bottom', 'mid', 'top'}, default 'all'
            Which surface(s) to return.
            
        Returns
        -------
        tuple of ndarray
            If surface='all': (sigma_bottom, sigma_mid, sigma_top)
            Otherwise: single ndarray for requested surface.
            Each stress vector is [σx, σy, τxy] in [Pa].
            
        Notes
        -----
        Stresses are discontinuous at layer interfaces due to different
        material properties and fiber orientations.
        """
        # Get strains at requested surfaces
        strains = self.get_layer_strains(layer_index, eps0, kappa, surface)
        
        # Get layer's transformed stiffness
        layer = self.layup.layers[layer_index]
        Q_bar = layer.Q_bar
        
        # Compute stresses
        if surface == 'all':
            eps_bottom, eps_mid, eps_top = strains
            sigma_bottom = Q_bar @ eps_bottom
            sigma_mid = Q_bar @ eps_mid
            sigma_top = Q_bar @ eps_top
            return sigma_bottom, sigma_mid, sigma_top
        else:
            # Single surface
            return Q_bar @ strains
    @property
    def z_coords(self) -> np.ndarray:
        """
        Through-thickness z-coordinates of layer boundaries.
        
        Returns
        -------
        ndarray
            Array of z-coordinates [m], length = n_layers + 1.
            First element is bottom surface, last is top surface.
        """
        positions = self.layup.get_layer_positions()
        
        # Extract unique boundaries: [z0_bottom, z1_top, z2_top, ..., zn_top]
        z = [positions[0][0]]  # First layer bottom
        for z_bottom, z_top in positions:
            z.append(z_top)
        
        return np.array(z)

    def is_symmetric(self, tol: float = 1e-6) -> bool:
        """
        Check if the laminate is geometrically symmetric.
        
        A symmetric laminate has B ≈ 0, meaning no coupling between
        extension and bending.
        
        Parameters
        ----------
        tol : float, default 1e-6
            Tolerance for near-zero elements in B matrix.
            
        Returns
        -------
        bool
            True if max|B_ij| < tol.
        """
        return np.allclose(self.B, 0, atol=tol)
    
    def summary(self):
        """
        Print a detailed summary of the laminate properties.
        
        Displays:
        - Total thickness
        - Number of layers
        - Symmetry status
        - A, B, D matrices
        - Layer-by-layer details
        """
        print("=" * 60)
        print("LAMINATE SUMMARY")
        print("=" * 60)
        print(f"Total thickness: {self.layup.total_thickness*1e3:.4f} mm")
        print(f"Number of layers: {len(self.layup.layers)}")
        print(f"Symmetric: {'Yes' if self.is_symmetric() else 'No'}")
        print()
        
        print("A matrix [N/m]:")
        print(self.A)
        print()
        
        print("B matrix [N]:")
        print(self.B)
        print()
        
        print("D matrix [N·m]:")
        print(self.D)
        print()
        
        print("-" * 60)
        print("Layer details:")
        print("-" * 60)
        for i, layer in enumerate(self.layup.layers):
            z_bot = self.z_coords[i] * 1e3
            z_top = self.z_coords[i + 1] * 1e3
            print(f"Layer {i+1}: θ={layer.angle:>6.1f}°  "
                  f"t={layer.thickness*1e3:>6.4f} mm  "
                  f"z=[{z_bot:>7.4f}, {z_top:>7.4f}] mm")
        print("=" * 60)
