"""
Hashin Failure Criterion for Composite Materials

Implements the Hashin failure criterion with four failure modes:
- Fiber Tension (FT)
- Fiber Compression (FC)
- Matrix Tension (MT)
- Matrix Compression (MC)

Also includes Progressive Failure Analysis (PFA) capability.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import copy


# Default degradation factors for failed material properties
DEFAULT_DEGRADATION_FACTORS = {
    'FT': {'E1': 0.01, 'E2': 1.0, 'G12': 0.01, 'nu12': 0.01},
    'FC': {'E1': 0.01, 'E2': 1.0, 'G12': 0.01, 'nu12': 0.01},
    'MT': {'E1': 1.0, 'E2': 0.01, 'G12': 0.01, 'nu12': 0.01},
    'MC': {'E1': 1.0, 'E2': 0.01, 'G12': 0.01, 'nu12': 0.01}
}


class HashinFailure:
    """
    Hashin failure criterion implementation for composite laminates.
    
    This class provides methods to evaluate failure indices for individual layers
    and perform progressive failure analysis on the entire laminate.
    
    Parameters
    ----------
    laminate : Laminate
        The composite laminate object containing layup and material properties
    
    Attributes
    ----------
    laminate : Laminate
        Reference to the laminate being analyzed
    """
    
    def __init__(self, laminate):
        """
        Initialize the Hashin failure criterion.
        
        Parameters
        ----------
        laminate : Laminate
            Composite laminate object to analyze
        """
        self.laminate = laminate
        self._validate_materials()
    
    def _validate_materials(self) -> None:
        """
        Validate that all materials have required strength properties.
        
        Raises
        ------
        ValueError
            If any material lacks required strength properties (Xt, Xc, Yt, Yc, S)
        """
        required_props = ['Xt', 'Xc', 'Yt', 'Yc', 'S']
        
        for i, layer in enumerate(self.laminate.layup.layers):
            material = layer.material
            missing = [prop for prop in required_props if not hasattr(material, prop)]
            
            if missing:
                raise ValueError(
                    f"Layer {i} material '{material.name}' is missing required "
                    f"strength properties: {missing}. Required: {required_props}"
                )
    
    @staticmethod
    def _get_strain_transformation_matrix(theta: float) -> np.ndarray:
        """
        Compute the strain transformation matrix from global to local coordinates.
        
        Parameters
        ----------
        theta : float
            Layer angle in degrees
        
        Returns
        -------
        np.ndarray
            3x3 strain transformation matrix
        """
        theta_rad = np.radians(theta)
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        
        T = np.array([
            [c**2, s**2, 2*c*s],
            [s**2, c**2, -2*c*s],
            [-c*s, c*s, c**2 - s**2]
        ])
        
        return T
    
    def evaluate_fiber_tension(
        self, 
        sigma_local: np.ndarray, 
        material
    ) -> float:
        """
        Evaluate fiber tension failure index (σ₁ ≥ 0).
        
        Parameters
        ----------
        sigma_local : np.ndarray
            Local stress vector [σ₁, σ₂, τ₁₂]
        material : Material
            Material object with strength properties
        
        Returns
        -------
        float
            Failure index (≥1 indicates failure)
        """
        sigma_1 = sigma_local[0]
        tau_12 = sigma_local[2]
        
        if sigma_1 >= 0:
            FT = (sigma_1 / material.Xt)**2 + (tau_12 / material.S)**2
            return FT
        else:
            return 0.0
    
    def evaluate_fiber_compression(
        self, 
        sigma_local: np.ndarray, 
        material
    ) -> float:
        """
        Evaluate fiber compression failure index (σ₁ < 0).
        
        Parameters
        ----------
        sigma_local : np.ndarray
            Local stress vector [σ₁, σ₂, τ₁₂]
        material : Material
            Material object with strength properties
        
        Returns
        -------
        float
            Failure index (≥1 indicates failure)
        """
        sigma_1 = sigma_local[0]
        
        if sigma_1 < 0:
            FC = (sigma_1 / material.Xc)**2
            return FC
        else:
            return 0.0
    
    def evaluate_matrix_tension(
        self, 
        sigma_local: np.ndarray, 
        material
    ) -> float:
        """
        Evaluate matrix tension failure index (σ₂ ≥ 0).
        
        Parameters
        ----------
        sigma_local : np.ndarray
            Local stress vector [σ₁, σ₂, τ₁₂]
        material : Material
            Material object with strength properties
        
        Returns
        -------
        float
            Failure index (≥1 indicates failure)
        """
        sigma_2 = sigma_local[1]
        tau_12 = sigma_local[2]
        
        if sigma_2 >= 0:
            MT = (sigma_2 / material.Yt)**2 + (tau_12 / material.S)**2
            return MT
        else:
            return 0.0
    
    def evaluate_matrix_compression(
        self, 
        sigma_local: np.ndarray, 
        material
    ) -> float:
        """
        Evaluate matrix compression failure index (σ₂ < 0).
        
        Parameters
        ----------
        sigma_local : np.ndarray
            Local stress vector [σ₁, σ₂, τ₁₂]
        material : Material
            Material object with strength properties
        
        Returns
        -------
        float
            Failure index (≥1 indicates failure)
        """
        sigma_2 = sigma_local[1]
        tau_12 = sigma_local[2]
        
        if sigma_2 < 0:
            MC = (sigma_2 / (2 * material.S))**2 + \
                 ((material.Yc / (2 * material.S))**2 - 1) * (sigma_2 / material.Yc) + \
                 (tau_12 / material.S)**2
            return MC
        else:
            return 0.0
    
    def calculate_failure_indices(
        self, 
        eps0: np.ndarray, 
        kappa: np.ndarray
    ) -> List[Dict]:
        """
        Calculate Hashin failure indices for all layers.
        
        Parameters
        ----------
        eps0 : np.ndarray
            Midplane strain vector [ε₀ₓ, ε₀ᵧ, γ₀ₓᵧ]
        kappa : np.ndarray
            Curvature vector [κₓ, κᵧ, κₓᵧ]
        
        Returns
        -------
        List[Dict]
            List of dictionaries containing failure indices for each layer:
            - 'layer': layer index
            - 'z': z-coordinate at layer mid-plane
            - 'FT': fiber tension index
            - 'FC': fiber compression index
            - 'MT': matrix tension index
            - 'MC': matrix compression index
            - 'max_index': maximum failure index
            - 'critical_mode': critical failure mode
        """
        results = []
        layer_positions = self.laminate.layup.get_layer_positions()
        
        for i, layer in enumerate(self.laminate.layup.layers):
            # Z-coordinate at layer mid-plane
            z_bottom, z_top = layer_positions[i]
            z_mid = (z_bottom + z_top) / 2
            
            # Strain at layer mid-plane (global coordinates)
            eps_global = eps0 + z_mid * kappa
            
            # Transform to local coordinates
            theta = layer.angle
            T_eps = self._get_strain_transformation_matrix(theta)
            eps_local = T_eps @ eps_global
            
            # Calculate stress in local coordinates
            Q = layer.material.stiffness_matrix
            sigma_local = Q @ eps_local
            
            # Calculate failure indices
            material = layer.material
            FT = self.evaluate_fiber_tension(sigma_local, material)
            FC = self.evaluate_fiber_compression(sigma_local, material)
            MT = self.evaluate_matrix_tension(sigma_local, material)
            MC = self.evaluate_matrix_compression(sigma_local, material)
            
            # Find maximum index and critical mode
            indices = {'FT': FT, 'FC': FC, 'MT': MT, 'MC': MC}
            max_mode = max(indices, key=indices.get)
            max_index = indices[max_mode]
            
            results.append({
                'layer': i,
                'z': z_mid,
                'FT': FT,
                'FC': FC,
                'MT': MT,
                'MC': MC,
                'max_index': max_index,
                'critical_mode': max_mode
            })
        
        return results
    
    def get_first_ply_failure_load(
        self,
        N: np.ndarray,
        M: np.ndarray,
        load_factor_range: Tuple[float, float] = (0.0, 10.0),
        num_steps: int = 100
    ) -> Tuple[float, int, str]:
        """
        Find the load factor at which first ply failure (FPF) occurs.
        
        Parameters
        ----------
        N : np.ndarray
            Reference force resultant vector [Nₓ, Nᵧ, Nₓᵧ]
        M : np.ndarray
            Reference moment resultant vector [Mₓ, Mᵧ, Mₓᵧ]
        load_factor_range : Tuple[float, float], optional
            Range of load factors to search (default: 0.0 to 10.0)
        num_steps : int, optional
            Number of load steps (default: 100)
        
        Returns
        -------
        Tuple[float, int, str]
            - Load factor at FPF
            - Failed layer index
            - Failure mode
        """
        load_factors = np.linspace(load_factor_range[0], load_factor_range[1], num_steps)
        
        for lf in load_factors:
            N_scaled = lf * N
            M_scaled = lf * M
            
            eps0, kappa = self.laminate.get_midplane_strains(N_scaled, M_scaled)
            failure_data = self.calculate_failure_indices(eps0, kappa)
            
            # Check if any layer has failed
            for layer_data in failure_data:
                if layer_data['max_index'] >= 1.0:
                    return lf, layer_data['layer'], layer_data['critical_mode']
        
        return None, None, None
    
    def progressive_failure_analysis(
        self,
        N: np.ndarray,
        M: np.ndarray,
        degradation_factors: Optional[Dict] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Dict:
        """
        Perform progressive failure analysis (PFA) on the laminate.
        
        This iteratively:
        1. Calculates failure indices
        2. Degrades failed layers
        3. Recalculates laminate stiffness
        4. Repeats until convergence or total failure
        
        Parameters
        ----------
        N : np.ndarray
            Applied force resultant vector [Nₓ, Nᵧ, Nₓᵧ]
        M : np.ndarray
            Applied moment resultant vector [Mₓ, Mᵧ, Mₓᵧ]
        degradation_factors : Dict, optional
            Custom degradation factors for each failure mode
        max_iterations : int, optional
            Maximum number of iterations (default: 100)
        tolerance : float, optional
            Convergence tolerance (default: 1e-6)
        
        Returns
        -------
        Dict
            Results containing:
            - 'converged': bool
            - 'iterations': int
            - 'failed_layers': List of failed layer indices
            - 'failure_sequence': List of (iteration, layer, mode) tuples
            - 'final_strains': (eps0, kappa)
            - 'final_failure_indices': List[Dict]
        """
        if degradation_factors is None:
            degradation_factors = DEFAULT_DEGRADATION_FACTORS
        
        # Create a working copy of the laminate
        working_laminate = copy.deepcopy(self.laminate)
        hashin = HashinFailure(working_laminate)
        
        failed_layers = set()
        failure_sequence = []
        
        for iteration in range(max_iterations):
            # Calculate current strains
            eps0, kappa = working_laminate.get_midplane_strains(N, M)
            
            # Calculate failure indices
            failure_data = hashin.calculate_failure_indices(eps0, kappa)
            
            # Find newly failed layers
            newly_failed = []
            for layer_data in failure_data:
                layer_idx = layer_data['layer']
                if layer_idx not in failed_layers and layer_data['max_index'] >= 1.0:
                    newly_failed.append((layer_idx, layer_data['critical_mode']))
            
            # If no new failures, analysis converged
            if not newly_failed:
                return {
                    'converged': True,
                    'iterations': iteration,
                    'failed_layers': list(failed_layers),
                    'failure_sequence': failure_sequence,
                    'final_strains': (eps0, kappa),
                    'final_failure_indices': failure_data
                }
            
            # Degrade newly failed layers
            for layer_idx, mode in newly_failed:
                failed_layers.add(layer_idx)
                failure_sequence.append((iteration, layer_idx, mode))
                
                # Apply degradation
                layer = working_laminate.layup.layers[layer_idx]
                material = layer.material
                factors = degradation_factors[mode]
                
                # Degrade material properties
                material.E1 *= factors['E1']
                material.E2 *= factors['E2']
                material.G12 *= factors['G12']
                material.nu12 *= factors['nu12']
                
                # Recalculate material stiffness matrix
                material._compute_stiffness()
            
            # Recalculate laminate ABD matrix
            working_laminate._compute_ABD()
            
            # Check for total failure (all layers failed)
            if len(failed_layers) == len(working_laminate.layup.layers):
                return {
                    'converged': False,
                    'iterations': iteration + 1,
                    'failed_layers': list(failed_layers),
                    'failure_sequence': failure_sequence,
                    'final_strains': (eps0, kappa),
                    'final_failure_indices': failure_data,
                    'total_failure': True
                }
        
        # Maximum iterations reached
        eps0, kappa = working_laminate.get_midplane_strains(N, M)
        failure_data = hashin.calculate_failure_indices(eps0, kappa)
        
        return {
            'converged': False,
            'iterations': max_iterations,
            'failed_layers': list(failed_layers),
            'failure_sequence': failure_sequence,
            'final_strains': (eps0, kappa),
            'final_failure_indices': failure_data,
            'max_iterations_reached': True
        }
    
    def summary(
        self,
        eps0: np.ndarray,
        kappa: np.ndarray,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Generate a summary of failure analysis results.
        
        Parameters
        ----------
        eps0 : np.ndarray
            Midplane strain vector
        kappa : np.ndarray
            Curvature vector
        verbose : bool, optional
            Print summary to console (default: True)
        
        Returns
        -------
        List[Dict]
            Failure indices for all layers
        """
        failure_data = self.calculate_failure_indices(eps0, kappa)
        
        if verbose:
            print("\n" + "="*80)
            print("HASHIN FAILURE ANALYSIS SUMMARY")
            print("="*80)
            
            for data in failure_data:
                print(f"\nLayer {data['layer']} (z = {data['z']:.6f} mm):")
                print(f"  Fiber Tension (FT):        {data['FT']:.6f}")
                print(f"  Fiber Compression (FC):    {data['FC']:.6f}")
                print(f"  Matrix Tension (MT):       {data['MT']:.6f}")
                print(f"  Matrix Compression (MC):   {data['MC']:.6f}")
                print(f"  Maximum Index:             {data['max_index']:.6f}")
                print(f"  Critical Mode:             {data['critical_mode']}")
                
                if data['max_index'] >= 1.0:
                    print(f"  *** FAILURE DETECTED ***")
            
            print("\n" + "="*80)
        
        return failure_data


def print_pfa_results(results: Dict) -> None:
    """
    Print formatted progressive failure analysis results.
    
    Parameters
    ----------
    results : Dict
        Results dictionary from progressive_failure_analysis()
    """
    print("\n" + "="*80)
    print("PROGRESSIVE FAILURE ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nConverged: {results['converged']}")
    print(f"Iterations: {results['iterations']}")
    print(f"Number of failed layers: {len(results['failed_layers'])}")
    
    if results['failure_sequence']:
        print("\nFailure Sequence:")
        for iteration, layer, mode in results['failure_sequence']:
            print(f"  Iteration {iteration}: Layer {layer} failed in {mode} mode")
    
    if 'total_failure' in results:
        print("\n*** TOTAL LAMINATE FAILURE ***")
    
    if 'max_iterations_reached' in results:
        print("\n*** WARNING: Maximum iterations reached without convergence ***")
    
    print("\n" + "="*80)
