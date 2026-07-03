"""
Test script for Hashin failure criterion with progressive failure analysis.
Displays failure progression and plots Nx vs axial strain until complete failure.
"""

import numpy as np
import matplotlib.pyplot as plt

from compositelab.materials.orthotropic import Orthotropic2D
from compositelab.layers.layer import Layer
from compositelab.layers.layup import Layup
from compositelab.mechanics.laminate import Laminate
from compositelab.failure.hashin import HashinFailure


def apply_degradation_to_layer(layer, mode, degradation_factors):
    """
    Apply material property degradation to a specific layer.
    
    Parameters
    ----------
    layer : Layer
        The layer to degrade
    mode : str
        Failure mode ('FT', 'FC', 'MT', 'MC')
    degradation_factors : dict
        Degradation factors for each mode
    """
    factors = degradation_factors.get(mode, {})
    material = layer.material
    
    # Apply degradation
    material.E1 *= factors.get('E1', 1.0)
    material.E2 *= factors.get('E2', 1.0)
    material.G12 *= factors.get('G12', 1.0)
    material.nu12 *= factors.get('nu12', 1.0)
    
    # Force recalculation of stiffness matrix
    if hasattr(material, '_stiffness_matrix'):
        delattr(material, '_stiffness_matrix')
    
    # Recalculate stiffness (if method exists)
    if hasattr(material, '_compute_stiffness'):
        material._compute_stiffness()


def main():
    """
    Progressive failure test: Load until complete laminate failure.
    Track and plot each failure point.
    """
    
    print("="*80)
    print("HASHIN PROGRESSIVE FAILURE ANALYSIS - TENSION TEST")
    print("="*80)
    
    # ========================
    # Material Definition
    # ========================
    carbon_epoxy = Orthotropic2D(
        name="Carbon/Epoxy T300/5208",
        E1=181e3,      # MPa
        E2=10.3e3,     # MPa
        G12=7.17e3,    # MPa
        nu12=0.28,
        Xt=1500,       # MPa
        Xc=1500,       # MPa
        Yt=40,         # MPa
        Yc=246,        # MPa
        S=68           # MPa
    )
    
    print(f"\nMaterial: {carbon_epoxy.name}")
    
    # ========================
    # Laminate Definition
    # ========================
    ply_thickness = 0.125  # mm
    
    # Symmetric laminate [0/45/-45/90]s
    layers = [
        Layer(carbon_epoxy, ply_thickness, angle=0),
        Layer(carbon_epoxy, ply_thickness, angle=45),
        Layer(carbon_epoxy, ply_thickness, angle=-45),
        Layer(carbon_epoxy, ply_thickness, angle=90),
        Layer(carbon_epoxy, ply_thickness, angle=90),
        Layer(carbon_epoxy, ply_thickness, angle=-45),
        Layer(carbon_epoxy, ply_thickness, angle=45),
        Layer(carbon_epoxy, ply_thickness, angle=0),
    ]
    
    layup = Layup(layers)
    laminate = Laminate(layup)
    
    print(f"\nLaminate: [0/45/-45/90]s")
    print(f"  Number of plies: {len(layers)}")
    print(f"  Total thickness: {layup.total_thickness:.3f} mm")
    
    # ========================
    # Degradation Factors
    # ========================
    degradation_factors = {
        'FT': {'E1': 0.01, 'E2': 1.0, 'G12': 0.01, 'nu12': 0.01},
        'FC': {'E1': 0.01, 'E2': 1.0, 'G12': 0.01, 'nu12': 0.01},
        'MT': {'E1': 1.0, 'E2': 0.01, 'G12': 0.01, 'nu12': 0.01},
        'MC': {'E1': 1.0, 'E2': 0.01, 'G12': 0.01, 'nu12': 0.01}
    }
    
    # ========================
    # Initialize Hashin
    # ========================
    hashin = HashinFailure(laminate)
    
    # ========================
    # Progressive Loading with Degradation
    # ========================
    print("\n" + "="*80)
    print("PROGRESSIVE FAILURE ANALYSIS")
    print("="*80)
    
    # Storage for plotting
    all_Nx = []
    all_strain = []
    failure_points_Nx = []
    failure_points_strain = []
    failure_labels = []
    
    # Tracking
    failed_layers = set()
    iteration = 0
    max_iterations = 100
    
    # Load increment
    delta_N = 10.0  # N/mm per step
    N_current = 0.0
    M = np.array([0.0, 0.0, 0.0])
    
    print("\nStarting progressive loading...")
    print(f"Load increment: {delta_N} N/mm per step\n")
    
    while iteration < max_iterations:
        iteration += 1
        N_current += delta_N
        N = np.array([N_current, 0.0, 0.0])
        
        # Calculate strains
        try:
            eps0, kappa = laminate.get_midplane_strains(N, M)
        except np.linalg.LinAlgError:
            print(f"\n*** COMPLETE LAMINATE FAILURE ***")
            print(f"Laminate stiffness matrix is singular - unable to carry load")
            print(f"Final load: Nx = {N_current - delta_N:.2f} N/mm")
            print(f"Final strain: εx = {all_strain[-1]:.1f} με")
            break
        
        axial_strain = eps0[0] * 1e6  # microstrain
        
        # Store current state
        all_Nx.append(N_current)
        all_strain.append(axial_strain)
        
        # Check failure
        failure_data = hashin.calculate_failure_indices(eps0, kappa)
        
        # Identify newly failed layers
        newly_failed = []
        for data in failure_data:
            layer_idx = data['layer']
            if layer_idx not in failed_layers and data['max_index'] >= 1.0:
                newly_failed.append(data)
                failed_layers.add(layer_idx)
        
        # If new failures, degrade and mark
        if newly_failed:
            for data in newly_failed:
                layer_idx = data['layer']
                mode = data['critical_mode']
                fi = data['max_index']
                angle = layers[layer_idx].angle
                
                print(f"Iteration {iteration}: FAILURE at Nx = {N_current:.2f} N/mm, εx = {axial_strain:.1f} με")
                print(f"  → Layer {layer_idx} (θ={angle}°), Mode: {mode}, FI = {fi:.3f}")
                
                # Mark failure point
                failure_points_Nx.append(N_current)
                failure_points_strain.append(axial_strain)
                failure_labels.append(f"Layer {layer_idx}\n{mode}")
                
                # Apply degradation
                apply_degradation_to_layer(layers[layer_idx], mode, degradation_factors)
            
            # Recompute laminate ABD after all degradations
            laminate._compute_ABD()
            
            print(f"  Total failed layers: {len(failed_layers)}/{len(layers)}\n")
        
        # Check for complete failure
        if len(failed_layers) == len(layers):
            print(f"\n*** COMPLETE LAMINATE FAILURE ***")
            print(f"All {len(layers)} layers have failed")
            print(f"Final load: Nx = {N_current:.2f} N/mm")
            print(f"Final strain: εx = {axial_strain:.1f} με")
            break
        
        # Safety check: unrealistic strain
        if axial_strain > 50000:  # 5% strain
            print(f"\n*** EXCESSIVE DEFORMATION ***")
            print(f"Axial strain exceeded 5% - terminating")
            print(f"Final load: Nx = {N_current:.2f} N/mm")
            print(f"Final strain: εx = {axial_strain:.1f} με")
            break
    
    # ========================
    # Summary
    # ========================
    print("\n" + "="*80)
    print("FAILURE PROGRESSION SUMMARY")
    print("="*80)
    
    if failure_points_Nx:
        print(f"\nFirst Ply Failure (FPF):")
        print(f"  Nx = {failure_points_Nx[0]:.2f} N/mm")
        print(f"  εx = {failure_points_strain[0]:.1f} με")
        print(f"  {failure_labels[0].replace(chr(10), ' ')}")
        
        if len(failure_points_Nx) > 1:
            print(f"\nSubsequent Failures: {len(failure_points_Nx) - 1}")
            for i in range(1, len(failure_points_Nx)):
                print(f"  {i}. Nx = {failure_points_Nx[i]:.2f} N/mm, "
                      f"εx = {failure_points_strain[i]:.1f} με - "
                      f"{failure_labels[i].replace(chr(10), ' ')}")
        
        print(f"\nComplete Failure:")
        print(f"  Nx = {all_Nx[-1]:.2f} N/mm")
        print(f"  εx = {all_strain[-1]:.1f} με")
    
    # ========================
    # Plotting
    # ========================
    print("\n" + "="*80)
    print("GENERATING PLOT")
    print("="*80)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Main curve
    ax.plot(all_strain, all_Nx, 'b-', linewidth=2.5, label='Loading path')
    
    # Mark failure points
    if failure_points_Nx:
        ax.scatter(failure_points_strain, failure_points_Nx, 
                  c='red', s=150, zorder=5, marker='X', 
                  edgecolors='darkred', linewidths=2,
                  label='Failure events')
        
        # Annotate first and last failure
        ax.annotate(f'FPF\n{failure_labels[0]}',
                   xy=(failure_points_strain[0], failure_points_Nx[0]),
                   xytext=(20, 30), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                                 color='red', lw=2))
        
        if len(failure_points_Nx) > 1:
            ax.annotate(f'Final\n{failure_labels[-1]}',
                       xy=(failure_points_strain[-1], failure_points_Nx[-1]),
                       xytext=(20, -40), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3',
                                     color='red', lw=2))
    
    ax.set_xlabel('Axial Strain εₓ [με]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Applied Load Nₓ [N/mm]', fontsize=14, fontweight='bold')
    ax.set_title('Progressive Failure Analysis - Hashin Criterion\n[0/45/-45/90]s Carbon/Epoxy Laminate',
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    
    # Add info box
    info_text = f"Material: {carbon_epoxy.name}\n"
    info_text += f"Plies: {len(layers)}, Thickness: {layup.total_thickness:.3f} mm\n"
    if failure_points_Nx:
        info_text += f"FPF: Nx={failure_points_Nx[0]:.1f} N/mm, εx={failure_points_strain[0]:.0f} με\n"
        info_text += f"Ultimate: Nx={all_Nx[-1]:.1f} N/mm, εx={all_strain[-1]:.0f} με"
    
    ax.text(0.02, 0.98, info_text,
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('progressive_failure_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: progressive_failure_analysis.png")
    plt.show()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
