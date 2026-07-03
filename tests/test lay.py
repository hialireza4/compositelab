from compositelab.materials import Orthotropic2D,Orthotropic3D
from compositelab.layers import Layup
from compositelab.mechanics import Laminate
from compositelab.mechanics.statevisualizer import StateVisualizer
from compositelab.failure.hashin import HashinFailureCriterion, check_hashin_failure
import numpy as np
# Define material
carbon = Orthotropic2D(
    E1=150e9, E2=10e9, nu12=0.3, G12=5e9,
    Xt=1500e6, Xc=1200e6, Yt=50e6, Yc=250e6, S=70e6,
    thickness=0.125e-3, name="Carbon/Epoxy"
)



# Create layup
layup = Layup.from_stacking('[45/60/0]s', carbon, 0.1e-3)

# Create laminate for analysis
laminate1 = Laminate(layup)
# Access properties
print("A matrix:")
print(laminate1.A)

print("\nIs symmetric?", laminate1.is_symmetric())

# Apply loads
N = np.array([100000, 0, 0])  # N/mm
M = np.array([0, 0, 0])     # N·mm/mm

eps0, kappa = laminate1.get_midplane_strains(N, M)
print(f"\nMidplane strains: {eps0}")
print(f"Curvatures: {kappa}")

# Get stresses in specific layer
stress_layer_0 = laminate1.get_layer_stresses(0, eps0, kappa, surface='top')
print(f"\nStress in layer 0 (top): {stress_layer_0} MPa")

# Print full summary
laminate1.summary()
print(laminate1)
layup.summary()


# viz = StateVisualizer(laminate1, N, M)

# # نمودارها
# viz.plot_all_strains()
# viz.plot_all_stresses()
# viz.plot_combined_overview()
# viz.summary_table()
result = check_hashin_failure(laminate1, N, M)
print(result['summary'])

criterion = HashinFailureCriterion(laminate1)
pfa_result = criterion.progressive_failure_analysis(N, M)
print(f"Converged: {pfa_result['converged']}")
print(f"Failed layers: {pfa_result['failed_layers']}")