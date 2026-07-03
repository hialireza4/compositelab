"""Tests for isotropic material classes."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from compositelab.materials import Isotropic, TransverselyIsotropic


class TestIsotropic:
    """Tests for Isotropic class."""
    
    @pytest.fixture
    def aluminum(self):
        """Standard aluminum material."""
        return Isotropic(E=70e9, nu=0.3, name="Aluminum")
    
    def test_initialization_with_E_nu(self):
        """Test initialization with E and nu."""
        mat = Isotropic(E=70e9, nu=0.3)
        assert mat.E == 70e9
        assert mat.nu == 0.3
        assert mat.G is not None
    
    def test_initialization_with_E_G(self):
        """Test initialization with E and G."""
        mat = Isotropic(E=70e9, G=27e9)
        assert mat.E == 70e9
        assert mat.G == 27e9
        assert mat.nu is not None
    
    def test_initialization_fails_without_enough_params(self):
        """Test that initialization fails without enough parameters."""
        with pytest.raises(ValueError):
            Isotropic(E=70e9)  # Missing nu or G
    
    def test_shear_modulus_calculation(self, aluminum):
        """Test shear modulus calculation from E and nu."""
        G_expected = aluminum.E / (2 * (1 + aluminum.nu))
        assert_allclose(aluminum.G, G_expected)
    
    def test_stiffness_matrix_plane_stress(self):
        """Test stiffness matrix in plane stress (default)."""
        mat = Isotropic(E=70e9, nu=0.3, plane_stress=True)
        Q = mat.stiffness_matrix
        
        assert Q.shape == (3, 3)
        
        # Q11 = Q22 = E / (1 - nu^2)
        Q11_expected = mat.E / (1 - mat.nu**2)
        assert_allclose(Q[0, 0], Q11_expected)
        assert_allclose(Q[1, 1], Q11_expected)
        
        # Q12 = nu * E / (1 - nu^2)
        Q12_expected = mat.nu * mat.E / (1 - mat.nu**2)
        assert_allclose(Q[0, 1], Q12_expected)
        
        # Q33 = G
        assert_allclose(Q[2, 2], mat.G)
    
    def test_stiffness_matrix_plane_strain(self):
        """Test stiffness matrix in plane strain."""
        mat = Isotropic(E=70e9, nu=0.3, plane_stress=False)
        Q = mat.stiffness_matrix
        
        assert Q.shape == (3, 3)
        
        # Different formula for plane strain
        factor = mat.E / ((1 + mat.nu) * (1 - 2*mat.nu))
        Q11_expected = factor * (1 - mat.nu)
        
        assert_allclose(Q[0, 0], Q11_expected, rtol=1e-10)
    
    def test_stiffness_symmetry(self, aluminum):
        """Test that stiffness matrix is symmetric."""
        Q = aluminum.stiffness_matrix
        assert_allclose(Q, Q.T)
    
    def test_compliance_stiffness_inverse(self, aluminum):
        """Test that compliance = inverse of stiffness."""
        Q = aluminum.stiffness_matrix
        S = aluminum.compliance_matrix
        
        identity = np.eye(3)
        assert_allclose(Q @ S, identity, rtol=1e-10)
    
    def test_isotropic_properties(self, aluminum):
        """Test that Q11 = Q22 for isotropic material."""
        Q = aluminum.stiffness_matrix
        assert_allclose(Q[0, 0], Q[1, 1])


class TestTransverselyIsotropic:
    """Tests for TransverselyIsotropic class."""
    
    @pytest.fixture
    def ti_material(self):
        """Standard transversely isotropic material."""
        return TransverselyIsotropic(
            E_axial=150e9,
            E_transverse=10e9,
            nu_axial=0.3,
            nu_transverse=0.4,
            G_axial=5e9,
            name="TI_Material"
        )
    
    def test_initialization(self, ti_material):
        """Test basic initialization."""
        assert ti_material.E_axial == 150e9
        assert ti_material.E_transverse == 10e9
        assert ti_material.G_axial == 5e9
    
    def test_transverse_shear_modulus_calculation(self, ti_material):
        """Test G_transverse calculation."""
        G_trans_expected = ti_material.E_transverse / (2 * (1 + ti_material.nu_transverse))
        assert_allclose(ti_material.G_transverse, G_trans_expected)
    
    def test_stiffness_matrix_shape(self, ti_material):
        """Test stiffness matrix has correct shape (6x6)."""
        C = ti_material.stiffness_matrix
        assert C.shape == (6, 6)
    
    def test_stiffness_matrix_symmetry(self, ti_material):
        """Test stiffness matrix is symmetric."""
        C = ti_material.stiffness_matrix
        assert_allclose(C, C.T, rtol=1e-10)
    
    def test_transverse_isotropy_property(self, ti_material):
        """Test that C22 = C33 (transverse isotropy)."""
        C = ti_material.stiffness_matrix
        assert_allclose(C[1, 1], C[2, 2], rtol=1e-10)
    
    def test_compliance_stiffness_inverse(self, ti_material):
        """Test that compliance = inverse of stiffness."""
        C = ti_material.stiffness_matrix
        S = ti_material.compliance_matrix
        
        identity = np.eye(6)
        assert_allclose(C @ S, identity, rtol=1e-9)
