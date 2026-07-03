"""Tests for orthotropic material classes."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from compositelab.materials import Orthotropic2D, Orthotropic3D


class TestOrthotropic2D:
    """Tests for Orthotropic2D class."""
    
    @pytest.fixture
    def carbon_epoxy(self):
        """Standard carbon/epoxy material."""
        return Orthotropic2D(
            E1=150e9,
            E2=10e9,
            G12=5e9,
            nu12=0.3,
            name="Carbon/Epoxy"
        )
    
    def test_initialization(self, carbon_epoxy):
        """Test basic initialization."""
        assert carbon_epoxy.name == "Carbon/Epoxy"
        assert carbon_epoxy.E1 == 150e9
        assert carbon_epoxy.E2 == 10e9
        assert carbon_epoxy.G12 == 5e9
        assert carbon_epoxy.nu12 == 0.3
    
    def test_nu21_calculation(self, carbon_epoxy):
        """Test nu21 is calculated correctly from reciprocal relation."""
        nu21_expected = carbon_epoxy.nu12 * carbon_epoxy.E2 / carbon_epoxy.E1
        assert_allclose(carbon_epoxy.nu21, nu21_expected)
    
    def test_stiffness_matrix_shape(self, carbon_epoxy):
        """Test stiffness matrix has correct shape."""
        Q = carbon_epoxy.stiffness_matrix
        assert Q.shape == (3, 3)
    
    def test_stiffness_matrix_symmetry(self, carbon_epoxy):
        """Test stiffness matrix is symmetric."""
        Q = carbon_epoxy.stiffness_matrix
        assert_allclose(Q, Q.T, rtol=1e-10)
    
    def test_compliance_stiffness_inverse(self, carbon_epoxy):
        """Test that compliance = inverse of stiffness."""
        Q = carbon_epoxy.stiffness_matrix
        S = carbon_epoxy.compliance_matrix
        
        identity = np.eye(3)
        assert_allclose(Q @ S, identity, rtol=1e-10)
        assert_allclose(S @ Q, identity, rtol=1e-10)
    
    def test_rotation_at_zero_degrees(self, carbon_epoxy):
        """Test rotation at 0° returns original matrix."""
        Q_original = carbon_epoxy.stiffness_matrix
        Q_rotated = carbon_epoxy.get_rotated_stiffness_matrix(0)
        
        assert_allclose(Q_rotated, Q_original, rtol=1e-10)
    
    def test_rotation_at_90_degrees(self, carbon_epoxy):
        """Test rotation at 90° swaps E1 and E2."""
        Q_90 = carbon_epoxy.get_rotated_stiffness_matrix(90)
        
        # At 90°, Q11 should equal original Q22
        Q_original = carbon_epoxy.stiffness_matrix
        assert_allclose(Q_90[0, 0], Q_original[1, 1], rtol=1e-6)
        assert_allclose(Q_90[1, 1], Q_original[0, 0], rtol=1e-6)
    
    def test_rotation_symmetry(self, carbon_epoxy):
        """Test that rotating by θ then -θ returns original."""
        Q_original = carbon_epoxy.stiffness_matrix
        Q_rotated = carbon_epoxy.get_rotated_stiffness_matrix(45)
        Q_back = carbon_epoxy.get_rotated_stiffness_matrix(-45)
        
        # Apply both rotations
        from compositelab.materials.orthotropic import transformation_matrix
        T45 = transformation_matrix(45)
        Tn45 = transformation_matrix(-45)
        
        Q_final = Tn45.T @ T45.T @ Q_original @ T45 @ Tn45
        assert_allclose(Q_final, Q_original, rtol=1e-10)
    
    def test_invariant_tsai_pagano(self, carbon_epoxy):
        """Test Tsai-Pagano invariants are rotation-independent."""
        inv_0 = carbon_epoxy.invariant_tsai_pagano(0)
        inv_30 = carbon_epoxy.invariant_tsai_pagano(30)
        inv_45 = carbon_epoxy.invariant_tsai_pagano(45)
        inv_90 = carbon_epoxy.invariant_tsai_pagano(90)
        
        for key in inv_0.keys():
            assert_allclose(inv_0[key], inv_30[key], rtol=1e-10)
            assert_allclose(inv_0[key], inv_45[key], rtol=1e-10)
            assert_allclose(inv_0[key], inv_90[key], rtol=1e-10)
    
    def test_isotropic_limit(self):
        """Test that isotropic material (E1=E2, nu12=nu21) behaves correctly."""
        E = 70e9
        nu = 0.3
        G = E / (2 * (1 + nu))
        
        iso_as_ortho = Orthotropic2D(
            E1=E,
            E2=E,
            G12=G,
            nu12=nu,
            name="Isotropic_test"
        )
        
        Q = iso_as_ortho.stiffness_matrix
        
        # For isotropic: Q11 = Q22
        assert_allclose(Q[0, 0], Q[1, 1], rtol=1e-10)
        
        # For isotropic: Q12 = Q11 * nu / (1 - nu)
        Q12_expected = Q[0, 0] * nu
        assert_allclose(Q[0, 1], Q12_expected, rtol=1e-6)
    
    def test_with_density(self):
        """Test material with density."""
        mat = Orthotropic2D(
            E1=150e9,
            E2=10e9,
            G12=5e9,
            nu12=0.3,
            density=1600,
            name="With_density"
        )
        assert mat.density == 1600
    
    def test_with_strength_properties(self):
        """Test material with strength properties."""
        mat = Orthotropic2D(
            E1=150e9,
            E2=10e9,
            G12=5e9,
            nu12=0.3,
            Xt=1500e6,
            Xc=1200e6,
            Yt=50e6,
            Yc=200e6,
            S=70e6,
            name="With_strength"
        )
        assert mat.Xt == 1500e6
        assert mat.Yc == 200e6
        assert mat.S == 70e6


class TestOrthotropic3D:
    """Tests for Orthotropic3D class."""
    
    @pytest.fixture
    def carbon_epoxy_3d(self):
        """Standard 3D carbon/epoxy material."""
        return Orthotropic3D(
            E1=150e9,
            E2=10e9,
            E3=10e9,
            G12=5e9,
            G13=5e9,
            G23=3.5e9,
            nu12=0.3,
            nu13=0.3,
            nu23=0.4,
            name="Carbon/Epoxy_3D"
        )
    
    def test_initialization(self, carbon_epoxy_3d):
        """Test basic initialization."""
        assert carbon_epoxy_3d.E3 == 10e9
        assert carbon_epoxy_3d.G23 == 3.5e9
        assert carbon_epoxy_3d.nu23 == 0.4
    
    def test_stiffness_matrix_shape(self, carbon_epoxy_3d):
        """Test stiffness matrix has correct shape (6x6)."""
        C = carbon_epoxy_3d.stiffness_matrix
        assert C.shape == (6, 6)
    
    def test_stiffness_matrix_symmetry(self, carbon_epoxy_3d):
        """Test stiffness matrix is symmetric."""
        C = carbon_epoxy_3d.stiffness_matrix
        assert_allclose(C, C.T, rtol=1e-10)
    
    def test_compliance_stiffness_inverse(self, carbon_epoxy_3d):
        """Test that compliance = inverse of stiffness."""
        C = carbon_epoxy_3d.stiffness_matrix
        S = carbon_epoxy_3d.compliance_matrix
        
        identity = np.eye(6)
        assert_allclose(C @ S, identity, rtol=1e-10)
        assert_allclose(S @ C, identity, rtol=1e-10)
    
    def test_reciprocal_relations(self, carbon_epoxy_3d):
        """Test reciprocal relations for Poisson's ratios."""
        nu21 = carbon_epoxy_3d.nu12 * carbon_epoxy_3d.E2 / carbon_epoxy_3d.E1
        nu31 = carbon_epoxy_3d.nu13 * carbon_epoxy_3d.E3 / carbon_epoxy_3d.E1
        nu32 = carbon_epoxy_3d.nu23 * carbon_epoxy_3d.E3 / carbon_epoxy_3d.E2
        
        assert_allclose(carbon_epoxy_3d.nu21, nu21)
        assert_allclose(carbon_epoxy_3d.nu31, nu31)
        assert_allclose(carbon_epoxy_3d.nu32, nu32)
