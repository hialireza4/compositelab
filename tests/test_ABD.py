'''
Test of LaminateProperty
========================

Test LaminateProperty class using example from page 201 of reference 1

References
----------
1 - Mendonça, Paulo de Tarso R. Materiais Compostos e Estruturas-sanduíche: Projeto e Análise. 2005 Editora Manole Ltda.
'''

import pytest
import numpy as np

from compositelab.materials import Orthotropic2D
from compositelab.layers import Layup
from compositelab.mechanics import Laminate

@pytest.fixture
def stacking_plies():
    E1 = 54870e6 #pa
    E2 = 18320e6 #pa
    G12 = 8900e6 #pa
    v12 = 0.25
    mat = Orthotropic2D(E1, E2, v12, G12) #thickness m
    layup = Layup.from_stacking(notation='[45]',material=mat,thickness=3.0e-3)
    layup.add_layer(material=mat,thickness=6.0e-3,angle=0)
    layup.add_layer(material=mat,thickness=3.0e-3,angle=45)
    lam = Laminate(layup)
   
    return lam


def test_A(stacking_plies):
    A_reference = np.array([[515794.00930665, 100823.25453699, 55993.44346208],
                            [100823.25453699, 291820.23545835, 55993.44346208],
                            [ 55993.44346208, 55993.44346208, 151491.93392275]])
    A_reference = A_reference/1e-3
    np.testing.assert_almost_equal(stacking_plies.A, A_reference,decimal=5)


def test_B(stacking_plies):
    B_reference = np.array([[0., 0., 0.],
                            [0., 0., 0.],
                            [0., 0., 0.]])
    B_reference = B_reference
    np.testing.assert_almost_equal(stacking_plies.B, B_reference)


def test_D(stacking_plies):

    D_reference = np.array([[4779418.7240577, 1612106.45974872, 1175862.31270358],
                            [1612106.45974872, 4107497.4025128, 1175862.31270358],
                            [1175862.31270358, 1175862.31270358, 2220130.61237785]])
    D_reference = D_reference/1e3
    np.testing.assert_almost_equal(stacking_plies.D, D_reference,decimal=5)

