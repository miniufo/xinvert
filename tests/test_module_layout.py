"""Canonical module ownership and backward-compatible imports."""
import xinvert
from xinvert import apps, utils
from xinvert.animate import animate_iteration


def test_moved_public_functions_keep_import_compatibility():
    assert xinvert.animate_iteration is apps.animate_iteration is animate_iteration
    assert xinvert.cal_flow is apps.cal_flow is utils.cal_flow
    assert apps.invert_MultiGrid is utils.invert_MultiGrid
    assert apps._invert_omega_MG is utils._invert_omega_MG
    assert animate_iteration.__module__ == 'xinvert.animate'
    assert utils.cal_flow.__module__ == 'xinvert.utils'
