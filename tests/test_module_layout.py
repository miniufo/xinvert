"""Canonical module ownership and backward-compatible imports."""
import xinvert
from xinvert import apps, utils
from xinvert.animate import record_iteration


def test_moved_public_functions_keep_import_compatibility():
    assert xinvert.record_iteration is apps.record_iteration is record_iteration
    assert xinvert.cal_flow is apps.cal_flow is utils.cal_flow
    assert apps.invert_MultiGrid is utils.invert_MultiGrid
    assert apps._invert_omega_MG is utils._invert_omega_MG
    assert record_iteration.__module__ == 'xinvert.animate'
    assert utils.cal_flow.__module__ == 'xinvert.utils'


def test_animation_public_names_are_unambiguous():
    from xinvert import create_animation, record_iteration
    assert record_iteration is xinvert.record_iteration
    assert create_animation is utils.create_animation
    assert not hasattr(xinvert, 'animate_iteration')
    assert not hasattr(utils, 'createAnimation')
