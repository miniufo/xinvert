# -*- coding: utf-8 -*-
"""High-level application defaults should reach the common template."""
import pytest

import xinvert.apps as apps


@pytest.mark.parametrize('func', [apps.invert_omega, apps.invert_3DOcean])
def test_3d_apps_accept_default_model_parameters(monkeypatch, func):
    sentinel = object()

    def fake_template(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(apps, '__template', fake_template)
    result = func(object(), dims=['lev', 'lat', 'lon'])
    assert result is sentinel


def test_multigrid_is_explicitly_experimental():
    with pytest.warns(UserWarning, match='experimental'):
        with pytest.raises(ValueError, match='dims='):
            apps.invert_MultiGrid(apps.invert_Poisson, object())
