"""Boundary contracts checked against independent forcing and scalar inversion."""
import numpy as np
import pytest
import xarray as xr

from xinvert import invert_Poisson
from xinvert.helmholtz import _li2006 as li
from xinvert.helmholtz import decompose, streamfunction_potential


def field():
    y, x = np.arange(10.), np.arange(12.)
    u = xr.DataArray(np.sin(x[None, :]*.3)*np.cos(y[:, None]*.2),
                     dims=('y','x'), coords={'y':y,'x':x})
    v = .7*u.roll(x=1)
    wet = xr.ones_like(u, dtype=bool)
    wet[4:6,5] = False
    return u, v, wet


def forcing_oracle(u, v, wet, periodic):
    a, b, mask = u.values, v.values, wet.values
    east, north = np.zeros_like(a), np.zeros_like(b)
    ny,nx = a.shape
    for j,i in np.ndindex(a.shape):
        if mask[j,i] and (i+1<nx or periodic) and mask[j,(i+1)%nx]:
            east[j,i] = .5*(a[j,i]+a[j,(i+1)%nx])
        if mask[j,i] and j+1<ny and mask[j+1,i]:
            north[j,i] = .5*(b[j,i]+b[j+1,i])
    rhs = np.zeros_like(a)
    for j,i in np.ndindex(a.shape):
        rhs[j,i] = east[j,i]-(east[j,(i-1)%nx] if i>0 or periodic else 0)
        rhs[j,i] += north[j,i]-(north[j-1,i] if j>0 else 0)
    return xr.DataArray(rhs,dims=u.dims,coords=u.coords).where(wet)


@pytest.mark.parametrize('bcx',['fixed','extend','periodic'])
@pytest.mark.parametrize('internal',['fixed','extend','neumann'])
def test_sor_matches_standard_inversion(bcx, internal, architect):
    u,v,wet=field()
    initial=(u+v+2).where(wet)
    opts={'BCs':['fixed',bcx],'internal_bcs':internal,'architect':architect,
          'threads':2,'mxLoop':6000,'tolerance':1e-9,'printInfo':False}
    saved=opts.copy()
    reference=invert_Poisson(forcing_oracle(u,v,wet,bcx=='periodic'),['y','x'],
                             'cartesian',icbc=initial,iParams=opts)
    result=decompose(u,v,['y','x'],'cartesian',method='sor',mask=wet,
                      initial=initial,iParams=opts,return_potentials=True)
    xr.testing.assert_allclose(result.velocity_potential,reference,rtol=1e-10,atol=1e-11)
    assert opts==saved
    xr.testing.assert_allclose(result.urot+result.udiv,u.where(wet))
    xr.testing.assert_allclose(result.vrot+result.vdiv,v.where(wet))
    if internal=='fixed':
        assert float(result.velocity_potential[4,4])==float(initial[4,4])


def test_standard_extend_alias_is_exact_and_case_insensitive():
    u,v,wet=field()
    f=forcing_oracle(u,v,wet,False)
    opts={'BCs':[' FIXED ','Extend'],'mxLoop':3000,'tolerance':1e-9,'printInfo':False}
    a=invert_Poisson(f,['y','x'],'cartesian',iParams={**opts,'internal_bcs':'ExTeNd'})
    b=invert_Poisson(f,['y','x'],'cartesian',iParams={**opts,'internal_bcs':'neumann'})
    xr.testing.assert_identical(a,b)


def test_li_default_uses_solid_backend_and_public_fixed_names():
    u,v,wet=field()
    opts={'mxLoop':3000,'tolerance':1e-9,'on_nonconvergence':'raise'}
    out=streamfunction_potential(u,v,['y','x'],'cartesian',mask=wet,iParams=opts)
    psi,chi=li.helmholtz(u,v,['y','x'],'cartesian',mask=wet,
                        iParams={**opts,'BCs':['closed','closed']})
    xr.testing.assert_allclose(out.streamfunction,psi)
    xr.testing.assert_allclose(out.velocity_potential,chi)
    assert out.attrs['boundary_y']==out.attrs['boundary_x']=='fixed'
    assert 'BCs' not in opts


@pytest.mark.parametrize('bcs',[['open','open'],['closed','closed']])
def test_li_public_rejects_old_boundary_names(bcs):
    u,v,wet=field()
    with pytest.raises(ValueError,match='expected one of'):
        decompose(u,v,['y','x'],'cartesian',mask=wet,iParams={'BCs':bcs})


def test_li_extend_and_manual_coast_controls_are_explicit_errors():
    u,v,wet=field()
    with pytest.raises(NotImplementedError,match='extend'):
        decompose(u,v,['y','x'],'cartesian',mask=wet,iParams={'BCs':['fixed','extend']})
    with pytest.raises(ValueError,match='automatically'):
        decompose(u,v,['y','x'],'cartesian',mask=wet,iParams={'internal_bcs':'extend'})


def test_sor_boundary_names_are_normalized_in_metadata():
    u,v,wet=field()
    opts={'BCs':[' FIXED ','PERIODIC'],'internal_bcs':'ExTeNd',
          'mxLoop':3000,'tolerance':1e-9,'printInfo':False}
    result=decompose(u,v,['y','x'],'cartesian',method='sor',mask=wet,iParams=opts)
    assert result.attrs['boundary_y']=='fixed'
    assert result.attrs['boundary_x']=='periodic'
    assert opts['BCs']==[' FIXED ','PERIODIC']
