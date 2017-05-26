"""
Medium parameters
"""

import dolfin as dl


def createparam(CC, Vl, X, size):
    c = dl.interpolate(dl.Expression(' \
    c0 + (c1-c0)*(x[0]>=LL)*(x[0]<=RR)*(x[1]>=BB)*(x[1]<=TT)',\
    c0 = CC[0], c1=CC[1], \
    BB=0.5*(X-size), LL=0.5*(X-size), TT=0.5*(X+size), RR=0.5*(X+size)), Vl)
    return c


def targetmediumparameters(Vl, X, myplot=None):
    """
    Arguments:
        Vl = function space
        X = x dimension of domain
    """
    # medium parameters:
    size = 0.4
    CC = [1.0, 2.0]
    RR = [2.0, 5.0]
    LL, AA, BB = [], [], []
    for cc, rr in zip(CC, RR):
        ll = rr*cc*cc
        LL.append(ll)
        AA.append(1./ll)
        BB.append(1./rr)
    # velocity is in [km/s]
    c = createparam(CC, Vl, X, size)
    if not myplot == None:
        myplot.set_varname('c_target')
        myplot.plot_vtk(c)
    # density is in [10^12 kg/km^3]=[g/cm^3]
    # assume rocks shale-sand-shale + salt inside small rectangle
    # see Marmousi2 print-out
    rho = createparam(RR, Vl, X, size)
    if not myplot == None:
        myplot.set_varname('rho_target')
        myplot.plot_vtk(rho)
    # bulk modulus is in [10^12 kg/km.s^2]=[GPa]
    lam = createparam(LL, Vl, X, size)
    if not myplot == None:
        myplot.set_varname('lambda_target')
        myplot.plot_vtk(lam)
    #
    at = createparam(AA, Vl, X, size)
    if not myplot == None:
        myplot.set_varname('alpha_target')
        myplot.plot_vtk(at)
    bt = createparam(BB, Vl, X, size)
    if not myplot == None:
        myplot.set_varname('beta_target')
        myplot.plot_vtk(bt)
    # Check:
    ones = dl.interpolate(dl.Expression('1.0'), Vl)
    check1 = at.vector() * lam.vector()
    erra = dl.norm(check1 - ones.vector())
    assert erra < 2e-14, erra
    check2 = bt.vector() * rho.vector()
    errb = dl.norm(check2 - ones.vector())
    assert errb < 1e-16, errb

    return at, bt, c, lam, rho



def initmediumparameters(Vl, X, myplot=None):
    # medium parameters:
    size = 0.4
    CC = [1.0, 1.0]
    RR = [2.0, 2.0]
    LL, AA, BB = [], [], []
    for cc, rr in zip(CC, RR):
        ll = rr*cc*cc
        LL.append(ll)
        AA.append(1./ll)
        BB.append(1./rr)
    # velocity is in [km/s]
    c = createparam(CC, Vl, X, size)
    if not myplot == None:
        myplot.set_varname('c_init')
        myplot.plot_vtk(c)
    # density is in [10^12 kg/km^3]=[g/cm^3]
    # assume rocks shale-sand-shale + salt inside small rectangle
    # see Marmousi2 print-out
    rho = createparam(RR, Vl, X, size)
    if not myplot == None:
        myplot.set_varname('rho_init')
        myplot.plot_vtk(rho)
    # bulk modulus is in [10^12 kg/km.s^2]=[GPa]
    lam = createparam(LL, Vl, X, size)
    if not myplot == None:
        myplot.set_varname('lambda_init')
        myplot.plot_vtk(lam)
    #
    at = createparam(AA, Vl, X, size)
    if not myplot == None:
        myplot.set_varname('alpha_init')
        myplot.plot_vtk(at)
    bt = createparam(BB, Vl, X, size)
    if not myplot == None:
        myplot.set_varname('beta_init')
        myplot.plot_vtk(bt)
    # Check:
    ones = dl.interpolate(dl.Expression('1.0'), Vl)
    check1 = at.vector() * lam.vector()
    erra = dl.norm(check1 - ones.vector())
    assert erra < 1e-16, erra
    check2 = bt.vector() * rho.vector()
    errb = dl.norm(check2 - ones.vector())
    assert errb < 1e-16, errb

    return at, bt, c, lam, rho



def loadparameters(LARGE):
    if LARGE:
        Nxy = 125
        Dt = 2.5e-4   #Dt = h/(r*alpha)
        fpeak = 4.0
        t0, t1, t2, tf = 0.0, 0.2, 0.8, 1.0
    else:
        Nxy = 40
        Dt = 1.0e-3
        fpeak = 1.0
        t0, t1, t2, tf = 0.0, 0.5, 2.5, 3.0
    return Nxy, Dt, fpeak, t0, t1, t2, tf
