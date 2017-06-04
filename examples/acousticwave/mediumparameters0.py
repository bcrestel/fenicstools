"""
Medium parameters
"""

import numpy as np
import dolfin as dl


def createparam(CC, Vl, X, size):
    c = dl.interpolate(dl.Expression(' \
    c0 + (c1-c0)*(x[0]>=LL)*(x[0]<=RR)*(x[1]>=BB)*(x[1]<=TT)',\
    c0 = CC[0], c1=CC[1], \
    BB=0.5*(X-size), LL=0.5*(X-size), TT=0.5*(X+size), RR=0.5*(X+size),\
    degree=10), Vl)
    return c


def targetmediumparameters(Vl, X, myplot=None):
    """
    Arguments:
        Vl = function space
        X = x dimension of domain
    """
    # medium parameters:
    size = 0.4
    AA = [1.0, 1.4]
    BB = [1.0, 2.0]
    CC, LL, RR = [], [], []
    for aa, bb in zip(AA, BB):
        cc = np.sqrt(bb/aa)
        rr = 1./bb
        ll = 1./aa
        CC.append(cc)
        RR.append(rr)
        LL.append(ll)
    c = createparam(CC, Vl, X, size)
    if not myplot == None:
        myplot.set_varname('c_target')
        myplot.plot_vtk(c)
    rho = createparam(RR, Vl, X, size)
    if not myplot == None:
        myplot.set_varname('rho_target')
        myplot.plot_vtk(rho)
    lam = createparam(LL, Vl, X, size)
    if not myplot == None:
        myplot.set_varname('lambda_target')
        myplot.plot_vtk(lam)
    at = createparam(AA, Vl, X, size)
    if not myplot == None:
        myplot.set_varname('alpha_target')
        myplot.plot_vtk(at)
    bt = createparam(BB, Vl, X, size)
    if not myplot == None:
        myplot.set_varname('beta_target')
        myplot.plot_vtk(bt)
    # Check:
    ones = dl.interpolate(dl.Constant('1.0'), Vl)
    check1 = at.vector() * lam.vector()
    erra = dl.norm(check1 - ones.vector())
    assert erra < 1e-16, erra
    check2 = bt.vector() * rho.vector()
    errb = dl.norm(check2 - ones.vector())
    assert errb < 1e-16, errb

    return at, bt, c, lam, rho



def initmediumparameters(Vl, X, myplot=None):
    # medium parameters:
    a_initial = dl.Expression('1.0 + 0.1*sin(pi*x[0])*sin(pi*x[1])', degree=10)
    at = dl.interpolate(a_initial, Vl)
    b_initial = dl.Expression('1.0 + 0.25*sin(pi*x[0])*sin(pi*x[1])', degree=10)
    bt = dl.interpolate(b_initial, Vl)

    return at, bt,None,None,None



def loadparameters(LARGE):
    if LARGE:
        Nxy = 60
        Dt = 1.0e-3   #Dt = h/(r*alpha)
        fpeak = 2.0
        t0, t1, t2, tf = 0.0, 0.1, 2.0, 2.1
    else:
        Nxy = 20
        Dt = 2.5e-3
        fpeak = 0.5
        t0, t1, t2, tf = 0.0, 0.5, 6.0, 6.5
    return Nxy, Dt, fpeak, t0, t1, t2, tf
