"""
Medium parameters
"""

import dolfin as dl


def createparam(CC, Vl, X, H1, H2, H3, TT):
    c = dl.interpolate(dl.Expression(' \
    (x[0]>=LL)*(x[0]<=RR)*(x[1]>=HC-TT)*(x[1]<=HC+TT)*vva \
    + (1.0-(x[0]>=LL)*(x[0]<=RR)*(x[1]>=HC-TT)*(x[1]<=HC+TT))*( \
    vvb*(x[1]>HA) +  \
    vvc*(x[1]<=HA)*(x[1]>HB) + \
    vvd*(x[1]<=HB))', 
    vva=CC[0], vvb=CC[1], vvc=CC[2], vvd=CC[3],
    LL=X/4.0, RR=3.0*X/4.0, HA=H1, HB=H2, HC=H3, TT=TT), Vl)
    return c


def targetmediumparameters(Vl, X, myplot=None):
    """
    Arguments:
        Vl = function space
        X = x dimension of domain
    """
    # medium parameters:
    H1, H2, H3, TT = 0.8, 0.2, 0.6, 0.1
    CC = [5.0, 2.0, 3.0, 4.0]
    RR = [2.0, 2.1, 2.2, 2.5]
    LL, AA, BB = [], [], []
    for cc, rr in zip(CC, RR):
        ll = rr*cc*cc
        LL.append(ll)
        AA.append(1./ll)
        BB.append(1./rr)
    # velocity is in [km/s]
    c = createparam(CC, Vl, X, H1, H2, H3, TT)
    if not myplot == None:
        myplot.set_varname('c_target')
        myplot.plot_vtk(c)
    # density is in [10^12 kg/km^3]=[g/cm^3]
    # assume rocks shale-sand-shale + salt inside small rectangle
    # see Marmousi2 print-out
    rho = createparam(RR, Vl, X, H1, H2, H3, TT)
    if not myplot == None:
        myplot.set_varname('rho_target')
        myplot.plot_vtk(rho)
    # bulk modulus is in [10^12 kg/km.s^2]=[GPa]
    lam = createparam(LL, Vl, X, H1, H2, H3, TT)
    if not myplot == None:
        myplot.set_varname('lambda_target')
        myplot.plot_vtk(lam)
    #
    af = createparam(AA, Vl, X, H1, H2, H3, TT)
    if not myplot == None:
        myplot.set_varname('alpha_target')
        myplot.plot_vtk(af)
    bf = createparam(BB, Vl, X, H1, H2, H3, TT)
    if not myplot == None:
        myplot.set_varname('beta_target')
        myplot.plot_vtk(bf)
    # Check:
    ones = dl.interpolate(dl.Expression('1.0'), Vl)
    check1 = af.vector() * lam.vector()
    erra = dl.norm(check1 - ones.vector())
    assert erra < 1e-16
    check2 = bf.vector() * rho.vector()
    errb = dl.norm(check2 - ones.vector())
    assert errb < 1e-16

    return af, bf



def initmediumparameters(Vl, X, myplot=None):
    # medium parameters:
    H1, H2, H3, TT = 0.8, 0.2, 0.6, 0.1
    CC = [3.0, 2.0, 3.0, 4.0]
    RR = [2.2, 2.1, 2.2, 2.5]
    LL, AA, BB = [], [], []
    for cc, rr in zip(CC, RR):
        ll = rr*cc*cc
        LL.append(ll)
        AA.append(1./ll)
        BB.append(1./rr)
    # velocity is in [km/s]
    c = createparam(CC, Vl, X, H1, H2, H3, TT)
    if not myplot == None:
        myplot.set_varname('c_init')
        myplot.plot_vtk(c)
    # density is in [10^12 kg/km^3]=[g/cm^3]
    # assume rocks shale-sand-shale + salt inside small rectangle
    # see Marmousi2 print-out
    rho = createparam(RR, Vl, X, H1, H2, H3, TT)
    if not myplot == None:
        myplot.set_varname('rho_init')
        myplot.plot_vtk(rho)
    # bulk modulus is in [10^12 kg/km.s^2]=[GPa]
    lam = createparam(LL, Vl, X, H1, H2, H3, TT)
    if not myplot == None:
        myplot.set_varname('lambda_init')
        myplot.plot_vtk(lam)
    #
    af = createparam(AA, Vl, X, H1, H2, H3, TT)
    if not myplot == None:
        myplot.set_varname('alpha_init')
        myplot.plot_vtk(af)
    bf = createparam(BB, Vl, X, H1, H2, H3, TT)
    if not myplot == None:
        myplot.set_varname('beta_init')
        myplot.plot_vtk(bf)
    # Check:
    ones = dl.interpolate(dl.Expression('1.0'), Vl)
    check1 = af.vector() * lam.vector()
    erra = dl.norm(check1 - ones.vector())
    assert erra < 1e-16
    check2 = bf.vector() * rho.vector()
    errb = dl.norm(check2 - ones.vector())
    assert errb < 1e-16

    return af, bf



def loadparameters(LARGE):
    if LARGE:
        Nxy = 100
        Dt = 1.0e-4   #Dt = h/(r*alpha)
        fpeak = 6.0
        t0, t1, t2, tf = 0.0, 0.2, 0.8, 1.0
        nbtest = 5
    else:
        Nxy = 10
        Dt = 2.0e-3
        fpeak = 1.0
        t0, t1, t2, tf = 0.0, 0.5, 2.5, 3.0
        nbtest = 2
    return Nxy, Dt, fpeak, t0, t1, t2, tf
