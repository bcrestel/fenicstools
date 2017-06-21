"""
Medium parameters
"""

import dolfin as dl


def createparam(Vl, DD, Vt, Vb, Vp):
    c = dl.interpolate(dl.Expression('\
    (x[0]>=LL)*(x[0]<=RR)*(x[1]>=BB)*(x[1]<=TT)*vp\
    + (1.0-(x[0]>=LL)*(x[0]<=RR)*(x[1]>=BB)*(x[1]<=TT))*(\
    vt*(x[1]>=0.5) + vb*(x[1]<0.5))',\
    LL=0.5-0.5*DD, RR=0.5+0.5*DD, BB=0.5-0.5*DD, TT=0.5+0.5*DD,\
    vp=Vp, vt=Vt, vb=Vb, degree=10), Vl)
    return c

DD = 0.4
# medium parameters:
CC = [2.0, 3.0, 2.5]
RR = [2.1, 2.2, 2.15]
#CC = [5.0, 2.0, 3.0, 4.0]
#RR = [2.0, 2.1, 2.2, 2.5]
LL, AA, BB = [], [], []
for cc, rr in zip(CC, RR):
    ll = rr*cc*cc
    LL.append(ll)
    AA.append(1./ll)
    BB.append(1./rr)

def targetmediumparameters(Vl, X, myplot=None):
    """
    Arguments:
        Vl = function space
        X = x dimension of domain
    """
    # velocity is in [km/s]
    c = createparam(Vl, DD, CC[0], CC[1], CC[2])
    if not myplot == None:
        myplot.set_varname('c_target')
        myplot.plot_vtk(c)
    # density is in [10^12 kg/km^3]=[g/cm^3]
    # assume rocks shale-sand-shale + salt inside small rectangle
    # see Marmousi2 print-out
    rho = createparam(Vl, DD, RR[0], RR[1], RR[2])
    if not myplot == None:
        myplot.set_varname('rho_target')
        myplot.plot_vtk(rho)
    # bulk modulus is in [10^12 kg/km.s^2]=[GPa]
    lam = createparam(Vl, DD, LL[0], LL[1], LL[2])
    if not myplot == None:
        myplot.set_varname('lambda_target')
        myplot.plot_vtk(lam)
    #
    af = createparam(Vl, DD, AA[0], AA[1], AA[2])
    if not myplot == None:
        myplot.set_varname('alpha_target')
        myplot.plot_vtk(af)
    bf = createparam(Vl, DD, BB[0], BB[1], BB[2])
    if not myplot == None:
        myplot.set_varname('beta_target')
        myplot.plot_vtk(bf)
    # Check:
    ones = dl.interpolate(dl.Constant('1.0'), Vl)
    check1 = af.vector() * lam.vector()
    erra = dl.norm(check1 - ones.vector())
    assert erra < 1e-16
    check2 = bf.vector() * rho.vector()
    errb = dl.norm(check2 - ones.vector())
    assert errb < 1e-16

    return af, bf, c, lam, rho


def smoothstart(Vl, top, bott):
    return dl.interpolate(dl.Expression('\
    tp*(x[1]>=TT) + (tp + (bt-tp)*(TT-x[1])/dd)*(x[1]<TT)*(x[1]>BB) + bt*(x[1]<=BB)',\
    bt=bott, tp=top, TT=0.5+0.5*DD, BB=0.5-0.5*DD, dd=DD, degree=10), Vl)


def initmediumparameters(Vl, X, myplot=None):
    a0 = smoothstart(Vl, AA[0], AA[1])
    if not myplot == None:
        myplot.set_varname('alpha_init')
        myplot.plot_vtk(a0)
    b0 = smoothstart(Vl, BB[0], BB[1])
    if not myplot == None:
        myplot.set_varname('beta_init')
        myplot.plot_vtk(b0)

    return a0, b0, None, None, None



def loadparameters(LARGE):
    if LARGE:
        Nxy = 60
        Dt = 1.0e-3   #Dt = h/(r*alpha)
        fpeak = 4.0
        t0, t1, t2, tf = 0.0, 0.1, 0.9, 1.0
    else:
        Nxy = 20
        Dt = 1.5e-3
        fpeak = 2.0
        t0, t1, t2, tf = 0.0, 0.1, 1.7, 1.8
    return Nxy, Dt, fpeak, t0, t1, t2, tf
