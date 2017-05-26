"""
Acoustic wave inverse problem with a single low-frequency,
and absorbing boundary conditions on left, bottom, and right.
Check gradient and Hessian for joint inverse problem a and b
"""

import sys
from os.path import splitext, isdir
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt

import dolfin as dl
from dolfin import MPI

from fenicstools.plotfenics import PlotFenics, plotobservations
from fenicstools.acousticwave import AcousticWave
from fenicstools.sourceterms import PointSources, RickerWavelet
from fenicstools.observationoperator import TimeObsPtwise
from fenicstools.miscfenics import checkdt, setfct
from fenicstools.objectiveacoustic import ObjectiveAcoustic
from fenicstools.optimsolver import checkgradfd_med, checkhessabfd_med

from fenicstools.prior import LaplacianPrior
from fenicstools.jointregularization import SingleRegularization

#from fenicstools.examples.acousticwave.mediumparameters import \
from fenicstools.examples.acousticwave.mediumparameters0 import \
targetmediumparameters, initmediumparameters, loadparameters


LARGE = False
PARAM = 'a'
NOISE = True
PLOTTS = False

FDGRAD = True
ALL = False
nbtest = 3



Nxy, Dt, fpeak, t0, t1, t2, tf = loadparameters(LARGE)

# Define PDE:
h = 1./Nxy
# dist is in [km]
X, Y = 1, 1
#mesh = dl.RectangleMesh(dl.Point(0.0,0.0),dl.Point(X,Y),X*Nxy,Y*Nxy)
mesh = dl.UnitSquareMesh(Nxy, Nxy)
mpicomm = mesh.mpi_comm()
mpirank = MPI.rank(mpicomm)
Vl = dl.FunctionSpace(mesh, 'Lagrange', 1)
# Source term:
Ricker = RickerWavelet(fpeak, 1e-6)
r = 2   # polynomial degree for state and adj
V = dl.FunctionSpace(mesh, 'Lagrange', r)
Pt = PointSources(V, [[0.2*X,Y],[0.5*X,Y], [0.8*X,Y]])
srcv = dl.Function(V).vector()
# Boundary conditions:
class ABCdom(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[1] < Y)
# Computation:
if mpirank == 0: print '\n\th = {}, Dt = {}'.format(h, Dt)
Wave = AcousticWave({'V':V, 'Vm':Vl}, 
{'print':False, 'lumpM':True, 'timestepper':'backward'})
Wave.set_abc(mesh, ABCdom(), lumpD=False)
#
at, bt,_,_,_ = targetmediumparameters(Vl, X)
#
Wave.update({'b':bt, 'a':at, 't0':0.0, 'tf':tf, 'Dt':Dt,\
'u0init':dl.Function(V), 'utinit':dl.Function(V)})

# observation operator:
obspts = [[ii*float(X)/float(Nxy), 0.9*Y] for ii in range(Nxy+1)]
tfilterpts = [t0, t1, t2, tf]
obsop = TimeObsPtwise({'V':V, 'Points':obspts}, tfilterpts)

# define objective function:
if FDGRAD:
    waveobj = ObjectiveAcoustic(Wave, [Ricker, Pt, srcv], PARAM)
else:
    reg = LaplacianPrior({'Vm':Vl, 'gamma':1e-4, 'beta':1e-6})
    regul = SingleRegularization(reg, PARAM, (not mpirank))

    waveobj = ObjectiveAcoustic(Wave, [Ricker, Pt, srcv], PARAM, regul)
waveobj.obsop = obsop
#waveobj.GN = True

# Generate synthetic observations
if mpirank == 0:    print 'generate noisy data'
waveobj.solvefwd()
DD = waveobj.Bp[:]
if NOISE:
    SNRdB = 15.0   # [dB], i.e, log10(mu/sigma) = SNRdB/10
    np.random.seed(11)
    for ii, dd in enumerate(DD):
        nbobspt, dimsol = dd.shape
        mu = np.abs(dd).mean(axis=1)
        sigmas = mu/(10**(SNRdB/10.))
        rndnoise = np.random.randn(nbobspt*dimsol).reshape((nbobspt, dimsol))
        print 'mpirank={}, sigmas={}, |rndnoise|={}'.format(\
        mpirank, sigmas.sum()/len(sigmas), (rndnoise**2).sum().sum())
        DD[ii] = dd + sigmas.reshape((nbobspt,1))*rndnoise
        MPI.barrier(mpicomm)
        if mpirank == 0:    print ''
waveobj.dd = DD
if PLOTTS:
    if mpirank == 0:
        fig = plotobservations(waveobj.PDE.times, waveobj.Bp[1], waveobj.dd[1], 9)
        plt.show()
    MPI.barrier(mpicomm)
# check:
waveobj.solvefwd_cost()
costmisfit = waveobj.cost_misfit
if mpirank == 0:    print 'misfit at target = {}'.format(costmisfit)
#assert costmisfit < 1e-14, costmisfit

# Compute gradient at initial parameters
a0, b0,_,_,_ = initmediumparameters(Vl, X)
waveobj.update_PDE({'a':a0, 'b':b0})
waveobj.solvefwd_cost()
if mpirank == 0:    print 'misfit at initial state = {}'.format(waveobj.cost_misfit)
if PLOTTS:
    if mpirank == 0:
        fig = plotobservations(waveobj.PDE.times, waveobj.Bp[1], waveobj.dd[1], 9)
        plt.show()
    MPI.barrier(mpicomm)
    sys.exit(0)

if FDGRAD:
    if ALL and (PARAM == 'a' or PARAM == 'b') and mpirank == 0:
        print '*** Warning: Single inversion but changing both parameters'
    # Medium perturbations
    MPa = [
    dl.Expression('1.0'), 
    dl. Expression('sin(pi*x[0])*sin(pi*x[1])'),
    dl.Expression('x[0]'), dl.Expression('x[1]'), 
    dl.Expression('sin(3*pi*x[0])*sin(3*pi*x[1])')]
    MPb = [
    dl.Expression('1.0'), 
    dl. Expression('sin(pi*x[0])*sin(pi*x[1])'),
    dl.Expression('x[1]'), dl.Expression('x[0]'), 
    dl.Expression('sin(3*pi*x[0])*sin(3*pi*x[1])')]

    if ALL:
        Medium = []
        tmp = dl.Function(Vl*Vl)
        for ii in range(nbtest):
            tmp.vector().zero()
            dl.assign(tmp.sub(0), dl.interpolate(MPa[ii], Vl))
            dl.assign(tmp.sub(1), dl.interpolate(MPb[ii], Vl))
            Medium.append(tmp.vector().copy())
        if mpirank == 0:    print 'check gradient with FD'
        checkgradfd_med(waveobj, Medium, 1e-6, [1e-5, 1e-6, 1e-7], True)
        if mpirank == 0:    print '\ncheck Hessian with FD'
        checkhessabfd_med(waveobj, Medium, 1e-6, [1e-5, 1e-6, 1e-7], True, 'all')
    else:
        Mediuma, Mediumb = [], []
        tmp = dl.Function(Vl*Vl)
        for ii in range(nbtest):
            tmp.vector().zero()
            dl.assign(tmp.sub(0), dl.interpolate(MPa[ii], Vl))
            Mediuma.append(tmp.vector().copy())
            tmp.vector().zero()
            dl.assign(tmp.sub(1), dl.interpolate(MPb[ii], Vl))
            Mediumb.append(tmp.vector().copy())
        if mpirank == 0:    print 'check a-gradient with FD'
        if 'a' in PARAM:
            checkgradfd_med(waveobj, Mediuma, 1e-6, [1e-5, 1e-6, 1e-7], True)
        else:
            checkgradfd_med(waveobj, Mediuma[:1], 1e-6, [1e-5], True)
        if mpirank == 0:    print 'check b-gradient with FD'
        if 'b' in PARAM:
            checkgradfd_med(waveobj, Mediumb, 1e-6, [1e-5, 1e-6, 1e-7], True)
        else:
            checkgradfd_med(waveobj, Mediumb[:1], 1e-6, [1e-5], True)

        if mpirank == 0:    
            print '\n'
            print 'check a-Hessian with FD'
        checkhessabfd_med(waveobj, Mediuma, 1e-6, [1e-5, 1e-6, 1e-7], True, 'a')
        if mpirank == 0:    print 'check b-Hessian with FD'
        checkhessabfd_med(waveobj, Mediumb, 1e-6, [1e-5, 1e-6, 1e-7], True, 'b')
else:
    m0 = dl.Function(Vl*Vl)
    dl.assign(m0.sub(0), a0)
    dl.assign(m0.sub(1), b0)

    mt = dl.Function(Vl*Vl)
    dl.assign(mt.sub(0), at)
    dl.assign(mt.sub(1), bt)

    waveobj.inversion(m0, mt, boundsLS=[[0.005, 5.0], [0.02, 5.0]])