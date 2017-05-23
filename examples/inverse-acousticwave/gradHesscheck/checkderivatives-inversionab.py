"""
Acoustic wave inverse problem with a single low-frequency,
and homogeneous Neumann boundary conditions everywhere.
Check gradient and Hessian for joint inverse problem a and b
"""

import sys
from os.path import splitext, isdir
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt

import dolfin as dl
from dolfin import MPI

from fenicstools.plotfenics import PlotFenics
from fenicstools.acousticwave import AcousticWave
from fenicstools.sourceterms import PointSources, RickerWavelet
from fenicstools.observationoperator import TimeObsPtwise
from fenicstools.miscfenics import checkdt, setfct
from fenicstools.objectiveacoustic import ObjectiveAcoustic
from fenicstools.optimsolver import checkgradfd_med, checkhessabfd_med

from fenicstools.examples.acousticwave.mediumparameters import \
targetmediumparameters, initmediumparameters

ALL = False
LARGE = True


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

# Define PDE:
h = 1./Nxy
# dist is in [km]
X, Y = 1, 1
mesh = dl.RectangleMesh(dl.Point(0.0,0.0),dl.Point(X,Y),X*Nxy,Y*Nxy)
mpicomm = mesh.mpi_comm()
mpirank = MPI.rank(mpicomm)
Vl = dl.FunctionSpace(mesh, 'Lagrange', 1)
# Source term:
Ricker = RickerWavelet(fpeak, 1e-6)
# Boundary conditions:
class ABCdom(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[1] < Y)
r = 2   # polynomial degree for state and adj
V = dl.FunctionSpace(mesh, 'Lagrange', r)
Pt = PointSources(V, [[0.5*X,Y]])
srcv = dl.Function(V).vector()
# Computation:
if mpirank == 0: print '\n\th = {}, Dt = {}'.format(h, Dt)
Wave = AcousticWave({'V':V, 'Vm':Vl}, 
{'print':False, 'lumpM':True, 'timestepper':'backward'})
Wave.set_abc(mesh, ABCdom(), lumpD=False)
#
af, bf = targetmediumparameters(Vl, X)
#
Wave.update({'b':bf, 'a':af, 't0':0.0, 'tf':tf, 'Dt':Dt,\
'u0init':dl.Function(V), 'utinit':dl.Function(V)})

# observation operator:
obspts = [[ii*float(X)/float(Nxy), 0.9*Y] for ii in range(Nxy+1)]
tfilterpts = [t0, t1, t2, tf]
obsop = TimeObsPtwise({'V':V, 'Points':obspts}, tfilterpts)

# define objective function:
waveobj = ObjectiveAcoustic(Wave, [Ricker, Pt, srcv], 'ab')
waveobj.obsop = obsop

# Generate synthetic observations
if mpirank == 0:    print 'generate noisy data'
waveobj.solvefwd()
DD = waveobj.Bp[:]
#noiselevel = 0.1   # = 10%
#for ii, dd in enumerate(DD):
#    np.random.seed(11)
#    nbobspt, dimsol = dd.shape
#    sigmas = np.sqrt((dd**2).sum(axis=1)/dimsol)*noiselevel
#    rndnoise = np.random.randn(nbobspt*dimsol).reshape((nbobspt, dimsol))
#    DD[ii] = dd + sigmas.reshape((len(sigmas),1))*rndnoise
waveobj.dd = DD
# check:
waveobj.solvefwd_cost()
costmisfit = waveobj.cost_misfit
if mpirank == 0:    print 'misfit = {}'.format(costmisfit)
assert costmisfit < 1e-14, costmisfit

# Compute gradient at tested location
a0, b0 = initmediumparameters(Vl, X)
waveobj.update_PDE({'a':a0, 'b':b0})
waveobj.solvefwd_cost()
if mpirank == 0:    print 'misfit = {}'.format(waveobj.cost_misfit)

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
    checkgradfd_med(waveobj, Mediuma, 1e-6, [1e-5, 1e-6, 1e-7], True)
    if mpirank == 0:    print 'check b-gradient with FD'
    checkgradfd_med(waveobj, Mediumb, 1e-6, [1e-5, 1e-6, 1e-7], True)

    if mpirank == 0:    print '\ncheck a-Hessian with FD'
    checkhessabfd_med(waveobj, Mediuma, 1e-6, [1e-5, 1e-6, 1e-7, 1e-8], True, 'a')
    #checkhessabfd_med(waveobj, Mediuma, 1e-6, [1e-4, 1e-5, 1e-6, 1e-7], False, 'a')
    if mpirank == 0:    print 'check b-Hessian with FD'
    checkhessabfd_med(waveobj, Mediumb, 1e-6, [1e-5, 1e-6, 1e-7, 1e-8], True, 'b')
    #checkhessabfd_med(waveobj, Mediumb, 1e-6, [1e-3, 1e-4, 1e-5, 1e-6], False, 'b')
