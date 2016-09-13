"""
Compute MAP point then assemble the data-misfit part of the Hessian
"""
import sys
from os.path import splitext, isdir
from shutil import rmtree
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import dolfin as dl
from dolfin import MPI, mpi_comm_world
dl.set_log_active(False)

from petsc4py import PETSc

from fenicstools.plotfenics import PlotFenics
from fenicstools.objectiveacoustic import ObjectiveAcoustic
from fenicstools.prior import LaplacianPrior
from fenicstools.miscfenics import checkdt, setfct
from fenicstools.acousticwave import AcousticWave
from fenicstools.sourceterms import PointSources, RickerWavelet
from fenicstools.observationoperator import TimeObsPtwise
from fenicstools.linalg.miscroutines import setglobalvalue, setupPETScmatrix

from mpi4py.MPI import Wtime

outputdirectory = '/workspace/ben/fenics/assemble-Hessian/'

mpicomm = mpi_comm_world()
mpirank = MPI.rank(mpicomm)
mpisize = MPI.size(mpicomm)

# Input data:
Nxy = 100
Dt = 5e-4
T0TF = [[0.0, 0.02, 1.50, 1.52], [0.0, 0.04, 2.00, 2.02], [0.0, 0.5, 6.5, 7.0]]
FREQ = [10.0, 4.0, 0.5]
SKIP = [20, 20, 200]
checkdt(Dt, 1./Nxy, 2, np.sqrt(2.0), True)

# mesh
if mpirank == 0:    print 'meshing'
mesh = dl.UnitSquareMesh(Nxy, Nxy)
Vm = dl.FunctionSpace(mesh, 'Lagrange', 1)
V = dl.FunctionSpace(mesh, 'Lagrange', 2)

# target medium:
b_target = dl.Expression(\
'1.0 + 1.0*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
b_target_fn = dl.interpolate(b_target, Vm)
a_target = dl.Expression('1.0')
a_target_fn = dl.interpolate(a_target, Vm)

# observations:
obspts = [[0.0, ii/10.] for ii in range(1,10)] + \
[[1.0, ii/10.] for ii in range(1,10)] + \
[[ii/10., 0.0] for ii in range(1,10)] #+ [[ii/10., 1.0] for ii in range(1,10)]

# define pde operator:
if mpirank == 0:    print 'define wave pde'
wavepde = AcousticWave({'V':V, 'Vm':Vm})
wavepde.timestepper = 'backward'
wavepde.lump = True

regul = LaplacianPrior({'Vm':Vm,'gamma':5e-4,'beta':5e-4, 'm0':a_target_fn})


t0tf = T0TF[0]
freq = FREQ[0]
skip = SKIP[0]


# source:
if mpirank == 0:    print 'sources'
srcloc = [[ii/10., 1.0] for ii in range(1,10,2)]
#srcloc = [[0.5, 1.0]]
Ricker = RickerWavelet(freq, 1e-10)
Pt = PointSources(V, srcloc)
src = dl.Function(V)
srcv = src.vector()
mysrc = [Ricker, Pt, srcv]

if mpirank == 0:    print 'Define wave PDE'
obsop = TimeObsPtwise({'V':V, 'Points':obspts}, t0tf)
t0, t1, t2, tf = t0tf
wavepde.update({'a':a_target_fn, 'b':b_target_fn, \
't0':t0, 'tf':tf, 'Dt':Dt, 'u0init':dl.Function(V), 'utinit':dl.Function(V)})

# parameters
Vm = wavepde.Vm
V = wavepde.V
lenobspts = obsop.PtwiseObs.nbPts

# set up plots:
#filename, ext = splitext(sys.argv[0])
#myplot = PlotFenics(outputdirectory + filename + str(freq))
#MPI.barrier(mpicomm)
#myplot.set_varname('b_target')
#myplot.plot_vtk(b_target_fn)

if mpirank == 0:    print 'Define objective function'
# define objective function:
waveobj = ObjectiveAcoustic(wavepde, mysrc, 'b', regul)
waveobj.obsop = obsop

# noisy data
if mpirank == 0:    print 'generate noisy data'
waveobj.solvefwd()
DD = waveobj.Bp[:]
noiselevel = 0.1   # = 10%
for ii, dd in enumerate(DD):
    np.random.seed(11)
    nbobspt, dimsol = dd.shape
    sigmas = np.sqrt((dd**2).sum(axis=1)/dimsol)*noiselevel
    rndnoise = np.random.randn(nbobspt*dimsol).reshape((nbobspt, dimsol))
    DD[ii] = dd + sigmas.reshape((len(sigmas),1))*rndnoise
waveobj.dd = DD
waveobj.solvefwd_cost()
if mpirank == 0:
    print 'noise misfit={}, regul cost={}, ratio={}'.format(waveobj.cost_misfit, \
    waveobj.cost_reg, waveobj.cost_misfit/waveobj.cost_reg)
#myplot.plot_timeseries(waveobj.solfwd[2], 'pd', 0, skip, dl.Function(V))


# Matvec for Hessian
if mpirank == 0:    print 'Compute gradient'
waveobj.update_PDE({'b':b_target_fn})
waveobj.solvefwd_cost()
waveobj.solveadj_constructgrad()
x, y = dl.Function(Vm), dl.Function(Vm)
setfct(x, 1.0)
if mpirank == 0:    print 'Time Hessian matvec'
for ii in range(10):
    MPI.barrier(mpicomm)
    t0 = Wtime()
    waveobj.mult(x.vector(), y.vector())
    t1 = Wtime()
    dt = t1-t0
    mindt = MPI.min(mpicomm, t1-t0)
    maxdt = MPI.max(mpicomm, t1-t0)
    avgdt = MPI.sum(mpicomm, t1-t0) / float(mpisize)
    if mpirank == 0:
        print 'min={}, max={}, avg={}'.format(mindt, maxdt, avgdt)

