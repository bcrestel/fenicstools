"""
Compute MAP point for joint inverse problem with parameters a and b
Test different regularizations
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

from fenicstools.plotfenics import PlotFenics, plotobservations
from fenicstools.objectiveacoustic import ObjectiveAcoustic
from fenicstools.jointregularization import Tikhonovab
from fenicstools.miscfenics import checkdt
from fenicstools.acousticwave import AcousticWave
from fenicstools.sourceterms import PointSources, RickerWavelet
from fenicstools.observationoperator import TimeObsPtwise

mpicomm = mpi_comm_world()
mpirank = MPI.rank(mpicomm)
mpisize = MPI.size(mpicomm)

# Data = freq, Nxy, Dt, t0tf
Data={'0.5': [0.5, 20, 2.5e-3, [0.0, 0.5, 6.0, 6.5]],\
'1.0': [1.0, 50, 1e-3, [0.0, 0.5, 2.5, 3.0]], \
#'1.0': [1.0, 40, 1e-3, [0.0, 0.2, 3.0, 3.2]], \
'2.0': [2.0, 60, 1e-3, [0.0, 0.1, 2.0, 2.1]], \
'4.0': [4.0, 120, 5e-4, [0.0, 0.05, 1.6, 1.65]]
}


# Input data:
PLOT = False
freq, Nxy, Dt, t0tf = Data['1.0']
t0, t1, t2, tf = t0tf
skip = int(0.1/Dt)
checkdt(Dt, 1./Nxy, 2, np.sqrt(2.0), True)

# mesh
if mpirank == 0:    print 'Meshing'
mesh = dl.UnitSquareMesh(Nxy, Nxy)
Vm = dl.FunctionSpace(mesh, 'Lagrange', 1)
V = dl.FunctionSpace(mesh, 'Lagrange', 2)

# initial medium:
b_initial = dl.Expression('1.0 + 0.25*sin(pi*x[0])*sin(pi*x[1])')
b_initial_fn = dl.interpolate(b_initial, Vm)
a_initial = dl.Expression('1.0 + 0.1*sin(pi*x[0])*sin(pi*x[1])')
a_initial_fn = dl.interpolate(a_initial, Vm)
# target medium:
b_target = dl.Expression(\
'1.0 + 1.0*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
b_target_fn = dl.interpolate(b_target, Vm)
a_target = dl.Expression(\
'1.0 + 0.4*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
a_target_fn = dl.interpolate(a_target, Vm)

# define pde operator:
if mpirank == 0:    print 'Define wave PDE'
wavepde = AcousticWave({'V':V, 'Vm':Vm})
wavepde.timestepper = 'backward'
wavepde.lump = True
wavepde.update({'a':a_target_fn, 'b':b_target_fn, \
't0':t0, 'tf':tf, 'Dt':Dt, 'u0init':dl.Function(V), 'utinit':dl.Function(V)})

# set up plots:
if PLOT:
    filename, ext = splitext(sys.argv[0])
    if mpirank == 0 and isdir(filename + '/'):   
        rmtree(filename + '/')
    MPI.barrier(mpicomm)
    myplot = PlotFenics(filename)
    MPI.barrier(mpicomm)
    myplot.set_varname('b_target')
    myplot.plot_vtk(b_target_fn)
    myplot.set_varname('a_target')
    myplot.plot_vtk(a_target_fn)
else:   myplot = None

# observations along the top, left, bottom and right sides
obspts = [[0.0, ii/10.] for ii in range(1,10)] + \
[[1.0, ii/10.] for ii in range(1,10)] + \
[[ii/10., 0.0] for ii in range(1,10)] + \
[[ii/10., 1.0] for ii in range(1,10)]
obsop = TimeObsPtwise({'V':V, 'Points':obspts}, t0tf)

# sources along top edge
if mpirank == 0:    print 'Define sources'
#srcloc = [[ii/10., 1.0] for ii in range(1,10,2)]
srcloc = [[0.5, 1.0]]
Ricker = RickerWavelet(freq, 1e-10)
Pt = PointSources(V, srcloc)
src = dl.Function(V)
srcv = src.vector()
mysrc = [Ricker, Pt, srcv]

# define joint regularization
# Tikhonov + cross-gradient
#regul = Tikhonovab({'Vm':Vm,'gamma':5e-4,'beta':1e-4, 'm0':[1.0,1.0], 'cg':1.0})
# Tikhonov w/o similarity term
regul = Tikhonovab({'Vm':Vm,'gamma':5e-4,'beta':1e-8, 'm0':[1.0,1.0]})
#regul = Tikhonovab({'Vm':Vm,'gamma':5e-4,'beta':1e-4, 'm0':[1.0,1.0]})

# define objective function:
if mpirank == 0:    print 'Define objective function'
waveobj = ObjectiveAcoustic(wavepde, mysrc, 'ab', regul)
waveobj.obsop = obsop

# noisy data
if mpirank == 0:    print 'Generate noisy data'
waveobj.solvefwd()
DD = waveobj.Bp[:]
noiselevel = 0.1   # = 10%
np.random.seed(11)
for ii, dd in enumerate(DD):
    nbobspt, dimsol = dd.shape
    sigmas = np.sqrt((dd**2).sum(axis=1)/dimsol)*noiselevel
    rndnoise = np.random.randn(nbobspt*dimsol).reshape((nbobspt, dimsol))
    DD[ii] = dd + sigmas.reshape((len(sigmas),1))*rndnoise
waveobj.dd = DD
waveobj.solvefwd_cost()
if mpirank == 0:
    print 'noise misfit={}, regul cost={}, ratio={}'.format(waveobj.cost_misfit, \
    waveobj.cost_reg, waveobj.cost_misfit/waveobj.cost_reg)
if PLOT:
    for ii, solfwd in enumerate(waveobj.solfwd):
        myplot.plot_timeseries(solfwd, 'pd'+str(ii), 0, skip, dl.Function(V))
    if mpirank == 0:
        ii = 0
        for Bp, dd in zip(waveobj.Bp, waveobj.dd):
            fig = plotobservations(waveobj.PDE.times, Bp, dd)
            fig.savefig(filename + '/observations0' + '-' + str(ii) + '.eps')
            ii += 1

# Solve inverse problem
ab_target_fn = dl.Function(Vm*Vm)
dl.assign(ab_target_fn.sub(0), a_target_fn)
dl.assign(ab_target_fn.sub(1), b_target_fn)
ab_init_fn = dl.Function(Vm*Vm)
dl.assign(ab_init_fn.sub(0), a_initial_fn)
dl.assign(ab_init_fn.sub(1), b_initial_fn)
if mpirank == 0:    print 'Solve inverse problem'
waveobj.inversion(ab_init_fn, ab_target_fn, mpicomm, myplot=myplot)
if PLOT:
    for ii, solfwd in enumerate(waveobj.solfwd):
        myplot.plot_timeseries(solfwd, 'pMAP'+str(ii), 0, skip, dl.Function(V))
    if mpirank == 0:
        ii = 0
        for Bp, dd in zip(waveobj.Bp, waveobj.dd):
            fig = plotobservations(waveobj.PDE.times, Bp, dd)
            fig.savefig(filename + '/observationsMAP' + '-' + str(ii) + '.eps')
            ii += 1
