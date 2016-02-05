"""
Acoustic wave inverse problem with a single low-frequency,
and homogeneous Neumann boundary conditions everywhere.
"""

import sys
from os.path import splitext, isdir
from shutil import rmtree

from fenicstools.plotfenics import PlotFenics
from fenicstools.acousticwave import AcousticWave
from fenicstools.sourceterms import PointSources, RickerWavelet
from fenicstools.observationoperator import TimeObsPtwise
from fenicstools.miscfenics import checkdt, setfct
from fenicstools.objectiveacoustic import ObjectiveAcoustic
try:
    from dolfin import UnitSquareMesh, FunctionSpace, Constant, \
    interpolate, Expression, Function, plot, interactive, \
    MPI, mpi_comm_world
    mycomm = mpi_comm_world()
    myrank = MPI.rank(mycomm)
except:
    from dolfin import UnitSquareMesh, FunctionSpace, Constant, \
    interpolate, Expression, Function, plot, interactive
    mycomm = None
    myrank = 0

# Inputs:
fpeak = 0.4  #Hz
backgroundspeed = 1.0
perturbationspeed = 2.0
Nxy = 10
h = 1./Nxy
t0, tf = 0.0, 7.0
tfilterpts = [t0, t0+1., tf-1., tf]
r = 2   # order polynomial approx
Dt = 1e-4
checkdt(Dt, h, r, perturbationspeed, True)

# 
mesh = UnitSquareMesh(Nxy, Nxy)
Vl = FunctionSpace(mesh, 'Lagrange', 1)
V = FunctionSpace(mesh, 'Lagrange', r)
fctV = Function(V)
# set up plots:
filename, ext = splitext(sys.argv[0])
if myrank == 0: 
    if isdir(filename + '/'):   rmtree(filename + '/')
if not mycomm == None:  MPI.barrier(mycomm)
myplot = PlotFenics(filename)
# source:
Ricker = RickerWavelet(fpeak, 1e-10)
Pt = PointSources(V, [[.5,1.]])
mydelta = Pt[0].array()
def mysrc(tt):
    return Ricker(tt)*mydelta
# target medium:
lambda_target = Expression(\
'xb + xmax*((x[0]>=0.3)*(x[0]<=0.7)*(x[1]>=0.3)*(x[1]<=0.7))', \
xb=backgroundspeed, xmax = perturbationspeed**2 - backgroundspeed**2)
lambda_target_fn = interpolate(lambda_target, Vl)
myplot.set_varname('lambda_target')
myplot.plot_vtk(lambda_target_fn)
# initial medium:
lambda_init = Constant(backgroundspeed)
lambda_init_fn = interpolate(lambda_init, Vl)
myplot.set_varname('lambda_init')
myplot.plot_vtk(lambda_init_fn)
# observation operator:
"""
obspts = []
for ii in range(2,9):   obspts.append([ii/10.,.8])
for ii in range(3,8):   obspts.append([.2,ii/10.], [.8,ii/10.])
for ii in range(2,9):   obspts.append([ii/10.,.2])
"""
obspts = [[.5,.2], [.5,.8], [.2,.5], [.8,.5]]
obsop = TimeObsPtwise({'V':V, 'Points':obspts}, tfilterpts)
# define pde operator:
wavepde = AcousticWave({'V':V, 'Vl':Vl, 'Vr':Vl})
wavepde.timestepper = 'backward'
wavepde.lump = True
wavepde.update({'lambda':lambda_target_fn, 'rho':1.0, \
't0':t0, 'tf':tf, 'Dt':Dt, 'u0init':Function(V), 'utinit':Function(V)})
wavepde.ftime = mysrc
# define objective function:
waveobj = ObjectiveAcoustic(wavepde)
waveobj.obsop = obsop
#
print 'generate data'
waveobj.solvefwd()
myplot.plot_timeseries(waveobj.solfwd, 'pd', 0, 100, fctV)
dd = waveobj.Bp.copy()
#
print 'assemble gradient'
waveobj.dd = dd
waveobj.update_m(lambda_init_fn)
waveobj.solvefwd_cost()
print waveobj.misfit
myplot.plot_timeseries(waveobj.solfwd, 'p', 0, 100, fctV)
# sanity check
wavepde2 = AcousticWave({'V':V, 'Vl':Vl, 'Vr':Vl})
wavepde2.timestepper = 'backward'
wavepde2.lump = True
wavepde2.update({'lambda':lambda_init_fn, 'rho':1.0, \
't0':t0, 'tf':tf, 'Dt':Dt, 'u0init':Function(V), 'utinit':Function(V)})
wavepde2.ftime = mysrc
waveobj2 = ObjectiveAcoustic(wavepde2)
waveobj2.obsop = obsop
waveobj2.solvefwd()
print ((waveobj.Bp-waveobj2.Bp)**2).sum().sum()
#
waveobj.solveadj_constructgrad()
myplot.plot_timeseries(waveobj.soladj, 'v', 0, 100, fctV)
myplot.set_varname('grad')
myplot.plot_vtk(waveobj.Grad)

