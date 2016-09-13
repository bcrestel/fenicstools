"""
Routine to profile forward solve of acoustic wave problem

09/12/16 -- Profiled method iteration_backward in class AcousticWave on laptop
N = 50
Dt = 5e-4
t0tf = [0.0, 0.1, 2.00, 2.1]
freq = 2.0
    lump = True --> 2.5537 s
    lump = False --> 35.3994 s

09/12/16 -- Profiled waveobj.solvefwd() in parallel on ccgo1 (single run)
N = 50
Dt = 5e-4
t0tf = [0.0, 0.1, 2.00, 2.1]
freq = 2.0
    n = 1 --> 3.6 s
    n = 2 --> 3.3 s
    n = 4 --> 4.1 s

N = 100
Dt = 5e-4
t0tf = [0.0, 0.02, 2.00, 2.02]
freq = 10.0
    n = 1 --> 9.2 s
    n = 2 --> 6.3 s
    n = 4 --> 4.8 s
    n = 12 --> 4.6 s
    n = 16 --> 4.6 s
    n = 32 --> 5.3 s
"""

import sys
from os.path import splitext, isdir
import dolfin as dl
from fenicstools.acousticwave import AcousticWave
from fenicstools.objectiveacoustic import ObjectiveAcoustic
from fenicstools.plotfenics import PlotFenics
from fenicstools.sourceterms import PointSources, RickerWavelet
from fenicstools.observationoperator import TimeObsPtwise

dl.set_log_active(False)

from dolfin import MPI, mpi_comm_world
mpicomm = mpi_comm_world()
mpirank = MPI.rank(mpicomm)
mpisize = MPI.size(mpicomm)

from mpi4py.MPI import Wtime

#N = 50
#Dt = 5e-4
#t0tf = [0.0, 0.1, 2.00, 2.1]
#freq = 2.0
#skip = 20

N = 100
Dt = 5e-4
t0tf = [0.0, 0.02, 2.00, 2.02]
freq = 10.0
skip = 20
PLOT = False

# mesh
if mpirank == 0:    print 'Define mesh'
mesh = dl.UnitSquareMesh(N, N)
Vm = dl.FunctionSpace(mesh, 'Lagrange', 1)
V = dl.FunctionSpace(mesh, 'Lagrange', 2)

# target medium:
if mpirank == 0:    print 'Define medium properties'
b_target = dl.Expression(\
'1.0 + 1.0*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
b_target_fn = dl.interpolate(b_target, Vm)
a_target = dl.Expression('1.0')
a_target_fn = dl.interpolate(a_target, Vm)

# Define PDE
if mpirank == 0:    print 'Define PDE solver'
wavepde = AcousticWave({'V':V, 'Vm':Vm})
wavepde.timestepper = 'backward'
wavepde.lump = True
wavepde.update({'a':a_target_fn, 'b':b_target_fn, \
't0':t0tf[0], 'tf':t0tf[-1], 'Dt':Dt, 'u0init':dl.Function(V), 'utinit':dl.Function(V)})

# set up plots
if PLOT:
    if mpirank == 0:    print 'Set up plots'
    filename, ext = splitext(sys.argv[0])
    myplot = PlotFenics(filename + str(freq))
    MPI.barrier(mpicomm)
    myplot.set_varname('b_target')
    myplot.plot_vtk(b_target_fn)

# Define source
if mpirank == 0:    print 'Define source term'
Ricker = RickerWavelet(freq, 1e-10)
Pt = PointSources(V, [[0.5, 1.0]])
src = dl.Function(V)
srcv = src.vector()
mysrc = [Ricker, Pt, srcv]

# observations:
if mpirank == 0:    print 'Set up observation points'
obspts = [[0.0, ii/10.] for ii in range(1,10)] + \
[[1.0, ii/10.] for ii in range(1,10)] + \
[[ii/10., 0.0] for ii in range(1,10)] #+ [[ii/10., 1.0] for ii in range(1,10)]
obsop = TimeObsPtwise({'V':V, 'Points':obspts}, t0tf)

# define objective function:
if mpirank == 0:    print 'Fwd solve'
waveobj = ObjectiveAcoustic(wavepde, mysrc, 'b')
waveobj.obsop = obsop
MPI.barrier(mpicomm)
t0 = Wtime()
waveobj.solvefwd()
t1 = Wtime()
mindt = MPI.min(mpicomm, t1-t0)
maxdt = MPI.max(mpicomm, t1-t0)
avgdt = MPI.sum(mpicomm, t1-t0) / float(mpisize)
if mpirank == 0:    print 'mindt={}, maxdt={}, avgdt={}'.format(mindt, maxdt, avgdt)
if PLOT:    
    if mpirank == 0:    print 'Plot solution'
    myplot.plot_timeseries(waveobj.solfwd[0], 'pd', 0, skip, dl.Function(V))
