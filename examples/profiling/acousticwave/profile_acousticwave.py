"""
Routine to profile forward solve of acoustic wave problem

    -- 09/12/16 -- 
Profiled method iteration_backward in class AcousticWave on laptop
N = 50
Dt = 5e-4
t0tf = [0.0, 0.1, 2.00, 2.1]
freq = 2.0
    lump = True --> 2.5537 s
    lump = False --> 35.3994 s

    -- 09/12/16 -- 
Profiled waveobj.solvefwd() in parallel on ccgo1 (single run)
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

    -- 09/12/16 --
Profiled wavepde.solve() in parallel on ccgo1 with more granularity (single run)
dt = solve method
dt1 = 'compute u1'
dt2 = iterate(tt)
dt3 = time-loop minus iterate(tt)
N = 100
Dt = 5e-4
t0tf = [0.0, 0.02, 1.50, 1.52]
    n = 1
mindt=4.251621679, maxdt=4.251621679, avgdt=4.251621679
mindt1=0.001560618, maxdt1=0.001560618, avgdt1=0.001560618
mindt2=3.981263403, maxdt2=3.981263403, avgdt2=3.981263403
mindt3=0.257888035, maxdt3=0.257888035, avgdt3=0.257888035
    n = 4
mindt=2.365522464, maxdt=2.365647324, avgdt=2.36555717825
mindt1=0.001340694, maxdt1=0.001344395, avgdt1=0.00134227025
mindt2=2.252448366, maxdt2=2.26461217, avgdt2=2.25984499575
mindt3=0.088596042, maxdt3=0.099023882, avgdt3=0.09270732125
    n = 16
mindt=2.04740897, maxdt=2.047701066, avgdt=2.04749953175
mindt1=0.001090574, maxdt1=0.001100912, avgdt1=0.0010950368125
mindt2=1.979806566, maxdt2=1.989052526, avgdt2=1.98695616944
mindt3=0.04777686, maxdt3=0.055065266, avgdt3=0.0493254841875


Profiled wavepde.iteration_backward() in parallel on ccgo1 with more granularity (single run)
dt1 = first 3 axpy
dt2 = solverM
dt3 = the rest
N = 100
Dt = 5e-4
t0tf = [0.0, 0.02, 1.50, 1.52]
    n = 1
dt1=3.091638474, dt2=0.000161623, dt3=0.000122614
mindt=4.262460013, maxdt=4.262460013, avgdt=4.262460013
    n = 4
dt1=1.47597903825, dt2=0.000103967, dt3=7.021625e-05
mindt=2.14002041, maxdt=2.140113286, avgdt=2.14004462025
    n = 16
dt1=1.428137779, dt2=0.0001435494375, dt3=0.000107873375
mindt=2.303483536, maxdt=2.303629954, avgdt=2.30353878194
"""

import sys
from os.path import splitext, isdir
import numpy as np
import dolfin as dl
from fenicstools.acousticwave import AcousticWave
from fenicstools.plotfenics import PlotFenics
from fenicstools.sourceterms import PointSources, RickerWavelet
from fenicstools.miscfenics import checkdt

dl.set_log_active(False)

from dolfin import MPI, mpi_comm_world
mpicomm = mpi_comm_world()
mpirank = MPI.rank(mpicomm)
mpisize = MPI.size(mpicomm)


# Input data:
NNxy = [100, 100, 20]
DT = [5e-4, 5e-4, 1e-3]
T0TF = [[0.0, 0.02, 1.50, 1.52], [0.0, 0.04, 2.00, 2.04], [0.0, 0.5, 6.5, 7.0]]
FREQ = [10.0, 4.0, 0.5]
SKIP = [20, 20, 10]
# Problem size
pbindex = 0 
N, Dt, t0tf, freq, skip = NNxy[pbindex], DT[pbindex], T0TF[pbindex], FREQ[pbindex], SKIP[pbindex]
PLOT = False
checkdt(Dt, 1./N, 2, np.sqrt(2.0), True)

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
def srcterm(tt):
    srcv.zero()
    srcv.axpy(Ricker(tt), Pt[0])
    return srcv

# Solve PDE
if mpirank == 0:    print 'Fwd solve'
wavepde.set_fwd()
wavepde.ftime = srcterm
tt = dl.Timer()
MPI.barrier(mpicomm)
tt.start()
#solfwd,_,DT = wavepde.solve()
solfwd,_ = wavepde.solve()
dt = tt.stop()
mindt, maxdt, avgdt = MPI.min(mpicomm, dt), MPI.max(mpicomm, dt), MPI.sum(mpicomm, dt) / float(mpisize)
#mindt1, maxdt1, avgdt1 = MPI.min(mpicomm, DT[0]), MPI.max(mpicomm, DT[0]), MPI.sum(mpicomm, DT[0]) / float(mpisize)
#mindt2, maxdt2, avgdt2 = MPI.min(mpicomm, DT[1]), MPI.max(mpicomm, DT[1]), MPI.sum(mpicomm, DT[1]) / float(mpisize)
#mindt3, maxdt3, avgdt3 = MPI.min(mpicomm, DT[2]), MPI.max(mpicomm, DT[2]), MPI.sum(mpicomm, DT[2]) / float(mpisize)
if mpirank == 0:    
    print 'mindt={}, maxdt={}, avgdt={}'.format(mindt, maxdt, avgdt)
#    print 'mindt1={}, maxdt1={}, avgdt1={}'.format(mindt1, maxdt1, avgdt1)
#    print 'mindt2={}, maxdt2={}, avgdt2={}'.format(mindt2, maxdt2, avgdt2)
#    print 'mindt3={}, maxdt3={}, avgdt3={}'.format(mindt3, maxdt3, avgdt3)
if PLOT:    
    if mpirank == 0:    print 'Plot solution'
    myplot.plot_timeseries(solfwd, 'pd', 0, skip, dl.Function(V))
