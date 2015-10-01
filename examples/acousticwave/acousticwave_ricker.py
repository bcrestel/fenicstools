import sys
from os.path import splitext, isdir
from shutil import rmtree

from fenicstools.plotfenics import PlotFenics
from fenicstools.acousticwave import AcousticWave
from fenicstools.sourceterms import PointSources, RickerWavelet
try:
    from dolfin import UnitSquareMesh, FunctionSpace, Function, \
    MPI, mpi_comm_world
    mycomm = mpi_comm_world()
    myrank = MPI.rank(mycomm)
except:
    from dolfin import UnitSquareMesh, FunctionSpace, Function
    mycomm = None
    myrank = 0

# Inputs:
Nxy = 100
mesh = UnitSquareMesh(Nxy, Nxy, "crossed")
h = 1./Nxy
q = 5   # Polynomial order for solution
V = FunctionSpace(mesh, 'Lagrange', q)
Vl = FunctionSpace(mesh, 'Lagrange', 1)
Dt = h/(q*10.)
tf = 0.4    # Final time
# Source term:
Pt = PointSources(V, [[.5,.5]])
mydelta = Pt[0].array()
fpeak = 4.0 # 4.0Hz => up to 10Hz in input signal
Ricker = RickerWavelet(fpeak, 1e-10)
def mysrc(tt):
    return Ricker(tt)*mydelta
# Computation:
if myrank == 0: print '\n\th = {}, Dt = {}'.format(h, Dt)
Wave = AcousticWave({'V':V, 'Vl':Vl, 'Vr':Vl})
Wave.verbose = True
#Wave.lump = True
Wave.update({'lambda':1.0, 'rho':1.0, 't0':0.0, 'tf':tf, 'Dt':Dt,\
'u0init':Function(V), 'utinit':Function(V)})
Wave.ftime = mysrc
sol, error = Wave.solve()
MPI.barrier(mycomm)
# Plots:
filename, ext = splitext(sys.argv[0])
if myrank == 0: 
    if isdir(filename + '/'):   rmtree(filename + '/')
MPI.barrier(mycomm)
myplot = PlotFenics(filename)
myplot.set_varname('p')
plotp = Function(V)
for index, pp in enumerate(sol):
    if index%10 == 0:
        plotp.vector()[:] = pp[0]
        myplot.plot_vtk(plotp, index)
myplot.gather_vtkplots()
