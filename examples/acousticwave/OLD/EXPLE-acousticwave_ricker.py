"""
Ricker wavelet at the center of a unit square with dashpot absorbing boundary
conditions on all 4 boundaries
"""

import sys
from os.path import splitext, isdir
from shutil import rmtree

from fenicstools.plotfenics import PlotFenics
from fenicstools.acousticwave import AcousticWave
from fenicstools.sourceterms import PointSources, RickerWavelet
try:
    from dolfin import UnitSquareMesh, FunctionSpace, Constant, DirichletBC, \
    interpolate, Expression, Function, SubDomain, MPI, mpi_comm_world
    mycomm = mpi_comm_world()
    myrank = MPI.rank(mycomm)
except:
    from dolfin import UnitSquareMesh, FunctionSpace, Constant, DirichletBC, \
    interpolate, Expression, Function, SubDomain
    mycomm = None
    myrank = 0

RR = [1, 2, 3, 4]
ERROR = []

# Inputs:
Nxy = 100
mesh = UnitSquareMesh(Nxy, Nxy, "crossed")
h = 1./Nxy
Vl = FunctionSpace(mesh, 'Lagrange', 1)
Dt = 3e-4   #Dt = h/(r*alpha)
tf = 1.5

# Source term:
fpeak = 4. # .4Hz => up to 10Hz in input signal
Ricker = RickerWavelet(fpeak, 1e-10)

# Boundary conditions:
class AllFour(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

for r in RR:
    V = FunctionSpace(mesh, 'Lagrange', r)
    Pt = PointSources(V, [[.5,.5]])
    mydelta = Pt[0].array()
    def mysrc(tt):
        return Ricker(tt)*mydelta
    # Computation:
    if myrank == 0: print '\n\th = {}, Dt = {}'.format(h, Dt)
    Wave = AcousticWave({'V':V, 'Vm':Vl})
    #Wave.verbose = True
    Wave.timestepper = 'centered'
    Wave.lump = True
    Wave.set_abc(mesh, AllFour(), True)
    Wave.exact = Function(V)
    Wave.update({'b':1.0, 'a':1.0, 't0':0.0, 'tf':tf, 'Dt':Dt,\
    'u0init':Function(V), 'utinit':Function(V)})
    Wave.ftime = mysrc
    sol, error = Wave.solve()
    ERROR.append(error)
    if myrank == 0: print 'relative error = {:.5e}'.format(error)
    if not mycomm == None:  MPI.barrier(mycomm)

# Plots:
try:
    boolplot = int(sys.argv[1])
except:
    boolplot = 0
if boolplot > 0:
    filename, ext = splitext(sys.argv[0])
    if myrank == 0: 
        if isdir(filename + '/'):   rmtree(filename + '/')
    if not mycomm == None:  MPI.barrier(mycomm)
    myplot = PlotFenics(filename)
    myplot.set_varname('p')
    plotp = Function(V)
    for index, pp in enumerate(sol):
        if index%boolplot == 0:
            plotp.vector()[:] = pp[0]
            myplot.plot_vtk(plotp, index)
    myplot.gather_vtkplots()
