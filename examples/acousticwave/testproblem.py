"""
Ricker wavelet at the center of a unit square with dashpot absorbing boundary
conditions on all 4 boundaries
"""

import sys
from os.path import splitext, isdir
from shutil import rmtree

import dolfin as dl
from dolfin import MPI

from fenicstools.plotfenics import PlotFenics
from fenicstools.acousticwave import AcousticWave
from fenicstools.sourceterms import PointSources, RickerWavelet

from mediumparameters import targetmediumparameters

# Inputs:
Nxy = 100
h = 1./Nxy
# dist is in [km]
X, Y = 1, 1
mesh = dl.RectangleMesh(dl.Point(0.0,0.0),dl.Point(X,Y),X*Nxy,Y*Nxy)
mpicomm = mesh.mpi_comm()
mpirank = MPI.rank(mpicomm)
Vl = dl.FunctionSpace(mesh, 'Lagrange', 1)
Dt = 1.0e-4   #Dt = h/(r*alpha)
tf = 1.0

# Plots:
filename, ext = splitext(sys.argv[0])
if mpirank == 0: 
    if isdir(filename + '/'):   rmtree(filename + '/')
MPI.barrier(mpicomm)
myplot = PlotFenics(filename)

# Source term:
fpeak = 6.0
Ricker = RickerWavelet(fpeak, 1e-6)

# Boundary conditions:
class ABCdom(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[1] < Y)

r = 2
V = dl.FunctionSpace(mesh, 'Lagrange', r)
Pt = PointSources(V, [[0.5*X,Y]])
mydelta = Pt[0]
def mysrc(tt):
    return mydelta * Ricker(tt)
# Computation:
if mpirank == 0: print '\n\th = {}, Dt = {}'.format(h, Dt)
Wave = AcousticWave({'V':V, 'Vm':Vl}, 
{'print':(not mpirank), 'lumpM':True, 'timestepper':'backward'})
#Wave.set_abc(mesh, ABCdom(), lumpD=True)
Wave.exact = dl.Function(V)
Wave.ftime = mysrc
#
af, bf = targetmediumparameters(Vl, X, myplot)
#
Wave.update({'b':bf, 'a':af, 't0':0.0, 'tf':tf, 'Dt':Dt,\
'u0init':dl.Function(V), 'utinit':dl.Function(V)})

sol, error = Wave.solve()
if mpirank == 0: print 'relative error = {:.5e}'.format(error)
MPI.barrier(mesh.mpi_comm())

# Plots:
try:
    boolplot = int(sys.argv[1])
except:
    boolplot = 0
if boolplot > 0:
    myplot.set_varname('p')
    plotp = dl.Function(V)
    for index, pp in enumerate(sol):
        if index%boolplot == 0:
            plotp.vector()[:] = pp[0]
            myplot.plot_vtk(plotp, index)
    myplot.gather_vtkplots()
