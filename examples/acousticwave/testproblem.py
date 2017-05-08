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


# Inputs:
Nxy = 100
mesh = dl.UnitSquareMesh(Nxy, Nxy, "crossed")
myrank = MPI.rank(mesh.mpi_comm())
h = 1./Nxy
Vl = dl.FunctionSpace(mesh, 'Lagrange', 1)
Dt = 1e-4   #Dt = h/(r*alpha)
tf = 1.5

# Source term:
fpeak = 4.0 # .4Hz => up to 10Hz in input signal
Ricker = RickerWavelet(fpeak, 1e-10)

# Boundary conditions:
class AllFour(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

r = 2
V = dl.FunctionSpace(mesh, 'Lagrange', r)
Pt = PointSources(V, [[.5,.5]])
mydelta = Pt[0]
def mysrc(tt):
    return mydelta * Ricker(tt)
# Computation:
if myrank == 0: print '\n\th = {}, Dt = {}'.format(h, Dt)
Wave = AcousticWave({'V':V, 'Vm':Vl})
#Wave.verbose = True
Wave.timestepper = 'centered'
Wave.lump = True
Wave.set_abc(mesh, AllFour(), True)
Wave.exact = dl.Function(V)
Wave.update({'b':1.0, 'a':1.0, 't0':0.0, 'tf':tf, 'Dt':Dt,\
'u0init':dl.Function(V), 'utinit':dl.Function(V)})
Wave.ftime = mysrc
sol, error = Wave.solve()
if myrank == 0: print 'relative error = {:.5e}'.format(error)
MPI.barrier(mycomm)

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
    plotp = dl.Function(V)
    for index, pp in enumerate(sol):
        if index%boolplot == 0:
            plotp.vector()[:] = pp[0]
            myplot.plot_vtk(plotp, index)
    myplot.gather_vtkplots()
