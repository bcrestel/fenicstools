"""
Plot selected medium parameters
"""
import sys
from os.path import splitext, isdir
from shutil import rmtree
import dolfin as dl
from dolfin import MPI
from fenicstools.plotfenics import PlotFenics
from mediumparameters import targetmediumparameters, initmediumparameters

LARGE = True

if LARGE:
    Nxy = 100
    Dt = 1.0e-4   #Dt = h/(r*alpha)
    tf = 1.0
    fpeak = 6.0
else:
    Nxy = 10
    Dt = 2.0e-3
    tf = 3.0
    fpeak = 1.0

X, Y = 1, 1
mesh = dl.RectangleMesh(dl.Point(0.0,0.0),dl.Point(X,Y),X*Nxy,Y*Nxy)
Vl = dl.FunctionSpace(mesh, 'Lagrange', 1)

mpicomm = mesh.mpi_comm()
mpirank = MPI.rank(mpicomm)
filename, ext = splitext(sys.argv[0])
if mpirank == 0: 
    if isdir(filename + '/'):   rmtree(filename + '/')
MPI.barrier(mpicomm)
myplot = PlotFenics(filename)

af, bf = targetmediumparameters(Vl, X, myplot)
a0, b0 = initmediumparameters(Vl, X, myplot)
