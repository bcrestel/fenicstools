"""
Plot selected medium parameters
"""
import sys
from os.path import splitext, isdir
from shutil import rmtree
import dolfin as dl
from dolfin import MPI
from fenicstools.plotfenics import PlotFenics
from mediumparameters0 import \
targetmediumparameters, initmediumparameters, loadparameters

LARGE = True
Nxy, Dt, fpeak,_,_,_,tf = loadparameters(LARGE)

X, Y = 1, 1
#mesh = dl.RectangleMesh(dl.Point(0.0,0.0),dl.Point(X,Y),X*Nxy,Y*Nxy)
mesh = dl.UnitSquareMesh(Nxy, Nxy)
Vl = dl.FunctionSpace(mesh, 'Lagrange', 1)

mpicomm = mesh.mpi_comm()
mpirank = MPI.rank(mpicomm)
filename, ext = splitext(sys.argv[0])
if mpirank == 0: 
    if isdir(filename + '/'):   rmtree(filename + '/')
MPI.barrier(mpicomm)
myplot = PlotFenics(filename)

at, bt, c, lam, rho = targetmediumparameters(Vl, X, myplot)
a0, b0,_,_,_ = initmediumparameters(Vl, X, myplot)
