""" 1D (quadratic) wave hitting a layer + abs bndy conditions """

import sys
from os.path import splitext, isdir
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt

from fenicstools.plotfenics import PlotFenics
from fenicstools.acousticwave import AcousticWave
from fenicstools.miscfenics import checkdt_abc
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

direction = 0   # direction of the 1D wave
u0_expr = Expression(\
'100*pow(x[i]-0.5,2)*pow(x[i]-1.0,2)*(x[i]<=1.0)*(x[i]>=0.5)', i=direction)
class LeftRight(SubDomain):
    def inside(self, x, on_boundary):
        return (x[direction] < 1e-16 or x[direction] > 1.0 - 1e-16) \
        and on_boundary

q = 3
if myrank == 0: print '\npolynomial order = {}'.format(q)
alpha = 3.

Nxy = 40
h = 1./Nxy
mesh = UnitSquareMesh(Nxy, Nxy)
V = FunctionSpace(mesh, 'Lagrange', q)
Vl = FunctionSpace(mesh, 'Lagrange', 1)
c_max = 2
#Dt = h/(q*alpha*c_max)
Dt = 1e-3
checkdt_abc(Dt, h, q, c_max, True, True, 'centered')
if myrank == 0: print '\n\th = {}, Dt = {}'.format(h, Dt)

Wave = AcousticWave({'V':V, 'Vm':Vl})
#Wave.verbose = True
Wave.timestepper = 'centered'
Wave.lump = True
Wave.set_abc(mesh, LeftRight(), True)
# Medium ppties:
lam_expr = Expression('1.0 + 3.0*(x[i]<=0.25)', i=direction)
lam = interpolate(lam_expr, Vl)
Wave.update({'b':lam, 'a':1.0, 't0':0.0, 'tf':1.5, 'Dt':Dt,\
'u0init':interpolate(u0_expr, V), 'utinit':Function(V)})
Wave.ftime = lambda t: 0.0
sol, tmp = Wave.solve()
if not mycomm == None:  MPI.barrier(mycomm)

# Save plots:
try:
    boolplot = int(sys.argv[1])
except:
    boolplot = 10
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
    # Plot medium
    myplot.set_varname('lambda')
    myplot.plot_vtk(lam)

