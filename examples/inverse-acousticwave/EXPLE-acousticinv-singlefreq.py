"""
Acoustic wave inverse problem with a single low-frequency,
and homogeneous Neumann boundary conditions everywhere.
"""

from fenicstools.plotfenics import PlotFenics
from fenicstools.acousticwave import AcousticWave
from fenicstools.sourceterms import PointSources, RickerWavelet
from fenicstools.observationoperator import TimeObsPtwise
from fenicstools.miscfenics import checkdt, setfct
try:
    from dolfin import UnitSquareMesh, FunctionSpace, Constant, \
    interpolate, Expression, Function, MPI, mpi_comm_world
    mycomm = mpi_comm_world()
    myrank = MPI.rank(mycomm)
except:
    from dolfin import UnitSquareMesh, FunctionSpace, Constant, \
    interpolate, Expression, Function
    mycomm = None
    myrank = 0

# Inputs:
fpeak = 0.4  #Hz
backgroundspeed = 1.0
perturbationspeed = 2.0
Nxy = 10
h = 1./Nxy
t0, tf = 0.0, 1.2
r = 2   # order polynomial approx
Dt = 1e-4
checkdt(Dt, h, r, perturbationspeed, True)

# 
mesh = UnitSquareMesh(Nxy, Nxy)
Vl = FunctionSpace(mesh, 'Lagrange', 1)
V = FunctionSpace(mesh, 'Lagrange', r)
# source:
Ricker = RickerWavelet(fpeak, 1e-10)
Pt = PointSources(V, [[.5,1.]])
mydelta = Pt[0].array()
def mysrc(tt):
    return Ricker(tt)*mydelta
# target medium:
lambda_target = Expression('x0 + dx*(' \
'(x[0]>=0.3)*(x[0]<=0.7)*(x[1]>=0.3)*(x[1]<=0.7))', \
x0=backgroundspeed, dx=perturbationspeed**2-backgroundspeed**2) 
lambda_target_fn = interpolate(lambda_target, Vl)
# initial medium:
lambda_init = Constant('x0', x0=backgroundspeed)
lambda_init_fn = interpolate(lambda_init, Vl)
#TODO: continue here
