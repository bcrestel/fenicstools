from dolfin import UnitSquareMesh, FunctionSpace, TestFunction, TrialFunction,\
Constant, Expression, assemble, dx, Point, PointSource, plot, interactive,\
inner, nabla_grad, Function, solve, MPI, mpi_comm_world
import numpy as np

from fenicstools.sourceterms import PointSources

mycomm = mpi_comm_world()
myrank = MPI.rank(mycomm)

mesh = UnitSquareMesh(2,2)
V = FunctionSpace(mesh,'Lagrange', 1)
trial = TrialFunction(V)
test = TestFunction(V)
f0 = Constant('0')
L0 = f0*test*dx
b = assemble(L0)
P = Point(0.1,0.5)
delta = PointSource(V, P, 1.0)
delta.apply(b)

myown = PointSources(V, [[0.1,0.5], [0.9,0.5]])

print 'p{}: max(PointSource)={}, max(PointSources[0])={}, max(PointSources[1])={}'.format(\
myrank, max(abs(b.array())), max(abs(myown[0].array())), max(abs(myown[1].array())))
