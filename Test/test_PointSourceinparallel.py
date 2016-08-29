from dolfin import UnitSquareMesh, FunctionSpace, TestFunction, TrialFunction,\
Constant, Expression, assemble, dx, Point, PointSource, plot, interactive,\
inner, nabla_grad, Function, solve, MPI, mpi_comm_world
import numpy as np

mycomm = mpi_comm_world()
myrank = MPI.rank(mycomm)

mesh = UnitSquareMesh(2,2)
V = FunctionSpace(mesh,'Lagrange', 1)
trial = TrialFunction(V)
test = TestFunction(V)
f0 = Constant('0')
L0 = f0*test*dx
b = assemble(L0)
P = Point(0.5,0.5)
delta = PointSource(V, P, 1.0)
delta.apply(b)

print 'p{0}: max(|b|)={1},\n b={2}'.format(myrank, max(abs(b.array())), b.array())
