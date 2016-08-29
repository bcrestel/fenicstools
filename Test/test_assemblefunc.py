from dolfin import *
mycomm = mpi_comm_world()
mpisize = MPI.size(mycomm)

mesh = UnitSquareMesh(50,50)
V = FunctionSpace(mesh,'Lagrange',2)
u = interpolate(Constant("2"), V)
print assemble(inner(u,u)*dx)
MPI.barrier(mycomm)

test = TestFunction(V)
b = assemble(inner(u,test)*dx)
print b[0], b[1], len(b.array())
MPI.barrier(mycomm)

trial = TrialFunction(V)
M = assemble(inner(trial, test)*dx)
a = M*u.vector()
print a[0], a[1], len(a.array())
