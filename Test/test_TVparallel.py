"""
Test weird results with sqrt
In Fenics 1.6, with 
    mpicomm = dl.mpi_comm_self()
results are,
    cost = cost_serial * nb_proc
"""
import sys
import dolfin as dl

#mpicomm = dl.mpi_comm_world()
mpicomm = dl.mpi_comm_self()

mesh = dl.UnitSquareMesh(mpicomm, 50, 50)
V = dl.FunctionSpace(mesh, 'CG', 1)

#m = dl.interpolate(dl.Expression('4.0*x[0]*x[0]', degree=1), V)
x = dl.SpatialCoordinate(mesh)
m = 4.0*x[0]*x[0]

cost = dl.assemble(dl.sqrt(m)*dl.dx)
#cost = dl.assemble(dl.exp(m)*dl.dx)

print cost

