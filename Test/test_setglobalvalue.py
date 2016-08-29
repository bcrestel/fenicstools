""" Test how to fill in a vector in parallel """

import numpy as np
import dolfin as dl

from fenicstools.linalg.miscroutines import setglobalvalue

try:
    from dolfin import MPI, mpi_comm_world
    mycomm = mpi_comm_world()
    mpisize = MPI.size(mycomm)
    mpirank = MPI.rank(mycomm)
except:
    mpisize = 1
    mpirank = 0



mesh = dl.UnitSquareMesh(2,2)
V = dl.FunctionSpace(mesh, 'Lagrange', 1)
print 'dim(V)={}'.format(V.dim())

for ii in range(V.dim()):
    u = dl.Function(V)
    setglobalvalue(u, ii, 1.0)
    x = dl.Vector()
    u.vector().gather(x, np.array(range(V.dim()), "intc"))
    if mpirank == 0:    print 'ii={}'.format(ii), x.array()

