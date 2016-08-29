import dolfin as dl
from dolfin import MPI
import numpy as np
from fenicstools.linalg.miscroutines import get_diagonal

mycomm = dl.mpi_comm_world()
mpirank = MPI.rank(mycomm)

mesh = dl.UnitSquareMesh(4,4)
V = dl.FunctionSpace(mesh, 'Lagrange', 1)

test, trial = dl.TestFunction(V), dl.TrialFunction(V)
M = dl.assemble(test*trial*dl.dx)
Mdiag = get_diagonal(M)
print 'rank={}'.format(mpirank), Mdiag.array()
print 'rank={}'.format(mpirank), \
V.dofmap().tabulate_all_coordinates(mesh).reshape((-1,2))
x = dl.Vector()
Mdiag.gather(x, np.array(range(V.dim()), "intc"))
if mpirank == 0:    print x.array()

dl.plot(mesh, interactive=True)
