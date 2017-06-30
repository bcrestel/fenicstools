"""
Code should be run in serial and in parallel to check that both produces the
same set of eigenvalues
"""

import dolfin as dl
from fenicstools.linalg.miscroutines import compute_eigfenics

mesh = dl.UnitSquareMesh(5,5)
V = dl.FunctionSpace(mesh, 'CG', 1)
test = dl.TestFunction(V)
trial = dl.TrialFunction(V)
M = dl.assemble(dl.inner(dl.nabla_grad(test), dl.nabla_grad(trial))*dl.dx)

if mesh.mpi_comm() == dl.mpi_comm_self():
    compute_eigfenics(M, 'eigM-serial.out')
else:
    compute_eigfenics(M, 'eigM-parallel.out')
