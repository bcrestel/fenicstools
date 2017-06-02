"""
Test weird results with TV (and TVPD, V_TVPD) in parallel
"""

import dolfin as dl
from fenicstools.regularization import TV

mpicomm = dl.mpi_comm_self()
mesh = dl.UnitSquareMesh(mpicomm, 50, 50)
V = dl.FunctionSpace(mesh, 'CG', 1)

#TODO: continue example; reproduce bug w/o TV
reg = TV({'Vm':V, 'eps':1e-3, 'k':1e-5, 'print':True})

m = dl.interpolate(dl.Expression('sin(pi*x[0])*sin(pi*x[1])'), V)
cost = reg.cost(m)
print cost
