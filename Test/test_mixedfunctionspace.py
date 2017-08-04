"""
Goal is to test how mixed function space behaves in Fenics
"""
from dolfin import *
import numpy as np
from fenicstools.miscfenics import createMixedFS, createMixedFSi

mpicomm = mpi_comm_world()
mpirank = MPI.rank(mpicomm)

#mpicommmesh = mpicomm
mpicommmesh = mpi_comm_self()
mesh = UnitSquareMesh(mpicommmesh, 20,20)

V = FunctionSpace(mesh, 'CG', 1)
#VV = createMixedFS(V, V)
VV = createMixedFSi([V, V, V, V])

if mpirank == 0:
    print 'check gradient of m=(1.0,2.0,3.0,4.0)  returns 0'
m = Function(VV)
m1 = interpolate(Constant("1"), V)
m2 = interpolate(Constant("2"), V)
m3 = interpolate(Constant("3"), V)
m4 = interpolate(Constant("4"), V)
assign(m.sub(0), m1)
assign(m.sub(1), m2)
assign(m.sub(2), m3)
assign(m.sub(3), m4)
normgrad = assemble(inner(nabla_grad(m), nabla_grad(m))*dx)
if mpirank == 0:
    print normgrad

if mpirank == 0:
    print 'same with m=(x,y,3.0,4.0); should be 2'
m1 = interpolate(Expression("x[0]", degree=10), V)
m2 = interpolate(Expression("x[1]", degree=10), V)
assign(m.sub(0), m1)
assign(m.sub(1), m2)
assign(m.sub(2), m3)
assign(m.sub(3), m4)
normgrad = assemble(inner(nabla_grad(m), nabla_grad(m))*dx)
if mpirank == 0:
    print normgrad

