"""
Goal is to test how mixed function space behaves in Fenics
"""
from dolfin import *
import numpy as np

mpicomm = mpi_comm_world()
mpirank = MPI.rank(mpicomm)

#mpicommmesh = mpicomm
mpicommmesh = mpi_comm_self()
mesh = UnitSquareMesh(mpicommmesh, 20,20)

try:
    Vfem = FiniteElement('CG', mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, Vfem)
    VV = FunctionSpace(mesh, Vfem*Vfem)
    Vwfem = FiniteElement('DG', mesh.ufl_cell(), 0)
    Vw = FunctionSpace(mesh, Vwfem)
    VwVw = FunctionSpace(mesh, Vwfem*Vwfem)
except:
    V = FunctionSpace(mesh, 'CG', 1)
    VV = V*V
    Vw = FunctionSpace(mesh, 'DG', 0)
    VwVw = Vw*Vw

if mpirank == 0:
    print 'check gradient of m=(m1,m2) with m1,m2 cst returns 0'
m = Function(VV)
m1 = interpolate(Constant("1"), V)
m2 = interpolate(Constant("2"), V)
assign(m.sub(0), m1)
assign(m.sub(1), m2)
normgrad = assemble(inner(nabla_grad(m), nabla_grad(m))*dx)
if mpirank == 0:
    print normgrad

if mpirank == 0:
    print 'same with m=(x,2.0); should be 1'
m1 = interpolate(Expression("x[0]", degree=10), V)
assign(m.sub(0), m1)
assign(m.sub(1), m2)
normgrad = assemble(inner(nabla_grad(m), nabla_grad(m))*dx)
if mpirank == 0:
    print normgrad

