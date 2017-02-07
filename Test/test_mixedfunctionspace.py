"""
Goal is to test how mixed function space behaves in Fenics
"""
from dolfin import *
import numpy as np

mesh = UnitSquareMesh(20,20)

V = FunctionSpace(mesh, 'CG', 1)
VV = V*V
Vw = FunctionSpace(mesh, 'DG', 0)
VwVw = Vw*Vw

print 'check gradient of m=(m1,m2) with m1,m2 cst returns 0'
m = Function(VV)
m1 = interpolate(Constant("1"), V)
m2 = interpolate(Constant("2"), V)
assign(m.sub(0), m1)
assign(m.sub(1), m2)
normgrad = assemble(inner(nabla_grad(m), nabla_grad(m))*dx)
print normgrad

print 'same with m=(x,2.0); should be 1'
m1 = interpolate(Expression("x[0]"), V)
assign(m.sub(0), m1)
assign(m.sub(1), m2)
normgrad = assemble(inner(nabla_grad(m), nabla_grad(m))*dx)
print normgrad

