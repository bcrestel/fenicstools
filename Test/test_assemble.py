from time import time
from dolfin import UnitSquareMesh, FunctionSpace, TrialFunction, TestFunction, \
Function, inner, nabla_grad, dx, assemble

mesh = UnitSquareMesh(5, 5, "crossed")

V = FunctionSpace(mesh, 'Lagrange', 5)
Vl = FunctionSpace(mesh, 'Lagrange', 1)

trial = TrialFunction(V)
test = TestFunction(V)

lam1 = Function(Vl)
lam2 = Function(Vl)
lamV = Function(V)

lam1.vector()[:] = 1.0
lam2.vector()[:] = 1.0
lamV.vector()[:] = 1.0

weak_1 = lam1*inner(nabla_grad(trial), nabla_grad(test))*dx
weak_2 = inner(lam2*nabla_grad(trial), nabla_grad(test))*dx
weak_V = inner(lamV*nabla_grad(trial), nabla_grad(test))*dx

print 'Start assembling K1'
t0 = time()
K1 = assemble(weak_1)
t1 = time()
print 'Time to assemble K1 = {}'.format(t1-t0)

print 'Start assembling K2'
t0 = time()
K2 = assemble(weak_2)
t1 = time()
print 'Time to assemble K2 = {}'.format(t1-t0)

print 'Start assembling KV'
t0 = time()
KV = assemble(weak_V)
t1 = time()
print 'Time to assemble KV = {}'.format(t1-t0)
