"""
Test how partial derivatives (dx) work in fenics
"""
import sys
import numpy as np
from dolfin import *

from fenicstools.miscfenics import setfct, createMixedFS

mesh = UnitSquareMesh(20,20)

V = FunctionSpace(mesh, 'CG', 1)
VV = createMixedFS(V, V)

Vw = FunctionSpace(mesh, 'DG', 0)
VwVw = createMixedFS(Vw, Vw)

mpirank = MPI.rank(mesh.mpi_comm())
PRINT = (mpirank == 0)

if PRINT:   print 'Test 1'
test = TestFunction(V)
m = Function(V)
for ii in range(10):
    m.vector()[:] = np.random.randn(m.vector().local_size())
    v1 = assemble(inner(nabla_grad(test), nabla_grad(m))*dx)
    v1n = v1.norm('l2')
    v2 = assemble(inner(test.dx(0), m.dx(0))*dx + inner(test.dx(1), m.dx(1))*dx)
    v2n = v2.norm('l2')
    err = (v1-v2).norm('l2')/v1n
    if PRINT:   print 'v1n={}, v2n={}, err={:.2e}'.format(v1n, v2n, err)
    if err > 1e-14:
        if PRINT:   print '*** relative error too large!'
        sys.exit(1)

if PRINT:   print 'Test 2'
test = TestFunction(VwVw)
testx, testy = TestFunctions(VwVw)
m = Function(V)
for ii in range(10):
    m.vector()[:] = np.random.randn(m.vector().local_size())
    v1 = assemble(inner(test, nabla_grad(m))*dx)
    v1n = v1.norm('l2')
    v2 = assemble(inner(testx, m.dx(0))*dx + inner(testy, m.dx(1))*dx)
    v2n = v2.norm('l2')
    err = (v1-v2).norm('l2')/v1n
    if PRINT:   print 'v1n={}, v2n={}, err={:.2e}'.format(v1n, v2n, err)
    if err > 1e-14:
        if PRINT:   print '*** relative error too large!'
        sys.exit(1)

if PRINT:   print 'Test 3'
test = TestFunction(VwVw)
m = Function(VV)
for ii in range(10):
    m.vector()[:] = np.random.randn(m.vector().local_size())
    v1 = assemble(inner(test, nabla_grad(m)[0,:])*dx + inner(test, nabla_grad(m)[1,:])*dx)
    v1n = v1.norm('l2')
    v2 = assemble(inner(test, m.dx(0))*dx + inner(test, m.dx(1))*dx)
    v2n = v2.norm('l2')
    err = (v1-v2).norm('l2')/v1n
    if PRINT:   print 'v1n={}, v2n={}, err={:.2e}'.format(v1n, v2n, err)
    if err > 1e-14:
        if PRINT:   print '*** relative error too large!'
        sys.exit(1)

if PRINT:   print 'Test 4'
test = TestFunction(VwVw)
test1, test2 = test
m = Function(VV)
for ii in range(10):
    m.vector()[:] = np.random.randn(m.vector().local_size())
    m1, m2 = m.split(deepcopy=True)
    v1 = assemble(inner(test1, m1.dx(0))*dx + inner(test2, m2.dx(0))*dx)
    v1n = v1.norm('l2')
    v2 = assemble(inner(test, m.dx(0))*dx)
    v2n = v2.norm('l2')
    err = (v1-v2).norm('l2')/v1n
    if PRINT:   print 'v1n={}, v2n={}, err={:.2e}'.format(v1n, v2n, err)
    if err > 1e-14:
        if PRINT:   print '*** relative error too large!'
        sys.exit(1)

if PRINT:   print 'Test 5'
VwVwVwVw = createMixedFS(VwVw, VwVw)
testx, testy = TestFunctions(VwVwVwVw)
test1, test2 = TestFunction(VwVw)
m = Function(VV)
v1xf, v1yf = Function(VwVw), Function(VwVw)
v1f = Function(VwVwVwVw)
for ii in range(10):
    m.vector()[:] = np.random.randn(m.vector().local_size())
    m1, m2 = m.split(deepcopy=True)
    v1x = assemble(inner(test1, m1.dx(0))*dx + inner(test2, m2.dx(0))*dx)
    setfct(v1xf, v1x)
    v1y = assemble(inner(test1, m1.dx(1))*dx + inner(test2, m2.dx(1))*dx)
    setfct(v1yf, v1y)
    assign(v1f.sub(0), v1xf)
    assign(v1f.sub(1), v1yf)
    v1 = v1f.vector()
    v1n = v1.norm('l2')
    v2 = assemble(inner(testx, m.dx(0))*dx + inner(testy, m.dx(1))*dx)
    v2n = v2.norm('l2')
    err = (v1-v2).norm('l2')/v1n
    if PRINT:   print 'v1n={}, v2n={}, err={:.2e}'.format(v1n, v2n, err)
    if err > 1e-14:
        if PRINT:   print '*** relative error too large!'
        sys.exit(1)
