import numpy as np
import sys
import dolfin as dl

from fenicstools.jointregularization import SumRegularization
from fenicstools.prior import LaplacianPrior
from fenicstools.regularization import TV
from fenicstools.miscfenics import setfct, createMixedFS

mesh = dl.UnitSquareMesh(20,20)
V = dl.FunctionSpace(mesh,'Lagrange',2)

TIKH = False

if TIKH:
    print 'Test Tikhonov regularization'
    m0a = dl.interpolate(dl.Expression('0.1*sin(pi*x[0])*sin(pi*x[1])'), V)
    m0b = m0a

    #tikab = Tikhonovab({'Vm':V, 'gamma':1e-4, 'beta':1e-4})
    #reg = LaplacianPrior({'Vm':V, 'gamma':1e-4, 'beta':1e-4})

    reg1 = LaplacianPrior({'Vm':V, 'gamma':1e-4, 'beta':1e-4, 'm0':m0a})
    reg2 = LaplacianPrior({'Vm':V, 'gamma':1e-4, 'beta':1e-4, 'm0':m0a})
    reg1jt = LaplacianPrior({'Vm':V, 'gamma':1e-4, 'beta':1e-4, 'm0':m0a})
    reg2jt = LaplacianPrior({'Vm':V, 'gamma':1e-4, 'beta':1e-4, 'm0':m0a})
else:
    print 'Test TV regularization'
    reg1 = TV({'Vm':V, 'k':1e-4, 'eps':1e-4})
    reg2 = TV({'Vm':V, 'k':1e-4, 'eps':1e-4})
    reg1jt = TV({'Vm':V, 'k':1e-4, 'eps':1e-4})
    reg2jt = TV({'Vm':V, 'k':1e-4, 'eps':1e-4})

sumregul = SumRegularization(reg1jt, reg2jt)

print 'cost'
a, b = dl.Function(V), dl.Function(V)
for ii in range(10):
    a.vector()[:] = np.random.randn(V.dim())
    b.vector()[:] = np.random.randn(V.dim())
    costl = reg1.cost(a) + reg2.cost(b)
    costt2 = sumregul.costab(a, b)
    err2 = np.abs(costt2-costl)/np.abs(costl)
    print 'ii={}: err={}'.format(ii, err2)
    if err2 > 1e-14:
        print '*** WARNING: error too large'
        sys.exit(1)
print '\t=>> cost OK!'

print 'gradient'
VV = createMixedFS(V, V)
grad = dl.Function(VV)
grada = dl.Function(V)
gradb = dl.Function(V)
for ii in range(10):
    a.vector()[:] = np.random.randn(V.dim())
    b.vector()[:] = np.random.randn(V.dim())
    setfct(grada, reg1.grad(a))
    setfct(gradb, reg2.grad(b))
    dl.assign(grad.sub(0), grada)
    dl.assign(grad.sub(1), gradb)
    sumgradab = sumregul.gradab(a, b)
    err2 = dl.norm(sumgradab - grad.vector())/dl.norm(grad.vector())
    print 'ii={}: err={}'.format(ii, err2)
    if err2 > 1e-14:
        print '*** WARNING: error too large'
        sys.exit(1)
print '\t=>> gradient OK!'

print 'Hessian'
VV = createMixedFS(V, V)
hess = dl.Function(VV)
hessa = dl.Function(V)
hessb = dl.Function(V)
for ii in range(10):
    a.vector()[:] = np.random.randn(V.dim())
    b.vector()[:] = np.random.randn(V.dim())

    reg1.assemble_hessian(a)
    reg2.assemble_hessian(b)
    sumregul.assemble_hessianab(a, b)

    for jj in range(5):
        a.vector()[:] = np.random.randn(V.dim())
        b.vector()[:] = np.random.randn(V.dim())
        setfct(hessa, reg1.hessian(a.vector()))
        setfct(hessb, reg2.hessian(b.vector()))
        dl.assign(hess.sub(0), hessa)
        dl.assign(hess.sub(1), hessb)
        sumhessab = sumregul.hessianab(a.vector(), b.vector())
        err2 = dl.norm(sumhessab - hess.vector())/dl.norm(hess.vector())
        print 'ii={}, j={}: err={}'.format(ii, jj, err2)
        if err2 > 1e-14:
            print '*** WARNING: error too large'
            sys.exit(1)
    print ''
print '\t=>> Hessian OK!'

print 'preconditioner'
for ii in range(10):
    a.vector()[:] = np.random.randn(V.dim())
    b.vector()[:] = np.random.randn(V.dim())

    reg1.assemble_hessian(a)
    reg2.assemble_hessian(b)
    sumregul.assemble_hessianab(a, b)

    prec1 = reg1.getprecond()
    prec2 = reg2.getprecond()
    precjoint = sumregul.getprecond()

    for jj in range(5):
        grada.vector()[:] = np.random.randn(V.dim())
        gradb.vector()[:] = np.random.randn(V.dim())
        dl.assign(grad.sub(0), grada)
        dl.assign(grad.sub(1), gradb)

        prec1.solve(hessa.vector(), grada.vector())
        prec2.solve(hessb.vector(), gradb.vector())
        dl.assign(hess.sub(0), hessa)
        dl.assign(hess.sub(1), hessb)

        sumhessab.zero()
        precjoint.solve(sumhessab, grad.vector())
        err2 = dl.norm(sumhessab - hess.vector())/dl.norm(hess.vector())
        print 'ii={}, j={}: err2={}'.format(ii, jj, err2)
        if err2 > 1e-13:
            print '*** WARNING: error too large'
            sys.exit(1)
    print ''
print '\t=>> preconditioner OK!'
