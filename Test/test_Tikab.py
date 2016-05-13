import numpy as np
import dolfin as dl

from fenicstools.regularization import Tikhonovab
from fenicstools.prior import LaplacianPrior
from fenicstools.miscfenics import setfct

mesh = dl.UnitSquareMesh(10,10)
V = dl.FunctionSpace(mesh,'Lagrange',2)
tikab = Tikhonovab({'Vm':V, 'gamma':1e-2, 'beta':1e-8})
reg = LaplacianPrior({'Vm':V, 'gamma':1e-2, 'beta':1e-8})

print 'test cost',
a, b = dl.Function(V), dl.Function(V)
for ii in range(10):
    a.vector()[:] = np.random.randn(V.dim())
    b.vector()[:] = np.random.randn(V.dim())
    costt = tikab.costab(a, b)
    costl = reg.cost(a) + reg.cost(b)
    err = np.abs(costt-costl)/np.abs(costl)
    if err > 1e-12:
        print '*** Warning: err={}'.format(err)
        sys.exit(1)
print '\t=>> OK!'

print 'gradient',
ab = dl.Function(V*V)
abh = dl.Function(V*V)
direct = dl.Function(V*V)
for ii in range(10):
    a.vector()[:] = np.random.randn(V.dim())
    b.vector()[:] = np.random.randn(V.dim())
    dl.assign(ab.sub(0), a)
    dl.assign(ab.sub(1), b)
    h = 1e-5
    gradab = tikab.gradab(a, b)
    for jj in range(10):
        direct.vector()[:] = np.random.randn((V*V).dim())
        graddirect = gradab.inner(direct.vector())
        setfct(abh, ab)
        abh.vector().axpy(h, direct.vector())
        ah, bh = abh.split(deepcopy=True)
        cost1 = tikab.costab(ah, bh)
        setfct(abh, ab)
        abh.vector().axpy(-h, direct.vector())
        ah, bh = abh.split(deepcopy=True)
        cost2 = tikab.costab(ah, bh)
        fdgrad = (cost1-cost2)/(2*h)
        err = np.abs(fdgrad-graddirect)/np.abs(graddirect)
        if err > 1e-6:
            print '*** Warning: err={}'.format(err)
            sys.exit(1)
print '\t=>> OK!'
