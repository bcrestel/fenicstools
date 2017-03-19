import dolfin as dl
import numpy as np

from fenicstools.jointregularization import normalizedcrossgradient
from fenicstools.miscfenics import setfct

from hippylib.linalg import vector2Function

print 'Check exact results (cost only):'
print 'Test 1'
err = 1.0
N = 10
while err > 1e-6:
    N = 2*N
    mesh = dl.UnitSquareMesh(N, N)
    V = dl.FunctionSpace(mesh,'Lagrange', 1)
    cg = normalizedcrossgradient(V*V, {'eps':0.0})
    a = dl.interpolate(dl.Expression('x[0]'), V)
    b = dl.interpolate(dl.Expression('x[1]'), V)
    ncg = cg.costab(a,b)
    err = np.abs(ncg-0.5)/0.5
    print 'N={}, ncg(x,y)={:.6f}, err={:.3e}'.format(N, ncg, err)
print 'Test 2'
err = 1.0
N = 10
while err > 1e-6:
    N = 2*N
    mesh = dl.UnitSquareMesh(N, N)
    V = dl.FunctionSpace(mesh,'Lagrange', 2)
    cg = normalizedcrossgradient(V*V, {'eps':0.0})
    a = dl.interpolate(dl.Expression('x[0]*x[0]'), V)
    b = dl.interpolate(dl.Expression('x[1]*x[1]'), V)
    ncg = cg.costab(a,b)
    err = np.abs(ncg-0.5)/0.5
    print 'N={}, ncg(x^2,y^2)={:.6f}, err={:.3e}'.format(N, ncg, err)
#print 'Test 3'
#err = 1.0
#N = 160
#while err > 1e-6:
#    N = 2*N
#    mesh = dl.UnitSquareMesh(N, N)
#    V = dl.FunctionSpace(mesh,'Lagrange', 3)
#    cg = normalizedcrossgradient(V*V, {'eps':0.0})
#    a = dl.interpolate(dl.Expression('x[0]*x[1]'), V)
#    b = dl.interpolate(dl.Expression('x[0]*x[0]*x[1]*x[1]'), V)
#    ncg = cg.costab(a,b)
#    err = np.abs(ncg)
#    print 'N={}, ncg(xy,x^2y^2)={:.6f}, err={:.3e}'.format(N, ncg, err)


#############################################################
mesh = dl.UnitSquareMesh(20, 20)
V = dl.FunctionSpace(mesh,'Lagrange', 1)
cg = normalizedcrossgradient(V*V)

print '\nsinusoidal cross-gradients'
for ii in range(5):
    a = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])', n=ii), V)
    for jj in range(5):
        b = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])', n=jj), V)
        print cg.costab(a, b),
    print ''

#############################################################
print '\ncheck gradient'
ak, bk = dl.Function(V), dl.Function(V)
directab = dl.Function(V*V)
h = 1e-6
for ii in range(5):
    print 'ii={}'.format(ii)
    a = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])', n=ii), V)
    b = dl.interpolate(dl.Expression('pow(x[0], n)*pow(x[1], n)', n=ii), V)
    grad = cg.gradab(a, b)
    for jj in range(5):
        directa = dl.interpolate(dl.Expression('pow(x[0], n)*pow(x[1], n)', n=jj+1), V)
        directb = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])', n=jj+1), V)
        dl.assign(directab.sub(0), directa)
        dl.assign(directab.sub(1), directb)
        gradxdir = grad.inner(directab.vector())
        setfct(ak, a)
        setfct(bk, b)
        ak.vector().axpy(h, directa.vector())
        bk.vector().axpy(h, directb.vector())
        cost1 = cg.costab(ak, bk)
        setfct(ak, a)
        setfct(bk, b)
        ak.vector().axpy(-h, directa.vector())
        bk.vector().axpy(-h, directb.vector())
        cost2 = cg.costab(ak, bk)
        gradfddirect = (cost1-cost2)/(2*h)
        err = np.abs(gradxdir-gradfddirect)/np.abs(gradxdir)
        print 'grad={}, fd={}, err={}'.format(gradxdir, gradfddirect, err),
        if err > 1e-6:  print '\t =>> Warning!'
        else:   print ''

# Check other directions of Hessian
print '\n\ncheck Hessian--block(1,1)'
ak, bk = dl.Function(V), dl.Function(V)
directab = dl.Function(V*V)
h = 1e-5
for ii in range(5):
    print 'ii={}'.format(ii)
    a = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])', n=ii), V)
    b = dl.interpolate(dl.Expression('pow(x[0], n)*pow(x[1], n)', n=ii), V)
    cg.assemble_hessianab(a, b)
    for jj in range(5):
        directa = dl.interpolate(dl.Expression('pow(x[0], n)*pow(x[1], n)', n=jj+1), V)
        #directb = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])', n=jj+1), V)
        directb = dl.interpolate(dl.Constant('0.0'),V)
        dl.assign(directab.sub(0), directa)
        dl.assign(directab.sub(1), directb)
        Hxdir = cg.hessianab(directa, directb)
        #
        setfct(ak, a)
        setfct(bk, b)
        ak.vector().axpy(h, directa.vector())
        bk.vector().axpy(h, directb.vector())
        grad1 = cg.gradab(ak, bk)
        setfct(ak, a)
        setfct(bk, b)
        ak.vector().axpy(-h, directa.vector())
        bk.vector().axpy(-h, directb.vector())
        grad2 = cg.gradab(ak, bk)
        hessfddir = (grad1-grad2)/(2*h)
        #
        Hxdirfun = vector2Function(Hxdir, V*V)
        hessfddirfun = vector2Function(hessfddir, V*V)
        Hxdira, Hxdirb = Hxdirfun.split(deepcopy=True)
        hessfddira, hessfddirb = hessfddirfun.split(deepcopy=True)
        normHxdira = dl.norm(Hxdira.vector())
        if normHxdira > 1e-16:
            err = dl.norm(Hxdira.vector() - hessfddira.vector())/normHxdira
        else:
            err = dl.norm(Hxdira.vector() - hessfddira.vector())
#        err = np.linalg.norm((Hxdir-hessfddir).array())/np.linalg.norm(Hxdir.array())
        print '|Hxdir|={}, err={}'.format(normHxdira, err),
        if err > 1e-6:  print '\t =>> Warning!'
        else:   print ''
