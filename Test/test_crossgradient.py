import dolfin as dl
import numpy as np

from fenicstools.jointregularization import crossgradient
from fenicstools.miscfenics import setfct

print 'Check exact results (cost only):'
print 'Test 1'
err = 1.0
N = 10
while err > 1e-6:
    N = 2*N
    mesh = dl.UnitSquareMesh(N, N)
    V = dl.FunctionSpace(mesh,'Lagrange', 1)
    cg = crossgradient(V*V)
    a = dl.interpolate(dl.Expression('x[0]'), V)
    b = dl.interpolate(dl.Expression('x[1]'), V)
    cgc = cg.costab(a,b)
    err = np.abs(cgc-0.5)/0.5
    print 'N={}, x*y={:.6f}, err={:.3e}'.format(N, cgc, err)
print 'Test 2'
err = 1.0
N = 20
while err > 1e-6:
    N = 2*N
    mesh = dl.UnitSquareMesh(N, N)
    V = dl.FunctionSpace(mesh,'Lagrange', 2)
    cg = crossgradient(V*V)
    a = dl.interpolate(dl.Expression('x[0]*x[0]'), V)
    b = dl.interpolate(dl.Expression('x[1]*x[1]'), V)
    cgc = cg.costab(a,b)
    err = np.abs(cgc-8./9.)/(8./9.)
    print 'N={}, x^2*y^2={:.6f}, err={:.3e}'.format(N, cgc, err)
print 'Test 3'
err = 1.0
N = 40
while err > 1e-6:
    N = 2*N
    mesh = dl.UnitSquareMesh(N, N)
    V = dl.FunctionSpace(mesh,'Lagrange', 3)
    cg = crossgradient(V*V)
    a = dl.interpolate(dl.Expression('x[0]*x[1]'), V)
    b = dl.interpolate(dl.Expression('x[0]*x[0]*x[1]*x[1]'), V)
    cgc = cg.costab(a,b)
    err = np.abs(cgc)
    print 'N={}, xy*x^2y^2={:.6f}, err={:.3e}'.format(N, cgc, err)


#############################################################
mesh = dl.UnitSquareMesh(20, 20)
V = dl.FunctionSpace(mesh,'Lagrange', 1)
cg = crossgradient(V*V)

print '\nsinusoidal cross-gradients'
for ii in range(5):
    a = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])', n=ii), V)
    for jj in range(5):
        b = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])', n=ii), V)
        print cg.costab(a, b),
    print ''

#############################################################
print '\ncheck gradient'
ak, bk = dl.Function(V), dl.Function(V)
directab = dl.Function(V*V)
h = 1e-5
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

print 'check Hessian'
ak, bk = dl.Function(V), dl.Function(V)
directab = dl.Function(V*V)
h = 1e-5
for ii in range(5):
    print 'ii={}'.format(ii)
    a = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])', n=ii), V)
    b = dl.interpolate(dl.Expression('pow(x[0], n)*pow(x[1], n)', n=ii), V)
    for jj in range(5):
        directa = dl.interpolate(dl.Expression('pow(x[0], n)*pow(x[1], n)', n=jj+1), V)
        directb = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])', n=jj+1), V)
        dl.assign(directab.sub(0), directa)
        dl.assign(directab.sub(1), directb)
        cg.assemble_hessianab(a, b)
        Hxdir = cg.hessianab(directa, directb)
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
        err = np.linalg.norm((Hxdir-hessfddir).array())/np.linalg.norm(Hxdir.array())
        print '|Hxdir|={}, err={}'.format(np.linalg.norm(Hxdir.array()), err),
        if err > 1e-6:  print '\t =>> Warning!'
        else:   print ''
