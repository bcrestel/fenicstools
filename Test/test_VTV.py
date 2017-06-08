"""
Test joint regularization VTV again finite difference
"""
import numpy as np
import dolfin as dl

from fenicstools.miscfenics import setfct, createMixedFS
from fenicstools.regularization import TV
from fenicstools.jointregularization import VTV, V_TV


print 'Check VTV against TV (cost)'
mesh = dl.UnitSquareMesh(20,20)
V = dl.FunctionSpace(mesh, 'Lagrange', 1)

regTV = TV({'Vm':V, 'k':1.0, 'eps':1e-6})
#regVTV = VTV(V, {'k':1.0, 'eps':1e-6})
regVTV = V_TV(V, {'k':1.0, 'eps':1e-6})

m = dl.interpolate(dl.Expression("1.0", degree=10), V)
m1 = m
m2 = m
costTV = regTV.cost(m)
costVTV = regVTV.costab(m1, m2)
err = np.abs(costTV - costVTV)/np.abs(costTV)
print 'costTV={}, costVTV={}, err={:.2e}'.format(costTV, costVTV, err),
if err > 1e-12:
    print '\tWARNING!'
else:   print ''

m = dl.interpolate(dl.Expression("x[0]+x[1]", degree=10), V)
m1 = dl.interpolate(dl.Expression("x[0]", degree=10), V)
m2 = dl.interpolate(dl.Expression("x[1]", degree=10), V)
costTV = regTV.cost(m)
costVTV = regVTV.costab(m1, m2)
err = np.abs(costTV - costVTV)/np.abs(costTV)
print 'costTV={}, costVTV={}, err={:.2e}'.format(costTV, costVTV, err),
if err > 1e-12:
    print '\tWARNING!'
else:   print ''

m = dl.interpolate(dl.Expression("sqrt(2)*sin(pi*x[0])*sin(pi*x[1])", degree=10), V)
m1 = dl.interpolate(dl.Expression("sin(pi*x[0])*sin(pi*x[1])", degree=10), V)
m2 = dl.interpolate(dl.Expression("sin(pi*x[0])*sin(pi*x[1])", degree=10), V)
costTV = regTV.cost(m)
costVTV = regVTV.costab(m1, m2)
err = np.abs(costTV - costVTV)/np.abs(costTV)
print 'costTV={}, costVTV={}, err={:.2e}'.format(costTV, costVTV, err),
if err > 1e-12:
    print '\tWARNING!'
else:   print ''


print '\nCheck gradient with FD'
ak, bk = dl.Function(V), dl.Function(V)
VV = createMixedFS(V, V)
directab = dl.Function(VV)
HH = [1e-4, 1e-5, 1e-6, 1e-7]
for ii in range(5):
    print 'ii={}'.format(ii)
    a = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])',\
    n=ii, degree=10), V)
    b = dl.interpolate(dl.Expression('pow(x[0], n)*pow(x[1], n)',\
    n=ii, degree=10), V)
    grad = regVTV.gradab(a, b)
    for jj in range(5):
        directa = dl.interpolate(dl.Expression('pow(x[0], n)*pow(x[1], n)',\
        n=jj+1, degree=10), V)
        directb = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])',\
        n=jj+1, degree=10), V)
        dl.assign(directab.sub(0), directa)
        dl.assign(directab.sub(1), directb)
        gradxdir = grad.inner(directab.vector())
        for h in HH:
            setfct(ak, a)
            setfct(bk, b)
            ak.vector().axpy(h, directa.vector())
            bk.vector().axpy(h, directb.vector())
            cost1 = regVTV.costab(ak, bk)
            setfct(ak, a)
            setfct(bk, b)
            ak.vector().axpy(-h, directa.vector())
            bk.vector().axpy(-h, directb.vector())
            cost2 = regVTV.costab(ak, bk)
            gradfddirect = (cost1-cost2)/(2*h)
            if np.abs(gradxdir) < 1e-16:
                err = np.abs(gradxdir-gradfddirect)
            else:
                err = np.abs(gradxdir-gradfddirect)/np.abs(gradxdir)
            print '\th={}, grad={}, fd={}, err={:.2e}'.format(h, gradxdir, gradfddirect, err),
            if err > 1e-6:  print '\t =>> Warning!'
            else:   
                print '\n'
                break

print '\nCheck Hessian with FD'
ak, bk = dl.Function(V), dl.Function(V)
directab = dl.Function(VV)
HH = [1e-5, 1e-6, 1e-7]
for ii in range(5):
    print 'ii={}'.format(ii)
    a = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])',\
    n=ii, degree=10), V)
    b = dl.interpolate(dl.Expression('pow(x[0], n)*pow(x[1], n)',\
    n=ii, degree=10), V)
    for jj in range(5):
        directa = dl.interpolate(dl.Expression('pow(x[0], n)*pow(x[1], n)',\
        n=jj+1, degree=10), V)
        directb = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])',\
        n=jj+1, degree=10), V)
        dl.assign(directab.sub(0), directa)
        dl.assign(directab.sub(1), directb)
        regVTV.assemble_hessianab(a, b)
        Hxdir = regVTV.hessianab(directa, directb)
        for h in HH:
            setfct(ak, a)
            setfct(bk, b)
            ak.vector().axpy(h, directa.vector())
            bk.vector().axpy(h, directb.vector())
            grad1 = regVTV.gradab(ak, bk)
            setfct(ak, a)
            setfct(bk, b)
            ak.vector().axpy(-h, directa.vector())
            bk.vector().axpy(-h, directb.vector())
            grad2 = regVTV.gradab(ak, bk)
            hessfddir = (grad1-grad2)/(2*h)
            if np.linalg.norm(Hxdir.array()) < 1e-16:
                err = np.linalg.norm((Hxdir-hessfddir).array())
            else:
                err = np.linalg.norm((Hxdir-hessfddir).array())/np.linalg.norm(Hxdir.array())
            print '\th={}, |Hxdir|={}, |HxdirFD|={}, err={}'.format(h, \
            np.linalg.norm(Hxdir.array()), np.linalg.norm(hessfddir.array()), err),
            if err > 1e-6:  print '\t =>> Warning!'
            else:   
                print '\n'
                break
