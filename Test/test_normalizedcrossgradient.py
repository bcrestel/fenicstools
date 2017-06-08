import dolfin as dl
import numpy as np

from fenicstools.jointregularization import normalizedcrossgradient
from fenicstools.miscfenics import setfct, createMixedFS

from hippylib.linalg import vector2Function

print 'Check exact results (cost only):'
print 'Test 1'
err = 1.0
N = 10
while err > 1e-6:
    N = 2*N
    mesh = dl.UnitSquareMesh(N, N)
    V = dl.FunctionSpace(mesh,'Lagrange', 1)
    VV = createMixedFS(V, V)
    cg = normalizedcrossgradient(VV, {'eps':0.0})
    a = dl.interpolate(dl.Expression('x[0]', degree=10), V)
    b = dl.interpolate(dl.Expression('x[1]', degree=10), V)
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
    VV = createMixedFS(V, V)
    cg = normalizedcrossgradient(VV, {'eps':0.0})
    a = dl.interpolate(dl.Expression('x[0]*x[0]', degree=10), V)
    b = dl.interpolate(dl.Expression('x[1]*x[1]', degree=10), V)
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
VV = createMixedFS(V, V)
cg = normalizedcrossgradient(VV)

print '\nsinusoidal cross-gradients'
for ii in range(5):
    a = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])',\
    n=ii, degree=10), V)
    for jj in range(5):
        b = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])',\
        n=jj, degree=10), V)
        print cg.costab(a, b),
    print ''

#############################################################
print '\ncheck gradient'
ak, bk = dl.Function(V), dl.Function(V)
directab = dl.Function(VV)
H = [1e-6,1e-7,1e-5]
for ii in range(5):
    print 'ii={}'.format(ii)
    a = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])',\
    n=ii, degree=10), V)
    b = dl.interpolate(dl.Expression('pow(x[0], n)*pow(x[1], n)',\
    n=ii, degree=10), V)
    grad = cg.gradab(a, b)
    for jj in range(5):
        directa = dl.interpolate(dl.Expression('pow(x[0], n)*pow(x[1], n)',\
        n=jj+1, degree=10), V)
        directb = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])',\
        n=jj+1, degree=10), V)
        dl.assign(directab.sub(0), directa)
        dl.assign(directab.sub(1), directb)
        gradxdir = grad.inner(directab.vector())
        print 'grad={}, '.format(gradxdir)
        for h in H:
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
            if np.abs(gradxdir) > 1e-16:
                err = np.abs(gradxdir-gradfddirect)/np.abs(gradxdir)
            else:
                err = np.abs(gradxdir-gradfddirect)
            print '\th={}, fd={}, err={:.2e}'.format(h, gradfddirect, err),
            if err < 1e-6:
                print '\t =>> OK!'
                break
            else:   print ''

# Check other directions of Hessian
print '\n\ncheck Hessian--block(1,1)'
ak, bk = dl.Function(V), dl.Function(V)
directab = dl.Function(VV)
H = [1e-6,1e-7,1e-5]
for ii in range(5):
    print 'ii={}'.format(ii)
    a = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])',\
    n=ii, degree=10), V)
    b = dl.interpolate(dl.Expression('pow(x[0], n)*pow(x[1], n)',\
    n=ii, degree=10), V)
    cg.assemble_hessianab(a, b)
    for jj in range(5):
        directa = dl.interpolate(dl.Expression('pow(x[0], n)*pow(x[1], n)',\
        n=jj+1, degree=10), V)
        directb = dl.interpolate(dl.Constant('0.0'),V)
        dl.assign(directab.sub(0), directa)
        dl.assign(directab.sub(1), directb)
        Hxdir = cg.hessianab(directa, directb)
        Hxdirfun = vector2Function(Hxdir, VV)
        Hxdira, Hxdirb = Hxdirfun.split(deepcopy=True)
        normHxdira = dl.norm(Hxdira.vector())
        normHxdirb = dl.norm(Hxdirb.vector())
        print '|Hxdira|={}, |Hxdirb|={}'.format(normHxdira, normHxdirb)
        #
        for h in H:
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
            hessfddirfun = vector2Function(hessfddir, VV)
            hessfddira, hessfddirb = hessfddirfun.split(deepcopy=True)
            if normHxdira > 1e-16:
                err11 = dl.norm(Hxdira.vector() - hessfddira.vector())/normHxdira
            else:
                err11 = dl.norm(Hxdira.vector() - hessfddira.vector())
            if normHxdirb > 1e-16:
                err12 = dl.norm(Hxdirb.vector() - hessfddirb.vector())/normHxdirb
            else:
                err12 = dl.norm(Hxdirb.vector() - hessfddirb.vector())
            print '\th={}, |Hxdira|={}, |Hxdirb|={}, err11={:.2e}, err12={:.2e}'.format(\
            h, dl.norm(hessfddira.vector()), dl.norm(hessfddirb.vector()), err11, err12),
            if max(err11, err12) < 1e-6:  
                print '\t =>> OK!'
                break
            else:   print ''


print '\n\ncheck Hessian--block(2,2)'
H = [1e-6,1e-7,1e-5]
for ii in range(5):
    print 'ii={}'.format(ii)
    a = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])',\
    n=ii, degree=10), V)
    b = dl.interpolate(dl.Expression('pow(x[0], n)*pow(x[1], n)',\
    n=ii, degree=10), V)
    cg.assemble_hessianab(a, b)
    for jj in range(5):
        directa = dl.interpolate(dl.Constant('0.0'),V)
        directb = dl.interpolate(dl.Expression('1.0 + sin(n*pi*x[0])*sin(n*pi*x[1])',\
        n=jj+1, degree=10), V)
        dl.assign(directab.sub(0), directa)
        dl.assign(directab.sub(1), directb)
        Hxdir = cg.hessianab(directa, directb)
        Hxdirfun = vector2Function(Hxdir, VV)
        Hxdira, Hxdirb = Hxdirfun.split(deepcopy=True)
        normHxdira = dl.norm(Hxdira.vector())
        normHxdirb = dl.norm(Hxdirb.vector())
        print '|Hxdira|={}, |Hxdirb|={}'.format(normHxdira, normHxdirb)
        #
        for h in H:
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
            hessfddirfun = vector2Function(hessfddir, VV)
            hessfddira, hessfddirb = hessfddirfun.split(deepcopy=True)
            if normHxdira > 1e-16:
                err21 = dl.norm(Hxdira.vector() - hessfddira.vector())/normHxdira
            else:
                err21 = dl.norm(Hxdira.vector() - hessfddira.vector())
            if normHxdirb > 1e-16:
                err22 = dl.norm(Hxdirb.vector() - hessfddirb.vector())/normHxdirb
            else:
                err22 = dl.norm(Hxdirb.vector() - hessfddirb.vector())
            print '\th={}, |Hxdira|={}, |Hxdirb|={}, err21={:.2e}, err22={:.2e}'.format(\
            h, dl.norm(hessfddira.vector()), dl.norm(hessfddirb.vector()), err21, err22),
            if max(err21, err22) < 1e-6:  
                print '\t =>> OK!'
                break
            else:   print ''

