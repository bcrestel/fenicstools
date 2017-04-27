"""
Test joint regularization with Nuclear Norm
"""

import sys
from dolfin import *
from dolfin import MPI
import numpy as np
from fenicstools.jointregularization import NuclearNormSVD2D, NuclearNormformula
from fenicstools.miscfenics import setfct

set_log_level(WARNING)

#@profile
def test_cost(eps_in, k_in):
    mesh = UnitSquareMesh(10,10)
    NN = NuclearNormSVD2D(mesh, eps=eps_in, k=k_in)
    NN2 = NuclearNormformula(mesh, {'eps':eps_in, 'k':k_in})
    test = 0

    mpicomm = mesh.mpi_comm()
    mpirank = MPI.rank(mpicomm)
    if mpirank == 0:
        print '-------------------------------------'
        print 'Test cost functional. eps={}'.format(eps_in)
        print '-------------------------------------'

    test += 1
    if mpirank == 0:    print 'Test {}: (0,0)'.format(test)
    m1 = Function(NN.V)
    m2 = Function(NN.V)
    nn = NN.costab(m1, m2)
    nn2 = NN2.costab(m1, m2)
    tt = 2.0*np.sqrt(eps_in)*k_in
    if mpirank == 0:
        print 'cost={}, err={:.2e}'.format(nn, np.abs(nn-tt)/tt)
        print 'cost2={}, err={:.2e}'.format(nn2, np.abs(nn2-tt)/tt)

    test += 1
    if mpirank == 0:    print 'Test {}: (x,y)'.format(test)
    m1 = interpolate(Expression("x[0]"), NN.V)
    m2 = interpolate(Expression("x[1]"), NN.V)
    nn = NN.costab(m1, m2)
    nn2 = NN2.costab(m1, m2)
    tt = 2.0*np.sqrt(1.0 + eps_in)*k_in
    if mpirank == 0:
        print 'cost={}, err={:.2e}'.format(nn, np.abs(nn-tt)/tt)
        print 'cost2={}, err={:.2e}'.format(nn2, np.abs(nn2-tt)/tt)
    
    test += 1
    if mpirank == 0:    print 'Test {}: (x^2, y^2)'.format(test)
    m1 = interpolate(Expression("x[0]*x[0]"), NN.V)
    m2 = interpolate(Expression("x[1]*x[1]"), NN.V)
    nn = NN.costab(m1, m2)
    nn2 = NN2.costab(m1, m2)
    tt = 2.0*k_in    # only true for eps=0
    if mpirank == 0:
        print 'cost={}, err={:.2e}'.format(nn, np.abs(nn-tt)/tt)
        print 'cost2={}, err={:.2e}'.format(nn2, np.abs(nn2-tt)/tt)
    
    test += 1
    if mpirank == 0:    print 'Test {}: (x+y, x+y)'.format(test)
    m1 = interpolate(Expression("x[0] + x[1]"), NN.V)
    m2 = interpolate(Expression("x[0] + x[1]"), NN.V)
    nn = NN.costab(m1, m2)
    nn2 = NN2.costab(m1, m2)
    tt = (np.sqrt(2.0*2.0 + eps_in) + np.sqrt(eps_in))*k_in
    if mpirank == 0:
        print 'cost={}, err={:.2e}'.format(nn, np.abs(nn-tt)/tt)
        print 'cost2={}, err={:.2e}'.format(nn2, np.abs(nn2-tt)/tt)

    test += 1
    if mpirank == 0:    print 'Test {}: (x+y, x-y)'.format(test)
    m1 = interpolate(Expression("x[0] + x[1]"), NN.V)
    m2 = interpolate(Expression("x[0] - x[1]"), NN.V)
    nn = NN.costab(m1, m2)
    nn2 = NN2.costab(m1, m2)
    tt = 2.0*np.sqrt(2.0 + eps)*k_in
    if mpirank == 0:
        print 'cost={}, err={:.2e}'.format(nn, np.abs(nn-tt)/tt)
        print 'cost2={}, err={:.2e}'.format(nn2, np.abs(nn2-tt)/tt)

    test += 1
    if mpirank == 0:    print 'Test {}: (x^2+y, x+y^2)'.format(test)
    m1 = interpolate(Expression("x[0]*x[0] + x[1]"), NN.V)
    m2 = interpolate(Expression("x[0] + x[1]*x[1]"), NN.V)
    nn = NN.costab(m1, m2)
    nn2 = NN2.costab(m1, m2)
    if mpirank == 0:
        print 'cost={}, cost2={}, diff={:.2e}'.format(nn, nn2, np.abs(nn-nn2)/np.abs(nn2))

    test += 1
    if mpirank == 0:    print 'Test {}: (x^2*y, x*y^2)'.format(test)
    m1 = interpolate(Expression("x[0]*x[0]*x[1]"), NN.V)
    m2 = interpolate(Expression("x[0]*x[1]*x[1]"), NN.V)
    nn = NN.costab(m1, m2)
    nn2 = NN2.costab(m1, m2)
    if mpirank == 0:
        print 'cost={}, cost2={}, diff={:.2e}'.format(nn, nn2, np.abs(nn-nn2)/np.abs(nn2))

    test += 1
    if mpirank == 0:    print 'Test {}: coincide'.format(test)
    m1 = interpolate(Expression('log(10 - ' + \
    '(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.4) * (' + \
    '4*(x[0]<=0.5) + 8*(x[0]>0.5) ))'), NN.V)
    m2 = interpolate(Expression('log(10 - ' + \
    '(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.4) * (' + \
    '8*(x[0]<=0.5) + 4*(x[0]>0.5) ))'), NN.V)
    nn = NN.costab(m1, m2)
    nn2 = NN2.costab(m1, m2)
    if mpirank == 0:
        print 'cost={}, cost2={}, diff={:.2e}'.format(nn, nn2, np.abs(nn-nn2)/np.abs(nn2))

    test += 1
    if mpirank == 0:    print 'Test {}: coincide2'.format(test)
    m1 = interpolate(Expression('log(10 - ' + \
    '(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.4) * 8 )'), NN.V)
    m2 = interpolate(Expression('log(10 - ' + \
    '(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.4) * (' + \
    '8*(x[0]<=0.5) + 4*(x[0]>0.5) ))'), NN.V)
    nn = NN.costab(m1, m2)
    nn2 = NN2.costab(m1, m2)
    if mpirank == 0:
        print 'cost={}, cost2={}, diff={:.2e}'.format(nn, nn2, np.abs(nn-nn2)/np.abs(nn2))





def test_grad(eps_in, k_in):
    mesh = UnitSquareMesh(10,10)
    NN = NuclearNormSVD2D(mesh, eps=eps_in, k=k_in)
    NN2 = NuclearNormformula(mesh, {'eps':eps_in, 'k':k_in})
    direc12 = Function(NN.VV)
    m1h, m2h = Function(NN.V), Function(NN.V)
    H = [1e-4, 1e-5, 1e-6, 1e-7]

    mpicomm = mesh.mpi_comm()
    mpirank = MPI.rank(mpicomm)
    mpisize = MPI.size(mpicomm)
    if mpirank == 0:
        print '-------------------------------------'
        print 'Test gradient. eps={}'.format(eps)
        print '-------------------------------------'

    if mpirank == 0:    print 'Test 1'
    m1 = Function(NN.V)
    m2 = Function(NN.V)
    grad = NN.gradab(m1, m2)
    grad2 = NN2.gradab(m1, m2)
    normgrad = norm(grad)
    normgrad2 = norm(grad2)
    if mpirank == 0:
        print '|grad|={}, err={}'.format(normgrad, np.abs(normgrad))
        print '|grad2|={}, err={}'.format(normgrad2, np.abs(normgrad2))

    M1 = [interpolate(Expression("x[0] + x[1]"), NN.V),
    interpolate(Expression("x[0] * x[1]"), NN.V),
    interpolate(Expression("x[0]*x[0] + x[1]"), NN.V),
    interpolate(Expression('log(10 - ' + \
    '(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.4) * (' + \
    '4*(x[0]<=0.5) + 8*(x[0]>0.5) ))'), NN.V),
    interpolate(Expression('log(10 - ' + \
    '(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.4) * 8 )'), NN.V)]
    M2 = [interpolate(Expression("x[0] + x[1]"), NN.V),
    interpolate(Expression("1.0 + cos(x[0])"), NN.V),
    interpolate(Expression("x[1]*x[1] + x[0]"), NN.V),
    interpolate(Expression('log(10 - ' + \
    '(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.4) * (' + \
    '8*(x[0]<=0.5) + 4*(x[0]>0.5) ))'), NN.V),
    interpolate(Expression('log(10 - ' + \
    '(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.4) * (' + \
    '8*(x[0]<=0.5) + 4*(x[0]>0.5) ))'), NN.V)]
    tt = 2
    for m1, m2 in zip(M1, M2):
        if mpirank == 0:    print '\nTest {}'.format(tt)
        tt += 1
        grad = NN.gradab(m1, m2)
        grad2 = NN2.gradab(m1, m2)
        for nn in range(5):
            if mpirank == 0:    print '--direction {}'.format(nn)
            direc1 = interpolate(Expression('1 + sin(n*pi*x[0])*sin(n*pi*x[1])',\
            n=nn), NN.V)
            direc2 = interpolate(Expression('1 + cos(n*pi*x[0])*cos(n*pi*x[1])',\
            n=nn), NN.V)
            assign(direc12.sub(0), direc1)
            assign(direc12.sub(1), direc2)
            direcderiv = grad.inner(direc12.vector())
            direcderiv2 = grad2.inner(direc12.vector())
            if mpirank == 0:    
                print 'grad={}, '.format(direcderiv)
                print 'grad2={}, diff={:.2e} '.format(direcderiv2, \
                np.abs(direcderiv-direcderiv2)/np.abs(direcderiv))

            for hh in H:
                setfct(m1h, m1)
                m1h.vector().axpy(hh, direc1.vector())
                setfct(m2h, m2)
                m2h.vector().axpy(hh, direc2.vector())
                cost1 = NN.costab(m1h, m2h)

                setfct(m1h, m1)
                m1h.vector().axpy(-hh, direc1.vector())
                setfct(m2h, m2)
                m2h.vector().axpy(-hh, direc2.vector())
                cost2 = NN.costab(m1h, m2h)

                FDdirecderiv = (cost1-cost2)/(2.0*hh)
                if np.abs(direcderiv) > 1e-16:
                    err = np.abs(direcderiv-FDdirecderiv)/np.abs(direcderiv)
                else:
                    err = np.abs(direcderiv-FDdirecderiv)
                if mpirank == 0:
                    print '\th={}, fd={}, err={:.2e}'.format(hh, FDdirecderiv, err),
                if err < 1e-6:
                    if mpirank == 0:    print '\t =>> OK!'
                    break
                else:   
                    if mpirank == 0:    print ''






def test_hessian(eps_in):
    mesh = UnitSquareMesh(10,10)
    NN = NuclearNormSVD2D(mesh, eps=eps_in)
    NN2 = NuclearNormformula(mesh, {'eps':eps_in})
    direc12 = Function(NN.VV)
    m1h, m2h = Function(NN.V), Function(NN.V)
    H = [1e-4, 1e-5, 1e-6, 1e-7]

    mpicomm = mesh.mpi_comm()
    mpirank = MPI.rank(mpicomm)
    mpisize = MPI.size(mpicomm)
    if mpirank == 0:
        print '-------------------------------------'
        print 'Test Hessian. eps={}'.format(eps)
        print '-------------------------------------'

    M1 = [interpolate(Expression("x[0] + x[1]"), NN.V),
    interpolate(Expression("x[0] * x[1]"), NN.V),
    interpolate(Expression("x[0]*x[0] + x[1]"), NN.V),
    interpolate(Expression('log(10 - ' + \
    '(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.4) * (' + \
    '4*(x[0]<=0.5) + 8*(x[0]>0.5) ))'), NN.V),
    interpolate(Expression('log(10 - ' + \
    '(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.4) * 8 )'), NN.V)]
    M2 = [interpolate(Expression("x[0] + x[1]"), NN.V),
    interpolate(Expression("1.0 + cos(x[0])"), NN.V),
    interpolate(Expression("x[1]*x[1] + x[0]"), NN.V),
    interpolate(Expression('log(10 - ' + \
    '(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.4) * (' + \
    '8*(x[0]<=0.5) + 4*(x[0]>0.5) ))'), NN.V),
    interpolate(Expression('log(10 - ' + \
    '(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.4) * (' + \
    '8*(x[0]<=0.5) + 4*(x[0]>0.5) ))'), NN.V)]
    tt = 1
    for m1, m2 in zip(M1, M2):
        if mpirank == 0:    print '\nTest {}'.format(tt)
        tt += 1
        NN2.assemble_hessianab(m1, m2)
        for nn in range(5):
            if mpirank == 0:    print '--direction {}'.format(nn)
            direc1 = interpolate(Expression('1 + sin(n*pi*x[0])*sin(n*pi*x[1])',\
            n=nn), NN.V)
            direc2 = interpolate(Expression('1 + cos(n*pi*x[0])*cos(n*pi*x[1])',\
            n=nn), NN.V)
            assign(direc12.sub(0), direc1)
            assign(direc12.sub(1), direc2)
            Hv = NN2.hessianab(direc1.vector(), direc2.vector())
            HMv = (NN2.H + NN2.sMass)*direc12.vector()
            nHv = norm(Hv)
            nHMv = norm(HMv)
            ndiff = norm(Hv-HMv)/nHv
            if mpirank == 0:    
                print '|Hv|={}, |HMV|={}, |diff|={:.2e}'.format(nHv, nHMv, ndiff)

            for hh in H:
                setfct(m1h, m1)
                m1h.vector().axpy(hh, direc1.vector())
                setfct(m2h, m2)
                m2h.vector().axpy(hh, direc2.vector())
                grad1 = NN.gradab(m1h, m2h)

                setfct(m1h, m1)
                m1h.vector().axpy(-hh, direc1.vector())
                setfct(m2h, m2)
                m2h.vector().axpy(-hh, direc2.vector())
                grad2 = NN.gradab(m1h, m2h)

                FD = (grad1-grad2)/(2.0*hh)
                nFD = norm(FD)
                if nHv > 1e-16:
                    err = norm(FD-Hv)/nHv
                else:
                    err = norm(FD-Hv)
                if mpirank == 0:
                    print '\th={}, |FD|={}, err={:.2e}'.format(hh, nFD, err),
                if err < 1e-6:
                    if mpirank == 0:    print '\t =>> OK!'
                    break
                else:   
                    if mpirank == 0:    print ''









if __name__ == "__main__":
    try:
        case_nb = int(sys.argv[1])
    except:
        print 'Error: Wrong argument\nUsage: {} <case_nb> (eps k)'.format(\
        sys.argv[0])
        sys.exit(1)

    try:
        eps = float(sys.argv[2])
        k = float(sys.argv[3])
    except:
        eps = 1e-12
        k = 1.0

    if case_nb == 0:
        test_cost(eps, k)
    elif case_nb == 1:
        test_grad(eps, k)
    elif case_nb == 2:
        test_hessian(eps)
