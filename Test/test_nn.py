"""
Test joint regularization with Nuclear Norm
"""

import sys
from dolfin import *
import numpy as np
from fenicstools.jointregularization import NuclearNormSVD2D, NuclearNormformula
from fenicstools.miscfenics import setfct

set_log_level(WARNING)

def test_cost(eps_in):
    mesh = UnitSquareMesh(10,10)
    NN = NuclearNormSVD2D(mesh, eps=eps_in)
    NN2 = NuclearNormformula(mesh, eps=eps_in)

    print 'Test 1'
    m1 = Function(NN.V)
    m2 = Function(NN.V)
    nn = NN.costab(m1, m2)
    nn2 = NN2.costab(m1, m2)
    print 'cost={}, err={}'.format(nn, np.abs(nn))
    print 'cost2={}, err={}'.format(nn2, np.abs(nn2))

    print '\nTest 2'
    m1 = interpolate(Expression("x[0]"), NN.V)
    m2 = interpolate(Expression("x[1]"), NN.V)
    nn = NN.costab(m1, m2)
    nn2 = NN2.costab(m1, m2)
    tt = 2.0
    print 'cost={}, err={}'.format(nn, np.abs(nn-tt)/tt)
    print 'cost2={}, err={}'.format(nn2, np.abs(nn2-tt)/tt)
    
    print '\nTest 3'
    m1 = interpolate(Expression("x[0]*x[0]"), NN.V)
    m2 = interpolate(Expression("x[1]*x[1]"), NN.V)
    nn = NN.costab(m1, m2)
    nn2 = NN2.costab(m1, m2)
    tt = 2.0
    print 'cost={}, err={}'.format(nn, np.abs(nn-tt)/tt)
    print 'cost2={}, err={}'.format(nn2, np.abs(nn2-tt)/tt)
    
    print '\nTest 4'
    m1 = interpolate(Expression("x[0] + x[1]"), NN.V)
    m2 = interpolate(Expression("x[0] + x[1]"), NN.V)
    nn = NN.costab(m1, m2)
    nn2 = NN2.costab(m1, m2)
    tt = 2.0
    print 'cost={}, err={}'.format(nn, np.abs(nn-tt)/tt)
    print 'cost2={}, err={}'.format(nn2, np.abs(nn2-tt)/tt)

    print '\nTest 5'
    m1 = interpolate(Expression("x[0] + x[1]"), NN.V)
    m2 = interpolate(Expression("x[0] - x[1]"), NN.V)
    nn = NN.costab(m1, m2)
    nn2 = NN2.costab(m1, m2)
    tt = 2.0*np.sqrt(2.0)
    print 'cost={}, err={}'.format(nn, np.abs(nn-tt)/tt)
    print 'cost2={}, err={}'.format(nn2, np.abs(nn2-tt)/tt)

    print '\nTest 6'
    m1 = interpolate(Expression("x[0]*x[0] + x[1]"), NN.V)
    m2 = interpolate(Expression("x[0] + x[1]*x[1]"), NN.V)
    nn = NN.costab(m1, m2)
    nn2 = NN2.costab(m1, m2)
    print 'cost={}, cost2={}, err={}'.format(nn, nn2, np.abs(nn-nn2)/np.abs(nn2))

    print '\nTest 7'
    m1 = interpolate(Expression("x[0]*x[0]*x[1]"), NN.V)
    m2 = interpolate(Expression("x[0]*x[1]*x[1]"), NN.V)
    nn = NN.costab(m1, m2)
    nn2 = NN2.costab(m1, m2)
    print 'cost={}, cost2={}, err={}'.format(nn, nn2, np.abs(nn-nn2)/np.abs(nn2))


def test_grad(eps_in):
    mesh = UnitSquareMesh(10,10)
    NN = NuclearNormSVD2D(mesh, eps=eps_in)
    direc12 = Function(NN.VV)
    m1h, m2h = Function(NN.V), Function(NN.V)
    H = [1e-4, 1e-5, 1e-6, 1e-7]

    print 'Test 1'
    m1 = Function(NN.V)
    m2 = Function(NN.V)
    grad = NN.gradab(m1, m2)
    print '|grad|={}, err={}'.format(norm(grad), np.abs(norm(grad)))

    M1 = [interpolate(Expression("x[0] + x[1]"), NN.V),
    interpolate(Expression("x[0] * x[1]"), NN.V),
    interpolate(Expression("x[0]*x[0] + x[1]"), NN.V)]
    M2 = [interpolate(Expression("x[0] + x[1]"), NN.V),
    interpolate(Expression("1.0 + cos(x[0])"), NN.V),
    interpolate(Expression("x[1]*x[1] + x[0]"), NN.V)]
    tt = 2
    for m1, m2 in zip(M1, M2):
        print '\nTest {}'.format(tt)
        tt += 1
        grad = NN.gradab(m1, m2)
        for nn in range(5):
            print '--direction {}'.format(nn)
            direc1 = interpolate(Expression('1 + sin(n*pi*x[0])*sin(n*pi*x[1])',\
            n=nn), NN.V)
            direc2 = interpolate(Expression('1 + cos(n*pi*x[0])*cos(n*pi*x[1])',\
            n=nn), NN.V)
            assign(direc12.sub(0), direc1)
            assign(direc12.sub(1), direc2)
            direcderiv = grad.inner(direc12.vector())
            print 'grad={}, '.format(direcderiv)

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
                print '\th={}, fd={}, err={:.2e}'.format(hh, FDdirecderiv, err),
                if err < 1e-6:
                    print '\t =>> OK!'
                    break
                else:   print ''


if __name__ == "__main__":
    try:
        case_nb = int(sys.argv[1])
    except:
        print 'Error: Wrong argument\nUsage: {} <case_nb> (eps)'.format(\
        sys.argv[0])
        sys.exit(1)

    try:
        eps = float(sys.argv[2])
    except:
        eps = 1e-12

    if case_nb == 0:
        print '-------------------------------------'
        print 'Test cost functional. eps={}'.format(eps)
        print '-------------------------------------'
        test_cost(eps)
    elif case_nb == 1:
        print '-------------------------------------'
        print 'Test gradient. eps={}'.format(eps)
        print '-------------------------------------'
        test_grad(eps)
