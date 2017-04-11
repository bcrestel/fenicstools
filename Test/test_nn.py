"""
Test joint regularization with Nuclear Norm
"""

import sys
from dolfin import *
import numpy as np
from fenicstools.jointregularization import NuclearNormSVD2D, NuclearNormformula

def test_cost():
    mesh = UnitSquareMesh(10,10)
    NN = NuclearNormSVD2D(mesh, eps=0.0)
    NN2 = NuclearNormformula(mesh, eps=1e-16)

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



if __name__ == "__main__":
    try:
        case_nb = int(sys.argv[1])
    except:
        print 'Error: Wrong argument\nUsage: {} <case_nb>'.format(sys.argv[0])
        sys.exit(1)

    if case_nb == 0:
        print 'test cost functional'
        test_cost()
