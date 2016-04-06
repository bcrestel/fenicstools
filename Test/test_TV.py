"""
Test for the TV regularization class.
- All tests passed except along the most sinuating direction (last one)
- tested with k=1.0 and k=exp
- test with eps=0.0001, 0.001, 0.01 and eps=0.1.
- test with V order = 1, 2, 3. Works but some derivatives are too close to zero
  to check (Still okay!)
"""
#TODO: think about analytical derivatives

import dolfin as dl
import numpy as np

from fenicstools.regularization import TV
from fenicstools.miscfenics import setfct

HH = [1e-4, 1e-5, 1e-6]
#HH = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

mesh = dl.UnitSquareMesh(100, 100)
V = dl.FunctionSpace(mesh, 'Lagrange', 1)
k_exp = dl.Expression('1.0 + exp(x[0]*x[1])')
k = dl.interpolate(k_exp, V)
#k = dl.Constant(1.0)
m_in = dl.Function(V)
TV1 = TV({'Vm':V, 'eps':dl.Constant(0.0001), 'k':k, 'GNhessian':False})

print 'Test 1: Smooth medium'

# Verification point, i.e., point at which gradient and Hessian are checked
m_exp = dl.Expression('sin(n*pi*x[0])*sin(n*pi*x[1])', n=1)
m = dl.interpolate(m_exp, V)

print '\nGradient:'
failures = 0
for nn in range(8):
    print '\ttest ' + str(nn+1)
    dm_exp = dl.Expression('sin(n*pi*x[0])*sin(n*pi*x[1])', n=nn+1)
    dm = dl.interpolate(dm_exp, V)

    for h in HH:
        success = False
        setfct(m_in, m)
        m_in.vector().axpy(h, dm.vector())
        cost1 = TV1.cost(m_in)

        setfct(m_in, m)
        m_in.vector().axpy(-h, dm.vector())
        cost2 = TV1.cost(m_in)

        cost = TV1.cost(m)

        GradFD = (cost1 - cost2)/(2.*h)
        #GradFD = (cost1 - cost)/h

        Gradm = TV1.grad(m) 
        Gradm_h = Gradm.inner(dm.vector())

        err = np.abs(GradFD-Gradm_h)/np.abs(Gradm_h)
        print 'h={}, GradFD={}, Gradm_h={}, err={}'.format(\
        h, GradFD, Gradm_h, err)
        if err < 1e-6:  
            print 'test {}: OK!'.format(nn+1)
            success = True
            break
    if not success: failures+=1
print '\nTest gradient -- Summary: {} test(s) failed'.format(failures)

if failures < 5:
    print '\n\nHessian:'
    failures = 0
    for nn in range(8):
        print '\ttest ' + str(nn+1)
        dm_exp = dl.Expression('sin(n*pi*x[0])*sin(n*pi*x[1])', n=nn+1)
        dm = dl.interpolate(dm_exp, V)

        for h in HH:
            success = False
            setfct(m_in, m)
            m_in.vector().axpy(h, dm.vector())
            grad1 = TV1.grad(m_in)
            #
            setfct(m_in, m)
            m_in.vector().axpy(-h, dm.vector())
            grad2 = TV1.grad(m_in)
            #
            HessFD = (grad1 - grad2)/(2.*h)

            TV1.assemble_hessian(m)
            Hessmdm = TV1.hessian(dm.vector())

            err = (HessFD-Hessmdm).norm('l2')/Hessmdm.norm('l2')
            print 'h={}, err={}'.format(h, err)

            if err < 1e-6:  
                print 'test {}: OK!'.format(nn+1)
                success = True
                break
        if not success: failures+=1
    print '\nTest Hessian --  Summary: {} test(s) failed\n'.format(failures)
