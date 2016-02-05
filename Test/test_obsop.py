"""
Check consistency of gradient and Hessian wrt cost functional for observation
operators
"""
import numpy as np
import dolfin as dl

from fenicstools.observationoperator import ObsEntireDomain
from fenicstools.miscfenics import setfct

mesh = dl.UnitSquareMesh(100, 100)
V = dl.FunctionSpace(mesh, 'Lagrange', 1)
myn = 1
m_exp = dl.Expression('sin(n*pi*x[0])*sin(n*pi*x[1])', n=myn)
m = dl.interpolate(m_exp, V)
m_in = dl.Function(V)
mv = m.vector()
shm = mv.array().shape
HH = [1e-4, 1e-5, 1e-6]

# CONTINUOUS obsop:
# Cost:
obsopcont = ObsEntireDomain({'V':V}, None)
cost_ex = (.5-np.sin(2*np.pi*myn)/(4*np.pi*myn))**2
print 'relative error on cost: {:.2e}'.format(\
np.abs(2*obsopcont.costfct(mv.array(), np.zeros(shm)) - cost_ex) / cost_ex)
print 'relative error on cost_F: {:.2e}'.format(\
np.abs(2*obsopcont.costfct_F(m, dl.Function(V)) - cost_ex) / cost_ex)

md_exp = dl.Expression('sin(n*pi*x[0])*sin(n*pi*x[1])', n=3)
md = dl.interpolate(md_exp, V)
cost = obsopcont.costfct(mv.array(), md.vector().array())
cost_F = obsopcont.costfct_F(m, md)
print 'cost={}, cost_F={}, rel_err={:.2e}'.format(cost, cost_F,\
np.abs(cost-cost_F)/np.abs(cost_F))

# Gradient:
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
        cost1 = obsopcont.costfct_F(m_in, md)

        setfct(m_in, m)
        m_in.vector().axpy(-h, dm.vector())
        cost2 = obsopcont.costfct_F(m_in, md)

        cost = obsopcont.costfct_F(m, md)

        GradFD1 = (cost1 - cost)/h
        GradFD2 = (cost1 - cost2)/(2.*h)

        Gradm = obsopcont.grad(m, md)
        Gradm_h = Gradm.inner(dm.vector())

        err1 = np.abs(GradFD1-Gradm_h)/np.abs(Gradm_h)
        err2 = np.abs(GradFD2-Gradm_h)/np.abs(Gradm_h)
        print 'h={}, GradFD1={:.5e}, GradFD2={:.5e} Gradm_h={:.5e}, err1={:.2e}, err2={:.2e}'.format(\
        h, GradFD1, GradFD2, Gradm_h, err1, err2)
        if err2 < 1e-6:  
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
            grad1 = obsopcont.grad(m_in, md)
            #
            setfct(m_in, m)
            m_in.vector().axpy(-h, dm.vector())
            grad2 = obsopcont.grad(m_in, md)
            #
            HessFD = (grad1 - grad2)/(2.*h)

            Hessmdm = obsopcont.hessian(dm.vector())

            err = (HessFD-Hessmdm).norm('l2')/Hessmdm.norm('l2')
            print 'h={}, err={}'.format(h, err)

            if err < 1e-6:  
                print 'test {}: OK!'.format(nn+1)
                success = True
                break
        if not success: failures+=1
    print '\nTest Hessian --  Summary: {} test(s) failed\n'.format(failures)
