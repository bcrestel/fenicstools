"""
Test for the TV regularization class.
- All tests passed except along the most sinuating direction (last one)
- tested with k=1.0 and k=exp
- test with eps=0.0001, 0.001, 0.01 and eps=0.1.
- test with V order = 1, 2, 3. Works but some derivatives are too close to zero
  to check (Still okay!)
"""

import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt

from fenicstools.regularization import TV, TVPD
from fenicstools.miscfenics import setfct



def gradhesscheck():
    sndorder = True

    #HH = [1e-4]
    #HH = [1e-4, 1e-5, 1e-6]
    HH = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]


    mesh = dl.UnitSquareMesh(100, 100)
    V = dl.FunctionSpace(mesh, 'Lagrange', 1)
    #k_exp = dl.Expression('1.0 + exp(x[0]*x[1])')
    #k = dl.interpolate(k_exp, V)
    #k = dl.Constant(1.0)
    m_in = dl.Function(V)
    TV1 = TV({'Vm':V, 'eps':1e-4, 'k':1e-2, 'GNhessian':False})
    TV2 = TVPD({'Vm':V, 'eps':1e-4, 'k':1e-2, 'exact':True})

    #M_EXP = [dl.Expression('1.0'),\
    #dl.Expression('1.0 + (x[0]>=0.2)*(x[0]<=0.8)*(x[1]>=0.2)*(x[1]<=0.8)'), \
    #dl.Expression('1.0 + (x[0]>=0.2)*(x[0]<=0.8)*(x[1]>=0.2)*(x[1]<=0.8) ' + \
    #'+ (x[0]>=0.2)*(x[0]<=0.4)*(x[1]>=0.2)*(x[1]<=0.4)')]
    M_EXP = [dl.Expression('1.0 + (x[0]>=0.2)*(x[0]<=0.8)*(x[1]>=0.2)*(x[1]<=0.8)')]

    for ii, m_exp in enumerate(M_EXP):
        # Verification point, i.e., point at which gradient and Hessian are checked
        print 'Verification point', str(ii)
        #m_exp = dl.Expression('sin(n*pi*x[0])*sin(n*pi*x[1])', n=ii)
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
                cost12 = TV2.cost(m_in)
                print 'cost1={}, cost12={}, err={}'.format(cost1, cost12, \
                np.abs(cost1-cost12)/np.abs(cost1))

                if sndorder:
                    setfct(m_in, m)
                    m_in.vector().axpy(-h, dm.vector())
                    cost2 = TV1.cost(m_in)

                    GradFD = (cost1 - cost2)/(2.*h)
                else:
                    cost = TV1.cost(m)

                    GradFD = (cost1 - cost)/h


                Grad1m = TV1.grad(m) 
                Grad1m_h = Grad1m.inner(dm.vector())
                Grad2m = TV2.grad(m) 
                Grad2m_h = Grad2m.inner(dm.vector())

                if np.abs(Grad1m_h) > 1e-16:
                    err1 = np.abs(GradFD-Grad1m_h)/np.abs(Grad1m_h)
                    err2 = np.abs(Grad1m_h-Grad2m_h)/np.abs(Grad1m_h)
                else:
                    err1 = np.abs(GradFD-Grad1m_h)
                    err2 = np.abs(Grad1m_h-Grad2m_h)
                print 'h={}, GradFD={}, Grad1m_h={}, err1={:.2e}'.format(\
                h, GradFD, Grad1m_h, err1)
                print 'Grad2m_h={}, err12={}'.format(Grad2m_h, err2)
                if err1 < 1e-6:  
                    print 'test {}: OK!'.format(nn+1)
                    success = True
                    break
            if not success: failures+=1
        print '\nTest gradient -- Summary: {} test(s) failed'.format(failures)

        #if failures < 5:
        if True:
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

                    if sndorder:
                        setfct(m_in, m)
                        m_in.vector().axpy(-h, dm.vector())
                        grad2 = TV1.grad(m_in)
                    
                        HessFD = (grad1 - grad2)/(2.*h)
                    else:
                        grad = TV1.grad(m)

                        HessFD = (grad1 - grad)/h

                    TV1.assemble_hessian(m)
                    Hess1mdm = TV1.hessian(dm.vector())
                    TV2.assemble_hessian(m)
                    Hess2mdm = TV2.hessian(dm.vector())

                    err1 = (HessFD-Hess1mdm).norm('l2')/Hess1mdm.norm('l2')
                    err2 = (Hess1mdm-Hess2mdm).norm('l2')/Hess2mdm.norm('l2')
                    print 'h={}, err1={}, err12={}'.format(h, err1, err2)

                    if err1 < 1e-6:  
                        print 'test {}: OK!'.format(nn+1)
                        success = True
                        break
                if not success: failures+=1
            print '\nTest Hessian --  Summary: {} test(s) failed\n'.format(failures)


def checkprecond():
    """ Check sMass in preconditioner is negligible for realistic medium """
    mesh = dl.UnitSquareMesh(40, 40)
    V = dl.FunctionSpace(mesh, 'Lagrange', 1)
    regTV = TV({'Vm':V, 'eps':1e-4, 'k':1.0, 'GNhessian':False})

#    print 'Eigenvalues for M'
#    sM = dl.as_backend_type(regTV.sMass)
#    compute_eig(sM, 'eigsM.txt')
#
#    print 'Eigenvalues for H1'
#    m = dl.interpolate(dl.Expression(\
#    'pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.2'), V)
#    regTV.assemble_hessian(m)
#    H = dl.as_backend_type(regTV.H)
#    compute_eig(H, 'eigH1.txt')
#
#    print 'Eigenvalues for H2'
#    m = dl.interpolate(dl.Expression(\
#    '0.1*(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.2)'), V)
#    regTV.assemble_hessian(m)
#    H = dl.as_backend_type(regTV.H)
#    compute_eig(H, 'eigH2.txt')
#
#    print 'Eigenvalues for Hcst'
#    m = dl.interpolate(dl.Expression("1.0"), V)
#    regTV.assemble_hessian(m)
#    H = dl.as_backend_type(regTV.H)
#    compute_eig(H, 'eigHcst.txt')
#
#    print 'Eigenvalues for Hsin'
#    m = dl.interpolate(dl.Expression(\
#    'sin(pi*x[0])*sin(pi*x[1])'), V)
#    regTV.assemble_hessian(m)
#    H = dl.as_backend_type(regTV.H)
#    compute_eig(H, 'eigHsin.txt')

    print 'Eigenvalues for Hsin2'
    m = dl.interpolate(dl.Expression(\
    '0.1*sin(pi*x[0])*sin(pi*x[1])'), V)
    regTV.assemble_hessian(m)
    H = dl.as_backend_type(regTV.H)
    compute_eig(H, 'eigHsin2.txt')

    print 'Eigenvalues for Hsin5'
    m = dl.interpolate(dl.Expression(\
    'sin(5*pi*x[0])*sin(5*pi*x[1])'), V)
    regTV.assemble_hessian(m)
    H = dl.as_backend_type(regTV.H)
    compute_eig(H, 'eigHsin5.txt')


def compute_eig(M, filename):
    eigsolver = dl.SLEPcEigenSolver(M)
    eigsolver.solve()
    eig = []
    for ii in range(eigsolver.get_number_converged()):
        eig.append(eigsolver.get_eigenvalue(ii)[0])
    eig.sort()
    np.savetxt(filename, np.array(eig))


if __name__ == "__main__":
    #gradhesscheck()
    checkprecond()
