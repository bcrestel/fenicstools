import sys
import numpy as np

from dolfin import sqrt, inner, nabla_grad, grad, dx, \
Function, TestFunction, TrialFunction, Vector, assemble, solve, \
Constant, plot, interactive, assign, FunctionSpace, \
PETScKrylovSolver, PETScLUSolver, mpi_comm_world, PETScMatrix
from miscfenics import isVector, setfct
from linalg.miscroutines import get_diagonal, setupPETScmatrix
from hippylib.linalg import MatMatMult, Transpose

import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc


#----------------------------------------------------------------------
#----------------------------------------------------------------------
class TV():
    """
    Define Total Variation regularization
    """

    def __init__(self, parameters=[]):
        """ parameters should be:
            - k(x) = factor inside TV
            - eps = regularization parameter
        |f|_TV = int k(x) sqrt{|grad f|^2 + eps} dx """

        self.parameters = {'k':1.0, 'eps':1e-2, 'GNhessian':False} # default parameters

        assert parameters.has_key('Vm')
        self.parameters.update(parameters)
        GN = self.parameters['GNhessian']
        self.Vm = self.parameters['Vm']
        eps = self.parameters['eps']
        k = self.parameters['k']

        self.m = Function(self.Vm)
        test, trial = TestFunction(self.Vm), TrialFunction(self.Vm)
        factM = 1e-2*k
        M = assemble(inner(test, trial)*dx)
        self.sMass = M*factM

        self.Msolver = PETScLUSolver("petsc")
        self.Msolver.parameters['reuse_factorization'] = True
        self.Msolver.parameters['symmetric'] = True
        self.Msolver.set_operator(M)

        self.fTV = inner(nabla_grad(self.m), nabla_grad(self.m)) + Constant(eps)
        self.kovsq = Constant(k) / sqrt(self.fTV)

        self.wkformcost = Constant(k) * sqrt(self.fTV) * dx

        self.wkformgrad = self.kovsq*inner(nabla_grad(self.m), nabla_grad(test))*dx

        self.wkformGNhess = self.kovsq*inner(nabla_grad(trial), nabla_grad(test))*dx
        self.wkformFhess = self.kovsq*( \
        inner(nabla_grad(trial), nabla_grad(test)) - \
        inner(nabla_grad(self.m), nabla_grad(test))*\
        inner(nabla_grad(trial), nabla_grad(self.m))/self.fTV )*dx
        if GN: 
            self.wkformhess = self.wkformGNhess
            print 'TV regularization -- GN Hessian'
        else:   
            self.wkformhess = self.wkformFhess
            print 'TV regularization -- full Hessian'

        try:
            solver = PETScKrylovSolver('cg', 'ml_amg')
            self.amgprecond = 'ml_amg'
        except:
            self.amgprecond = 'petsc_amg'


    def isTV(self): return True
    def isPD(self): return False


    def cost(self, m_in):
        """ returns the cost functional for self.m=m_in """
        setfct(self.m, m_in)
        return assemble(self.wkformcost)

    def costvect(self, m_in):   return self.cost(m_in)


    def grad(self, m_in):
        """ returns the gradient (in vector format) evaluated at self.m = m_in """
        setfct(self.m, m_in)
        return assemble(self.wkformgrad)

    def gradvect(self, m_in):   return self.grad(m_in)


    def assemble_hessian(self, m_in):
        """ Assemble the Hessian of TV at m_in """
        setfct(self.m, m_in)
        self.H = assemble(self.wkformhess)
        self.precond = self.H + self.sMass

    def assemble_GNhessian(self, m_in):
        """ Assemble the Gauss-Newton Hessian at m_in 
        Not used anymore (wkformhess selects GN Hessian if needed)
        Left here for back-compatibility """
        setfct(self.m, m_in)
        self.H = assemble(self.wkformGNhess)
        self.precond = self.H + self.sMass

    def hessian(self, mhat):
        """ returns the Hessian applied along a direction mhat """
        isVector(mhat)
        return self.H * mhat


    def getprecond(self):
        """ Precondition by inverting the TV Hessian """

        solver = PETScKrylovSolver('cg', self.amgprecond)
        solver.parameters["maximum_iterations"] = 2000
        solver.parameters["relative_tolerance"] = 1e-24
        solver.parameters["absolute_tolerance"] = 1e-24
        solver.parameters["error_on_nonconvergence"] = True 
        solver.parameters["nonzero_initial_guess"] = False 

        # used to compare iterative application of preconditioner 
        # with exact application of preconditioner:
        #solver = PETScLUSolver("petsc")
        #solver.parameters['symmetric'] = True
        #solver.parameters['reuse_factorization'] = True

        solver.set_operator(self.precond)
        return solver


    def init_vector(self, u, dim):
        self.sMass.init_vector(u, dim)



#----------------------------------------------------------------------
#----------------------------------------------------------------------
class TVPD():
    """ Total variation using primal-dual Newton """

    def __init__(self, parameters, mpicomm=PETSc.COMM_WORLD):
        """ parameters must have key 'Vm' for parameter function space,
        key 'exact' run exact TV, w/o primal-dual; used to check w/ FD """
        self.parameters = {'k':1.0, 'eps':1e-2, 'exact':False}
        assert parameters.has_key('Vm')
        self.parameters.update(parameters)
        self.Vm = self.parameters['Vm']
        k = self.parameters['k']
        eps = self.parameters['eps']
        exact = self.parameters['exact']

        self.m = Function(self.Vm)
        testm = TestFunction(self.Vm)
        trialm = TrialFunction(self.Vm)

        # WARNING: should not be changed.
        # As it is, code only works with DG0
        if self.parameters.has_key('Vw'):
            Vw = self.parameters['Vw']
        else:
            Vw = FunctionSpace(self.Vm.mesh(), 'DG', 0)
        self.wx = Function(Vw)
        self.wxrs = Function(Vw)   # re-scaled dual variable
        self.wxhat = Function(Vw)
        self.gwx = Function(Vw)
        self.wy = Function(Vw)
        self.wyrs = Function(Vw)   # re-scaled dual variable
        self.wyhat = Function(Vw)
        self.gwy = Function(Vw)
        testw = TestFunction(Vw)
        trialw = TrialFunction(Vw)

        normm = inner(nabla_grad(self.m), nabla_grad(self.m))
        TVnormsq = normm + Constant(eps)
        TVnorm = sqrt(TVnormsq)
        self.wkformcost = Constant(k)*TVnorm*dx
        if exact:
            sys.exit(1)
#            self.w = nabla_grad(self.m)/TVnorm # full Hessian
#            self.Htvw = inner(Constant(k) * nabla_grad(testm), self.w) * dx

        self.misfitwx = inner(testw, self.wx*TVnorm - self.m.dx(0))*dx
        self.misfitwy = inner(testw, self.wy*TVnorm - self.m.dx(1))*dx

        self.Htvx = assemble(inner(Constant(k)*testm.dx(0), trialw)*dx)
        self.Htvy = assemble(inner(Constant(k)*testm.dx(1), trialw)*dx)

        self.massw = inner(TVnorm*testw, trialw)*dx

        invMwMat, VDM, VDM = setupPETScmatrix(Vw, Vw, 'aij', mpicomm)
        for ii in VDM.dofs():
            invMwMat[ii, ii] = 1.0
        invMwMat.assemblyBegin()
        invMwMat.assemblyEnd()
        self.invMwMat = PETScMatrix(invMwMat)
        self.invMwd = Vector()
        self.invMwMat.init_vector(self.invMwd, 0)

        self.wkformAx = inner(testw, trialm.dx(0) - \
        self.wx * inner(nabla_grad(self.m), nabla_grad(trialm)) / TVnorm) * dx
        self.wkformAxrs = inner(testw, trialm.dx(0) - \
        self.wxrs * inner(nabla_grad(self.m), nabla_grad(trialm)) / TVnorm) * dx
        self.wkformAy = inner(testw, trialm.dx(1) - \
        self.wy * inner(nabla_grad(self.m), nabla_grad(trialm)) / TVnorm) * dx
        self.wkformAyrs = inner(testw, trialm.dx(1) - \
        self.wyrs * inner(nabla_grad(self.m), nabla_grad(trialm)) / TVnorm) * dx

        factM = 1e-2*k
        M = assemble(inner(testm, trialm)*dx)
        self.sMass = M*factM

        self.Msolver = PETScLUSolver("petsc")
        self.Msolver.parameters['reuse_factorization'] = True
        self.Msolver.parameters['symmetric'] = True
        self.Msolver.set_operator(M)

        print 'TV regularization -- primal-dual method'
        try:
            solver = PETScKrylovSolver('cg', 'ml_amg')
            self.amgprecond = 'ml_amg'
        except:
            self.amgprecond = 'petsc_amg'


    def isTV(self): return True
    def isPD(self): return True

    
    def cost(self, m):
        """ evaluate the cost functional at m """
        setfct(self.m, m)
        return assemble(self.wkformcost)

    def costvect(self, m_in):   return self.cost(m_in)


    def _assemble_invMw(self):
        """ Assemble inverse of matrix Mw,
        weighted mass matrix in dual space """
        # WARNING: only works if Mw is diagonal (e.g, DG0)
        Mw = assemble(self.massw)
        Mwd = get_diagonal(Mw)
        self.invMwd[:] = 1./Mwd.array()
        self.invMwMat.set_diagonal(self.invMwd)


    def grad(self, m):
        """ compute the gradient at m """
        setfct(self.m, m)
        self._assemble_invMw()

        self.gwx.vector().zero()
        self.gwx.vector().axpy(1.0, assemble(self.misfitwx))
        normgwx = np.linalg.norm(self.gwx.vector().array())

        self.gwy.vector().zero()
        self.gwy.vector().axpy(1.0, assemble(self.misfitwy))
        normgwy = np.linalg.norm(self.gwy.vector().array())

        print '|gw|={}'.format(np.sqrt(normgwx**2 + normgwy**2))

        return self.Htvx*(self.wx.vector() - self.invMwd*self.gwx.vector()) \
        + self.Htvy*(self.wy.vector() - self.invMwd*self.gwy.vector())
        #return assemble(self.Htvw) - self.Htv*(self.invMwd*self.gw.vector())

    def gradvect(self, m_in):   return self.grad(m_in)


    def assemble_hessian(self, m):
        """ build Hessian matrix at given point m """
        setfct(self.m, m)
        self._assemble_invMw()

        self.Ax = assemble(self.wkformAx)
        Hxasym = MatMatMult(self.Htvx, MatMatMult(self.invMwMat, self.Ax))
        Hx = (Hxasym + Transpose(Hxasym)) * 0.5
        Axrs = assemble(self.wkformAxrs)
        Hxrsasym = MatMatMult(self.Htvx, MatMatMult(self.invMwMat, Axrs))
        Hxrs = (Hxrsasym + Transpose(Hxrsasym)) * 0.5

        self.Ay = assemble(self.wkformAy)
        Hyasym = MatMatMult(self.Htvy, MatMatMult(self.invMwMat, self.Ay))
        Hy = (Hyasym + Transpose(Hyasym)) * 0.5
        Ayrs = assemble(self.wkformAyrs)
        Hyrsasym = MatMatMult(self.Htvy, MatMatMult(self.invMwMat, Ayrs))
        Hyrs = (Hyrsasym + Transpose(Hyrsasym)) * 0.5

        self.H = Hx + Hy
        self.Hrs = Hxrs + Hyrs

        self.precond = self.Hrs + self.sMass


    def hessian(self, mhat):
        return self.Hrs * mhat
        #return self.H * mhat


    def compute_what(self, mhat):
        """ Compute update direction for what, given mhat """
        self.wxhat.vector().zero()
        self.wxhat.vector().axpy(1.0, self.invMwd*(self.Ax*mhat - self.gwx.vector()))
        normwxhat = np.linalg.norm(self.wxhat.vector().array())

        self.wyhat.vector().zero()
        self.wyhat.vector().axpy(1.0, self.invMwd*(self.Ay*mhat - self.gwy.vector()))
        normwyhat = np.linalg.norm(self.wyhat.vector().array())

        print '|what|={}'.format(np.sqrt(normwxhat**2 + normwyhat**2))


    def update_w(self, mhat, alphaLS, compute_what=True):
        """ update dual variable in direction what 
        and update re-scaled version """
        if compute_what:    self.compute_what(mhat)
        self.wx.vector().axpy(alphaLS, self.wxhat.vector())
        self.wy.vector().axpy(alphaLS, self.wyhat.vector())

        rescaledradiusdual = 1.0    # 1.0: checked empirically to be max radius acceptable
        wxa, wya = self.wx.vector().array(), self.wy.vector().array()
        normw = np.sqrt(wxa**2 + wya**2)
        factorw = [max(1.0, ii/rescaledradiusdual) for ii in normw]
        nbrescaled = [1.0*(ii > rescaledradiusdual) for ii in normw]
        print 'perc. dual entries rescaled={:.2f} %, min(factorw)={}, max(factorw)={}'.format(\
        100.*sum(nbrescaled)/len(nbrescaled), min(factorw), max(factorw))
        setfct(self.wxrs, wxa/factorw)
        setfct(self.wyrs, wya/factorw)
        

    def getprecond(self):
        """ Precondition by inverting the TV Hessian """

        solver = PETScKrylovSolver('cg', self.amgprecond)
        solver.parameters["maximum_iterations"] = 2000
        solver.parameters["relative_tolerance"] = 1e-24
        solver.parameters["absolute_tolerance"] = 1e-24
        solver.parameters["error_on_nonconvergence"] = True 
        solver.parameters["nonzero_initial_guess"] = False 

        # used to compare iterative application of preconditioner 
        # with exact application of preconditioner:
        #solver = PETScLUSolver("petsc")
        #solver.parameters['symmetric'] = True
        #solver.parameters['reuse_factorization'] = True

        solver.set_operator(self.precond)

        return solver


    def init_vector(self, u, dim):
        self.sMass.init_vector(u, dim)





#### MATRIX-FREE VERSION -- NOT SUITABLE FOR FENICS PRECONDITIONERS
#    def grad(self, m):
#        """ compute the gradient at m (and what2) """
#
#        setfct(self.m, m)
#        self._assemble_invMw()
#
#        self.what2.vector().zero()
#        self.what2.vector().axpy(-1.0, self.invMwd * assemble(self.misfitw))
#        #print '|what2|={}'.format(np.linalg.norm(self.what2.vector().array()))
#
#        return assemble(self.wkformgrad) + self.Htv*self.what2.vector()
#        #return assemble(self.wkformgrad)
#
#
#    def assemble_hessian(self, m):
#        """ For PD-TV, we do not really assemble the Hessian,
#        but instead assemble the matrices that will be inverted
#        in the evaluation of what """
#
#        setfct(self.m, m)
#        self._assemble_invMw()
#
#        self.Ars = assemble(self.wkformArs)
#        #self.A = assemble(self.wkformA)
#        self.A = self.Ars   # using re-scaled dual variable for Hessian
#        self.yA, self.xA = Vector(), Vector()
#        self.A.init_vector(self.yA, 1)
#        self.A.init_vector(self.xA, 0)
#
#
#    def hessian(self, mhat):
#        """ evaluate H*mhat, with H symmetric
#        mhat must be a dolfin vector """
#
#        rhswhat1 = self.A * mhat
#        self.what1.vector().zero()
#        self.what1.vector().axpy(1.0, self.invMwd * rhswhat1)
#
#        self.xH.zero()
#        self.xH.axpy(1.0, mhat)
#        self.Htv.transpmult(self.xH , self.rhswhat1s)
#        self.what1s.vector().zero()
#        self.what1s.vector().axpy(1.0, self.invMwd * self.rhswhat1s)
#
#        self.what.vector().zero()
#        self.what.vector().axpy(1.0, self.what1.vector())
#        self.what.vector().axpy(1.0, self.what2.vector())
#
#        self.xA.zero()
#        self.xA.axpy(1.0, self.what1s.vector())
#        self.A.transpmult(self.xA, self.yA)
#        return 0.5*(self.Htv * self.what.vector() + self.yA)
#        #return self.Htv * self.what.vector()
#
#
#    def hessianrs(self, mhat):
#        """ evaluate H*mhat, with H symmetric and positive definite
#        mhat must be a dolfin vector """
#
#        rhswhat1 = self.Ars * mhat
#
#        self.xH.zero()
#        self.xH.axpy(1.0, mhat)
#        self.Htv.transpmult(self.xH , self.rhswhat1s)
#
#        self.wtmp.vector().zero()
#        self.wtmp.vector().axpy(1.0, self.invMwd * rhswhat1)
#        self.wtmp.vector().axpy(1.0, self.what2.vector())
#
#        self.xA.zero()
#        self.xA.axpy(1.0, self.invMwd * self.rhswhat1s)
#        self.Ars.transpmult(self.xA, self.yA)
#
#        return 0.5*(self.Htv * self.wtmp.vector() + self.yA)




#### OLDER VERSION
#class TVPD(TV):
#    """ Total variation using primal-dual Newton """
#
#    def isPD(self): return True
#
#    def updatePD(self):
#        """ Update the parameters.
#        parameters should be:
#            - k(x) = factor inside TV
#            - eps = regularization parameter
#            - Vm = FunctionSpace for parameter. 
#        ||f||_TV = int k(x) sqrt{|grad f|^2 + eps} dx
#        """
#        # primal dual variables
#        self.Vw = FunctionSpace(self.Vm.mesh(), 'DG', 0)
#        self.wH = Function(self.Vw*self.Vw)  # dual variable used in Hessian (re-scaled)
#        #self.wH = nabla_grad(self.m)/sqrt(self.fTV) # full Hessian
#        self.w = Function(self.Vw*self.Vw)  # dual variable for primal-dual, initialized at 0
#        self.dm = Function(self.Vm)
#        self.dw = Function(self.Vw*self.Vw)  
#        self.testw = TestFunction(self.Vw*self.Vw)
#        self.trialw = TrialFunction(self.Vw*self.Vw)
#        # investigate convergence of dual variable
#        self.dualres = self.w*sqrt(self.fTV) - nabla_grad(self.m)
#        self.dualresnorm = inner(self.dualres, self.dualres)*dx
#        self.normgraddm = inner(nabla_grad(self.dm), nabla_grad(self.dm))*dx
#        # Hessian
#        self.wkformPDhess = self.kovsq * ( \
#        inner(nabla_grad(self.trial), nabla_grad(self.test)) - \
#        0.5*( inner(self.wH, nabla_grad(self.test))*\
#        inner(nabla_grad(self.trial), nabla_grad(self.m)) + \
#        inner(nabla_grad(self.m), nabla_grad(self.test))*\
#        inner(nabla_grad(self.trial), self.wH) ) / sqrt(self.fTV) \
#        )*dx
#        # update dual variable
#        self.Mw = assemble(inner(self.trialw, self.testw)*dx)
#        self.rhswwk = inner(-self.w, self.testw)*dx + \
#        inner(nabla_grad(self.m)+nabla_grad(self.dm), self.testw) \
#        /sqrt(self.fTV)*dx + \
#        inner(-inner(nabla_grad(self.m),nabla_grad(self.dm))* \
#        self.w/sqrt(self.fTV), self.testw)*dx
#        #self.wH/sqrt(self.fTV), self.testw)*dx # not sure why self.wH
#
#    def compute_dw(self, dm):
#        """ Compute dw """
#        setfct(self.dm, dm)
#        b = assemble(self.rhswwk)
#        solve(self.Mw, self.dw.vector(), b)
#        # Compute self.dw in a pointwise sense, where
#        # self.dw is defined at the quadrature nodes, and
#        # the values of nabla_grad(self.m) and nabla_grad(self.dm) are
#        # also read the quadrature nodes
#
#
#    def update_w(self, alpha, printres=True):
#        """ update w and re-scale wH """
#        self.w.vector().axpy(alpha, self.dw.vector())
#        # project each w (coord-wise) onto unit sphere to get wH
#        (wx, wy) = self.w.split(deepcopy=True)
#        wxa, wya = wx.vector().array(), wy.vector().array()
#        normw = np.sqrt(wxa**2 + wya**2)
#        factorw = [max(1.0, ii) for ii in normw]
#        setfct(wx, wxa/factorw)
#        setfct(wy, wya/factorw)
#        assign(self.wH.sub(0), wx)
#        assign(self.wH.sub(1), wy)
#        # check
#        (wx,wy) = self.wH.split(deepcopy=True)
#        wxa, wya = wx.vector().array(), wy.vector().array()
#        assert np.amax(np.sqrt(wxa**2 + wya**2)) <= 1.0 + 1e-14
#        # Print results
#        if printres:
#            dualresnorm = assemble(self.dualresnorm)
#            normgraddm = assemble(self.normgraddm)
#            print 'line search dual variable: max(|w|)={}, err(w,df)={}, |grad(dm)|={}'.\
#            format(np.amax(np.sqrt(normw)), np.sqrt(dualresnorm), np.sqrt(normgraddm))
