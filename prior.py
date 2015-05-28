import abc
import numpy as np

from dolfin import Function, TrialFunction, TestFunction, Vector, \
assemble, inner, nabla_grad, dx, \
PETScMatrix, SLEPcEigenSolver, LUSolver
from miscfenics import isFunction, isVector


class GaussianPrior():
    """Define a general class for a Gaussian prior
    method _assemble needs to define 
    """
    __metaclass__ = abc.ABCMeta
    
    #Instantiation
    def __init__(self, Parameters=None):
        self.Parameters = Parameters
        self._assemble()

    def update_Parameters(self, Paramters):
        self.Parameters = Parameters
        self._assemble()

    @abc.abstractmethod
    def _assemble(self):    return None

    @abc.abstractmethod
    def Minvpriordot(self, vect):   return None
       
    def get_precond(self):
        return self.precond

    def cost(self, m_in):
        isFunction(m_in)
        diff = m_in.vector() - self.m0.vector()
        return 0.5*np.dot(diff.array(), self.Minvpriordot(diff).array())

    def grad(self, m_in):
        isFunction(m_in)
        diff = m_in.vector() - self.m0.vector()
        return self.Minvpriordot(diff)

    def hessian(self, m_in):
        isVector(m_in)
        return self.Minvpriordot(m_in)

    def Ldot(self, vect):   
        """Square root operator of prior; used to sample"""
        raise NotImplementedError("Subclasses should implement this!")

    def sample(self):
        self.sample.assign(self.m0)
        self.draw.vector()[:] = np.random.standard_normal(self.Vm.dim())
        self.sample.vector().axpy(1.0, self.Ldot(self.draw.vector()))
        return self.sample


############################################################3
# Derived classes
############################################################3

class LaplacianPrior(GaussianPrior):
    """Gaussian prior
    Parameters must be a dictionary containing:
        gamma = multiplicative factor applied to <Grad u, Grad v> term
        beta = multiplicative factor applied to <u,v> term (default=0.0)
        m0 = mean (or reference parameter when used as regularization)
        Vm = function space for parameter
    cost = 1/2 * (m-m0)^T.R.(m-m0)"""
    def _assemble(self):
        # Get input:
        self.gamma = self.Parameters['gamma']
        if self.Parameters.has_key('beta'): self.beta = self.Parameters['beta']
        else:   self.beta = 0.0
        self.Vm = self.Parameters['Vm']
        if self.Parameters.has_key('m0'):   
            self.m0 = self.Parameters['m0'].copy(deepcopy=True)
            isFunction(self.m0)
        else:   self.m0 = Function(self.Vm)
        self.mtrial = TrialFunction(self.Vm)
        self.mtest = TestFunction(self.Vm)
        self.sample = Function(self.Vm)
        self.draw = Function(self.Vm)
        # Assemble:
        self.R = assemble(inner(nabla_grad(self.mtrial), \
        nabla_grad(self.mtest))*dx)
        self.M = assemble(inner(self.mtrial, self.mtest)*dx)
        # preconditioner is Gamma^{-1}:
        if self.beta > 1e-16: self.precond = self.gamma*self.R + self.beta*self.M
        else:   self.precond = self.gamma*self.R + (1e-14)*self.M
        # Minvprior is M.A^2 (if you use M inner-product):
        self.Minvprior = self.gamma*self.R + self.beta*self.M
        # L is used to sample

    def Minvpriordot(self, vect):
        return self.Minvprior * vect


class BilaplacianPrior(GaussianPrior):
    """Gaussian prior
    Parameters must be a dictionary containing:
        gamma = multiplicative factor applied to <Grad u, Grad v> term
        beta = multiplicative factor applied to <u,v> term (default=0.0)
        m0 = mean (or reference parameter when used as regularization)
        Vm = function space for parameter
    cost = 1/2 * (m-m0)^T.R.(m-m0)"""
    def _assemble(self):
        # Get input:
        self.gamma = self.Parameters['gamma']
        if self.Parameters.has_key('beta'): self.beta = self.Parameters['beta']
        else:   self.beta = 0.0
        self.Vm = self.Parameters['Vm']
        if self.Parameters.has_key('m0'):   
            self.m0 = self.Parameters['m0'].copy(deepcopy=True)
            isFunction(self.m0)
        else:   self.m0 = Function(self.Vm)
        self.mtrial = TrialFunction(self.Vm)
        self.mtest = TestFunction(self.Vm)
        self.sample = Function(self.Vm)
        self.draw = Function(self.Vm)
        # Assemble:
        self.R = assemble(inner(nabla_grad(self.mtrial), \
        nabla_grad(self.mtest))*dx)
        M = PETScMatrix()
        assemble(inner(self.mtrial, self.mtest)*dx, tensor=self.M)
        # preconditioner is Gamma^{-1}:
        if self.beta > 1e-16: self.precond = self.gamma*self.R + self.beta*self.M
        else:   self.precond = self.gamma*self.R + (1e-14)*self.M
        # Discrete operator K:
        self.K = self.gamma*self.R + self.beta*self.M
        # Get eigenvalues for M:
        self.eigsolM = SLEPcEigenSolver(self.M)
        self.eigsolM.solve()
        # Solver for M^{-1}:
        self.solverM = LUSolver()
        self.solverM.parameters['reuse_factorization'] = True
        self.solverM.parameters['symmetric'] = True
        self.solverM.set_operator(self.M)
        # Solver for K^{-1}:
        self.solverK = LUSolver()
        self.solverK.parameters['reuse_factorization'] = True
        self.solverK.parameters['symmetric'] = True
        self.solverK.set_operator(self.K)

    def Minvpriordot(self, vect):
        """Here M.Gamma^{-1} = K M^{-1} K"""
        mhat = Function(self.Vm)
        self.solverM.solve(mhat.vector(), self.K*vect)
        return self.K * mhat.vector()

    def apply_sqrtM(self, vect):
        """Compute M^{1/2}.vect from Vector() vect"""
        sqrtMv = Function(self.Vm)
        for ii in range(self.Vm.dim()):
            r, c, rx, cx = self.eigsolM.get_eigenpair(ii)
            RX = Vector(rx)
            sqrtMv.vector().axpy(np.sqrt(r)*np.dot(rx.array(), vect.array()), RX)
        return sqrtMv.vector()

    def Ldot(self, vect):   
        """Here L = K^{-1} M^{1/2}"""
        Lb = Function(self.Vm)
        self.solverK.solve(Lb.vector(), self.apply_sqrtM(vect))
        return Lb.vector()