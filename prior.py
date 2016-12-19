import abc
import numpy as np

from dolfin import Function, TrialFunction, TestFunction, Vector, \
assemble, inner, nabla_grad, dx, \
PETScMatrix, LUSolver, PETScKrylovSolver
try:
    from dolfin import SLEPcEigenSolver
except:
    pass
from miscfenics import isFunction, isVector, setfct


class GaussianPrior():
    """Define a general class for a Gaussian prior
    method _assemble needs to define 
    """
    __metaclass__ = abc.ABCMeta
    
    #Instantiation
    def __init__(self, Parameters=None):
        self.Parameters = Parameters
        self._assemble()

    def update_Parameters(self, Parameters):
        self.Parameters = Parameters
        self._assemble()

    # For compatibility with Total Variation regularization
    def isTV(self): return False
    def isPD(self): return False

    @abc.abstractmethod
    def _assemble(self):    return None

    @abc.abstractmethod
    def Minvpriordot(self, vect):   return None
       
    def getprecond(self):
        """
        solver = PETScKrylovSolver("richardson", "amg")
        solver.parameters["maximum_iterations"] = 1
        solver.parameters["error_on_nonconvergence"] = False
        solver.parameters["nonzero_initial_guess"] = False
        """
        solver = PETScKrylovSolver("cg", "amg")
        solver.parameters["maximum_iterations"] = 1000
        solver.parameters["relative_tolerance"] = 1e-24
        solver.parameters["absolute_tolerance"] = 1e-24
        solver.parameters["error_on_nonconvergence"] = True 
        solver.parameters["nonzero_initial_guess"] = False 
        solver.set_operator(self.precond)
        return solver


    # Note: this returns global value (in parallel)
    def cost(self, m_in):
        diff = m_in.vector() - self.m0.vector()
        return 0.5 * self.Minvpriordot(diff).inner(diff)

    def costvect(self, m_in):
        diff = m_in - self.m0.vector()
        return 0.5 * self.Minvpriordot(diff).inner(diff)


    def grad(self, m_in):
        isFunction(m_in)
        diff = m_in.vector() - self.m0.vector()
        return self.Minvpriordot(diff)

    def gradvect(self, m_in):
        diff = m_in - self.m0.vector()
        return self.Minvpriordot(diff)


    def assemble_hessian(self, m_in):
        pass

    def hessian(self, m_in):
        isVector(m_in)
        return self.Minvpriordot(m_in)

    def Ldot(self, vect):   
        """Square root operator of prior; used to sample"""
        raise NotImplementedError("Subclasses should implement this!")

    def sample(self):
        self.mysample.assign(self.m0)
        self.draw.vector()[:] = \
        np.random.standard_normal(len(self.draw.vector().array()))
        self.mysample.vector().axpy(1.0, self.Ldot(self.draw.vector()))
        return self.mysample


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
        self.m0 = Function(self.Vm)
        if self.Parameters.has_key('m0'):   
            setfct(self.m0, self.Parameters['m0'])
        self.mtrial = TrialFunction(self.Vm)
        self.mtest = TestFunction(self.Vm)
        self.mysample = Function(self.Vm)
        self.draw = Function(self.Vm)
        # Assemble:
        self.R = assemble(inner(nabla_grad(self.mtrial), \
        nabla_grad(self.mtest))*dx)
        self.M = assemble(inner(self.mtrial, self.mtest)*dx)

        self.Msolver = LUSolver()
        self.Msolver.parameters['reuse_factorization'] = True
        self.Msolver.parameters['symmetric'] = True
        self.Msolver.set_operator(self.M)

        # preconditioner is Gamma^{-1}:
        if self.beta > 1e-10: self.precond = self.gamma*self.R + self.beta*self.M
        else:   self.precond = self.gamma*self.R + (1e-10)*self.M
        # Minvprior is M.A^2 (if you use M inner-product):
        self.Minvprior = self.gamma*self.R + self.beta*self.M
        # L is used to sample


    def Minvpriordot(self, vect):
        return self.Minvprior * vect


    def init_vector(self, u, dim):
        self.R.init_vector(u, dim)


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
        self.mysample = Function(self.Vm)
        self.draw = Function(self.Vm)
        # Assemble:
        self.R = assemble(inner(nabla_grad(self.mtrial), \
        nabla_grad(self.mtest))*dx)
        self.M = PETScMatrix()
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
