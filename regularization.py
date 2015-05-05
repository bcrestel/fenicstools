import abc
import numpy as np

from dolfin import *
from exceptionsfenics import WrongInstanceError


class Regularization():
    """Defines regularization operator
    Abstract class"""
    __metaclass__ = abc.ABCMeta
    
    #Instantiation
    def __init__(self, Parameters=None):
        self.Parameters = Parameters
        self._assemble()

    def update_Parameters(self, Paramters):
        self.Parameters = Parameters
        self._assemble()

    @abc.abstractmethod
    def get_R(self):    print "Needs to be implemented"

    @abc.abstractmethod
    def get_precond(self):    print "Needs to be implemented"

    @abc.abstractmethod
    def _assemble(self):    print "Needs to be implemented"

    @abc.abstractmethod
    def cost(self, m_in):   print "Needs to be implemented"

    @abc.abstractmethod
    def grad(self, m_in):   print "Needs to be implemented"
        
    @abc.abstractmethod
    def hessian(self, m_in):    print "Needs to be implemented"

    # Checkers
    def isFunction(self, m_in):
        if not isinstance(m_in, Function):
         raise WrongInstanceError("m_in should be a Dolfin Function")

    def isVector(self, m_in):
        if not isinstance(m_in, GenericVector):
         raise WrongInstanceError("m_in should be a Dolfin Generic Vector")
            

###########################################################
# Derived Classes
###########################################################

class TikhonovH1(Regularization):
    """Tikhonov regularization with H1 semi-norm
    Parameters must be a dictionary containing:
        gamma = multiplicative factor applied to H1 semi-norm
        beta = multiplicative factor applied to mass matrix (default=0.0)
        m0 = reference medium (default to zero vector)
        Vm = function space for parameter
    cost = 1/2 * (m-m0)^T.R.(m-m0)"""

    def _assemble(self):
        self.gamma = self.Parameters['gamma']
        if self.Parameters.has_key('beta'): self.beta = self.Parameters['beta']
        else:   self.beta = 0.0
        self.Vm = self.Parameters['Vm']
        if self.Parameters.has_key('m0'):   
            self.m0 = self.Parameters['m0'].copy(deepcopy=True)
            self.isFunction(self.m0)
        else:   self.m0 = Function(self.Vm)
        self.mtrial = TrialFunction(self.Vm)
        self.mtest = TestFunction(self.Vm)
        self.R = assemble(inner(nabla_grad(self.mtrial), \
        nabla_grad(self.mtest))*dx)
        self.M = assemble(inner(self.mtrial, self.mtest)*dx)
       
    def get_R(self):
        return self.gamma*self.R + self.beta*self.M

    def get_precond(self):
        if self.beta > 1e-16: return self.get_R()
        else:   return self.gamma*self.R + min(1e-14, self.gamma/10.)*self.M

    def cost(self, m_in):
        self.isFunction(m_in)
        diff = m_in.vector() - self.m0.vector()
        return 0.5*( self.gamma*np.dot(diff.array(), (self.R*diff).array()) + \
        self.beta*np.dot(diff.array(), (self.M*diff).array()) )

    def grad(self, m_in):
        self.isFunction(m_in)
        diff = m_in.vector() - self.m0.vector()
        return self.gamma*(self.R*diff) + self.beta*(self.M*diff)

    def hessian(self, m_in):
        self.isVector(m_in)
        return self.gamma*(self.R*m_in) + self.beta*(self.M*m_in)
