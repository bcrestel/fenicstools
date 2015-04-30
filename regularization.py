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
        Vm = function space for parameter"""

    def _assemble(self):
        self.gamma = self.Parameters['gamma']
        self.Vm = self.Parameters['Vm']
        self.mtrial = TrialFunction(self.Vm)
        self.mtest = TestFunction(self.Vm)
        self.R = assemble(inner(nabla_grad(self.mtrial), \
        nabla_grad(self.mtest))*dx)
       
    def get_R(self):
        return self.gamma * self.R

    def cost(self, m_in):
        self.isFunction(m_in)
        return 0.5*self.gamma * np.dot(m_in.vector().array(), \
        (self.R*m_in.vector()).array())

    def grad(self, m_in):
        self.isFunction(m_in)
        return self.gamma * (self.R * m_in.vector())

    def hessian(self, m_in):
        self.isVector(m_in)
        return self.gamma * (self.R * m_in)
