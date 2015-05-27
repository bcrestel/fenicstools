import abc
import numpy as np

from dolfin import Function, TrialFunction, TestFunction, assemble, inner, dx
from exceptionsfenics import WrongInstanceError


class ObservationOperator():
    """Define observation operator and all actions using this observation
    operator, i.e., cost function, rhs in adj eqn and term in incr. adj eqn"""
    __metaclass__ = abc.ABCMeta

    #Instantiation
    def __init__(self, Parameters=None):
        self.Parameters = Parameters
        self._assemble()

    @abc.abstractmethod
    def _assemble(self):    print "Needs to be implemented"

    @abc.abstractmethod
    def obs(self, uin): print "Needs to be implemented"

    @abc.abstractmethod
    def costfct(self, uin, udin):   print "Needs to be implemented"

    @abc.abstractmethod
    def assemble_rhsadj(self, uin, udin):   print "Needs to be implemented"
        
    @abc.abstractmethod
    def incradj(self, uin):    print "Needs to be implemented"

    # Checkers
    def isFunction(self, uin):
        if not isinstance(uin, Function):
         raise WrongInstanceError("uin should be a Dolfin Function")
            
    def isarray(self, uin, udin):
        if not (isinstance(uin, np.ndarray) and isinstance(udin, np.ndarray)):
         raise WrongInstanceError("uin and udin should be a Numpy array")


###########################################################
# Derived Classes
###########################################################

class ObsEntireDomain(ObservationOperator):
    """Observation operator over the entire domain
    with L2 misfit norm
    Parameters must be dictionary containing:
        V = function space for state variable"""
    
    def _assemble(self):
        self.V = self.Parameters['V']
        self.diff = Function(self.V)
        self.trial = TrialFunction(self.V)
        self.test = TestFunction(self.V)
        self.W = assemble(inner(self.trial, self.test)*dx)

    def obs(self, uin):
        self.isFunction(uin)
        return uin.vector().array()

    def costfct(self, uin, udin):
        self.isarray(uin, udin)
        self.diff.vector()[:] = uin - udin
        return 0.5*np.dot(self.diff.vector().array(), \
        (self.W * self.diff.vector()).array())

    def assemble_rhsadj(self, uin, udin, outp, bc):
        self.isarray(uin, udin)
        self.isFunction(outp)
        self.diff.vector()[:] = uin - udin
        outp.vector()[:] = - (self.W * self.diff.vector()).array()
        bc.apply(outp.vector())
        
    def incradj(self, uin):
        self.isFunction(uin)
        return self.W * uin.vector()
