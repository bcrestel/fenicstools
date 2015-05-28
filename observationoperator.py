import abc
import numpy as np

from dolfin import Function, TrialFunction, TestFunction, \
Constant, Point, PointSource, \
assemble, inner, dx
from scipy.sparse import csr_matrix
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


class ObsPointwise(ObservationOperator):
    """Observation operator at finite nb of points
    Parameters must be a dictionary containing:
        V = function space for state variable
        Points = list of coordinates"""

    def _assemble(self):
        self.V = self.Parameters['V']
        self.Points = self.Parameters['Points']
        self.nbPts = len(self.Points)
        self.test = TestFunction(self.V)
        # Build observation operator B and B^TB
        f = Constant('0')
        L = f*test*dx
        b = assemble(L)
        Dobs = np.zeros(NbPts*b.size(), float) 
        Dobs = Dobs.reshape((NbPts, b.size()), order='C')
        for index, pts in enumerate(self.Points):
            delta = PointSource(self.V, self.list2point(pts))
            bs = b.copy()
            delta.apply(bs)
            Dobs[index,:] = bs.array().transpose()
        self.B = csr_matrix(Dobs)
        self.BtB = csr_matrix((self.B.T).dot(Dr)) 

    def list2point(list_in):
        """Turn a list of coord into a Fenics Point
        list_in = list containing coordinates of the Point"""
        dim = np.size(list_in)
        return Point(dim, np.array(list_in, dtype=float))

    def obs(self, uin):
        self.isFunction(uin)
        return self.B.dot(uin.vector().array())

    def costfct(self, uin, udin):
        self.isarray(uin, udin)
        diff = uin - udin
        return 0.5*np.dot(diff, diff)

    def assemble_rhsadj(self, uin, udin, outp, bc):
        self.isarray(uin, udin)
        self.isFunction(outp)
        diff = uin - udin
        outp.vector()[:] = - (self.B.T).dot(diff)
        bc.apply(outp.vector())

    def incradj(self, uin):
        self.isFunction(uin)
        return self.BtB.dot(uin.vector().array())
