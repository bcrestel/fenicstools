import abc
import numpy as np

from dolfin import Function, TrialFunction, TestFunction, \
Constant, Point, PointSource, as_backend_type, \
assemble, inner, dx
#from scipy.sparse import csr_matrix
from exceptionsfenics import WrongInstanceError
from miscfenics import isFunction, isarray


class ObservationOperator():
    """Define observation operator and all actions using this observation
    operator, i.e., cost function, rhs in adj eqn and term in incr. adj eqn"""
    __metaclass__ = abc.ABCMeta

    #Instantiation
    def __init__(self, parameters=None):
        self.parameters = parameters
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



###########################################################
# Derived Classes
###########################################################

class ObsEntireDomain(ObservationOperator):
    """Observation operator over the entire domain
    with L2 misfit norm
    parameters must be dictionary containing:
        V = function space for state variable"""
    
    def _assemble(self):
        self.V = self.parameters['V']
        self.diff = Function(self.V)
        self.trial = TrialFunction(self.V)
        self.test = TestFunction(self.V)
        self.W = assemble(inner(self.trial, self.test)*dx)

    def obs(self, uin):
        isFunction(uin)
        return uin.vector().array()

    def costfct(self, uin, udin):
        isarray(uin, udin)
        self.diff.vector()[:] = uin - udin
        return 0.5*np.dot(self.diff.vector().array(), \
        (self.W * self.diff.vector()).array())

    def assemble_rhsadj(self, uin, udin, outp, bc):
        isarray(uin, udin)
        isFunction(outp)
        self.diff.vector()[:] = uin - udin
        outp.vector()[:] = - (self.W * self.diff.vector()).array()
        bc.apply(outp.vector())
        
    def incradj(self, uin):
        isFunction(uin)
        return self.W * uin.vector()


class ObsPointwise(ObservationOperator):
    """Observation operator at finite nb of points
    parameters must be a dictionary containing:
        V = function space for state variable
        Points = list of coordinates"""

    def _assemble(self):
        self.V = self.parameters['V']
        self.Points = self.parameters['Points']
        self.nbPts = len(self.Points)
        self.test = TestFunction(self.V)
        self.BtBu = Function(self.V)
        f = Constant('0')
        L = f*self.test*dx
        b = assemble(L)
        self.B = []
        for pts in self.Points:
            delta = PointSource(self.V, self.list2point(pts))
            bs = b.copy()
            delta.apply(bs)
            bs[:] = self.PointSourcecorrection(bs)
            #bs = as_backend_type(bs)   # Turn GenericVector into PETScVector
            self.B.append(bs)
# OLD VERSION:
#        Dobs = np.zeros(self.nbPts*b.size(), float) 
#        Dobs = Dobs.reshape((self.nbPts, b.size()), order='C')
#        for index, pts in enumerate(self.Points):
#            delta = PointSource(self.V, self.list2point(pts))
#            bs = b.copy()
#            delta.apply(bs)
#            Dobs[index,:] = bs.array().transpose()
#        self.B = csr_matrix(Dobs)
#        self.BtB = csr_matrix((self.B.T).dot(self.B)) 


    def PointSourcecorrection(self, b):
        """Correct PointSource in parallel"""
        # TODO: TO BE TESTED!!
        scale = b.array().sum()
        if abs(scale) > 1e-12:  
            return b.array()/scale
        else:   return b.array()
        

    def list2point(self, list_in):
        """Turn a list of coord into a Fenics Point
        list_in = list containing coordinates of the Point"""
        dim = np.size(list_in)
        return Point(dim, np.array(list_in, dtype=float))


    def Bdot(self, uin):
        """uin must be a Function(self.V)"""
        Bu = np.zeros(self.nbPts)
        for ii, bb in enumerate(self.B):
            Bu[ii] = bb.inner(uin.vector()) # Note: this returns the global inner-product
        return Bu


    def BTdot(self, uin):
        """uin must be a np.array"""
        u = Function(self.V)
        out = u.vector()
        for ii, bb in enumerate(self.B):
            out += bb*uin[ii]
        return out.array()


    def obs(self, uin):
        isFunction(uin)
        return self.Bdot(uin)


    # TODO: will require to fix PostProcess
    def costfct(self, uin, udin):
        isarray(uin, udin)
        diff = uin - udin
        return 0.5*np.dot(diff, diff)


    def assemble_rhsadj(self, uin, udin, outp, bc):
        isarray(uin, udin)
        isFunction(outp)
        diff = uin - udin
        outp.vector()[:] = -1.0 * self.BTdot(diff)
        bc.apply(outp.vector())


    def incradj(self, uin):
        isFunction(uin)
        self.BtBu.vector()[:] = self.BTdot( self.Bdot(uin) )
        return self.BtBu.vector()
