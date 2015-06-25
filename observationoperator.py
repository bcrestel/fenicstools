import abc
import numpy as np
from numpy import sqrt
from numpy.linalg import norm
from numpy.random import randn

from dolfin import Function, TrialFunction, TestFunction, \
Constant, Point, PointSource, as_backend_type, \
assemble, inner, dx
from exceptionsfenics import WrongInstanceError
from miscfenics import isFunction, isarray, arearrays


class ObservationOperator():
    """Define observation operator and all actions using this observation
    operator, i.e., cost function, rhs in adj eqn and term in incr. adj eqn"""
    __metaclass__ = abc.ABCMeta

    #Instantiation
    def __init__(self, parameters=None):
        self.parameters = parameters
        if self.parameters.has_key('noise'):
            self.noise = True
            self.noisepercent = self.parameters['noise']
        else:
            self.noise = False
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

    def apply_noise(self, uin):
        """Apply Gaussian noise to np.array of data.
        noisepercent = 0.02 => 2% noise level, i.e.,
        || u - ud || / || ud || = || noise || / || ud || = 0.02"""
        isarray(uin)
        noisevect = randn(len(uin))
        # Get norm of entire random vector:
        try:
            normrand = sqrt(MPI.sum(mycomm, norm(noisevect)**2))
        except:
            normrand = norm(noisevect)
        noisevect /= normrand
        # Get norm of entire vector ud (not just local part):
        try:
            normud = sqrt(MPI.sum(mycomm, norm(uin)**2))
        except:
            normud = norm(uin)
        noisevect *= self.noisepercent * normud
        objnoise_glob = (self.noisepercent * normud)**2
        UDnoise = uin + noisevect

        return UDnoise, objnoise_glob


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
        if not(self.noise): return uin.vector().array(), 0.0
        else:   return self.apply_noise(uin.vector().array())


    def costfct(self, uin, udin):
        arearrays(uin, udin)
        self.diff.vector()[:] = uin - udin
        return 0.5*np.dot(self.diff.vector().array(), \
        (self.W * self.diff.vector()).array())


    def assemble_rhsadj(self, uin, udin, outp, bc):
        arearrays(uin, udin)
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
        isFunction(uin)
        Bu = np.zeros(self.nbPts)
        for ii, bb in enumerate(self.B):
            Bu[ii] = bb.inner(uin.vector()) # Note: this returns the global inner-product
        return Bu


    def BTdot(self, uin):
        """uin must be a np.array"""
        isarray(uin)
        u = Function(self.V)
        out = u.vector()
        for ii, bb in enumerate(self.B):
            out += bb*uin[ii]
        return out.array()


    def obs(self, uin):
        """uin must be a Function(V)"""
        if not(self.noise): return self.Bdot(uin), 0.0
        else:
            Bref = self.Bdot(uin)
            uin_noise, tmp = self.apply_noise(uin.vector().array())
            unoise = Function(self.V)
            unoise.vector()[:] = uin_noise
            Bnoise = self.Bdot(unoise)
            diff = Bref - Bnoise
            noiselevel = np.dot(diff, diff)
            return Bnoise, noiselevel


    # TODO: will require to fix PostProcess
    # Needs to check what is global, what is local (regularization?)
    def costfct(self, uin, udin):
        arearrays(uin, udin)
        diff = uin - udin
        return 0.5*np.dot(diff, diff)


    def assemble_rhsadj(self, uin, udin, outp, bc):
        arearrays(uin, udin)
        isFunction(outp)
        diff = uin - udin
        outp.vector()[:] = -1.0 * self.BTdot(diff)
        bc.apply(outp.vector())


    def incradj(self, uin):
        isFunction(uin)
        self.BtBu.vector()[:] = self.BTdot( self.Bdot(uin) )
        return self.BtBu.vector()
