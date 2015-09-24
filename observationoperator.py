import abc
import numpy as np
from numpy import sqrt
from numpy.linalg import norm
from numpy.random import randn

try:
    from dolfin import Function, TrialFunction, TestFunction, \
    Constant, Point, as_backend_type, \
    assemble, inner, dx, MPI 
except:
    from dolfin import Function, TrialFunction, TestFunction, \
    Constant, Point, as_backend_type, \
    assemble, inner, dx
from exceptionsfenics import WrongInstanceError
from miscfenics import isFunction, isarray, arearrays
from sourceterms import PointSources


class ObservationOperator():
    """Define observation operator and all actions using this observation
    operator, i.e., cost function, rhs in adj eqn and term in incr. adj eqn"""
    __metaclass__ = abc.ABCMeta

    #Instantiation
    def __init__(self, parameters, mycomm):
        self.parameters = parameters
        if self.parameters.has_key('noise'):
            self.noise = True
            self.noisepercent = self.parameters['noise']
        else:
            self.noise = False
        self.mycomm = mycomm
        self._assemble()

    @abc.abstractmethod
    def _assemble(self):    print "Needs to be implemented"

    @abc.abstractmethod
    def obs(self, uin): print "Needs to be implemented"

    # Note: This must return the global value (in parallel)
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
        if self.mycomm == None: normrand = norm(noisevect)
        else:
            normrand = sqrt(MPI.sum(self.mycomm, norm(noisevect)**2))
        noisevect /= normrand
        # Get norm of entire vector ud (not just local part):
        if self.mycomm == None: normud = norm(uin)
        else:
            normud = sqrt(MPI.sum(self.mycomm, norm(uin)**2))
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


    # Note: this returns global value (in parallel)
    def costfct(self, uin, udin):
        arearrays(uin, udin)
        self.diff.vector()[:] = uin - udin
        return 0.5 * (self.W*self.diff.vector()).inner(self.diff.vector())


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
        self.BtBu = Function(self.V)
        PtSrc = PointSources(self.V, self.Points)
        self.B = PtSrc.PtSrc


    def Bdotlocal(self, uin):
        """Compute B.uin as a np.array, using only local info
        uin must be a Function(self.V)"""
        isFunction(uin)
        Bu = np.zeros(self.nbPts)
        for ii, bb in enumerate(self.B):
            Bu[ii] = np.dot(bb.array(), uin.vector().array())   # Note: local inner-product
        return Bu


    def Bdot(self, uin):
        """Compute B.uin as a np.array, using global info
        uin must be a Function(self.V)"""
        isFunction(uin)
        Bu = np.zeros(self.nbPts)
        for ii, bb in enumerate(self.B):
            Bu[ii] = bb.inner(uin.vector()) # Note: global inner-product
        return Bu


    def BTdot(self, uin):
        """Compute B^T.uin as a np.array
        uin must be a np.array"""
        isarray(uin)
        u = Function(self.V)
        out = u.vector()
        for ii, bb in enumerate(self.B):
            out += bb*uin[ii]
        return out.array()


    def obs(self, uin):
        """Compute B.uin + eps, where eps is noise
        uin must be a Function(V)"""
        if not(self.noise): return self.Bdot(uin), 0.0
        else:
            Bref = self.Bdot(uin)
            uin_noise, tmp = self.apply_noise(uin.vector().array())
            unoise = Function(self.V)
            unoise.vector()[:] = uin_noise
            Bnoise = self.Bdot(unoise)
            diff = Bref - Bnoise
            noiselevel = np.dot(diff, diff)
            try:
                noiselevel_glob = MPI.sum(self.mycomm, noiselevel)
            except:
                noiselevel_glob = noiselevel
            return Bnoise, noiselevel_glob


    def costfct(self, uin, udin):
        """Compute cost functional from observed fwd and data, i.e.,
        return .5*||uin - udin||^2.
        uin & udin are np.arrays"""
        arearrays(uin, udin)
        diff = uin - udin
        return 0.5*np.dot(diff, diff)


    def assemble_rhsadj(self, uin, udin, outp, bc):
        """Compute rhs term for adjoint equation and store it in outp, i.e.,
        outp = - B^T( uin - udin), where uin = obs(fwd solution)
        uin & udin = np.arrays
        outp = Function(self.V)
        bc = fenics' boundary conditons"""
        arearrays(uin, udin)
        isFunction(outp)
        diff = uin - udin
        outp.vector()[:] = -1.0 * self.BTdot(diff)
        bc.apply(outp.vector())


    def incradj(self, uin):
        """Compute the observation part of the incremental adjoint equation, i.e,
        return B^T.B.uin
        uin = Function(self.V)"""
        isFunction(uin)
        self.BtBu.vector()[:] = self.BTdot( self.Bdotlocal(uin) )
        return self.BtBu.vector()
