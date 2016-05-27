"""
Define observation operators for inverse problem
"""

import sys
import abc
import numpy as np
from numpy import sqrt
from numpy.linalg import norm
from numpy.random import randn
import matplotlib.pyplot as plt

try:
    from dolfin import Function, TrialFunction, TestFunction, \
    Constant, Point, as_backend_type, \
    assemble, inner, dx, MPI 
except:
    from dolfin import Function, TrialFunction, TestFunction, \
    Constant, Point, as_backend_type, \
    assemble, inner, dx
from exceptionsfenics import WrongInstanceError
from miscfenics import isFunction, isVector, isarray, arearrays, isequal, setfct
from sourceterms import PointSources

#TODO: fix pb with np.arrays that only soft copy (uin = uin.T). FIXED BUT CHECK

class TimeFilter():
    """ Create time filter to fade out data misfit (hence src term in adj eqn) """

    def __init__(self, times=None):
        """ Input times:
            times[0] = t0 = initial time
            times[1] = t1 = beginning of flat section at 1.0
            times[2] = t2 = end of flat section at 1.0
            times[3] = T = final time """
        if times == None:   
            self.t0 = -np.inf
            self.t1 = -np.inf
            self.t2 = np.inf
            self.T = np.inf
        else:
            self.t0 = times[0]
            self.t1 = times[1]
            self.t2 = times[2]
            self.T = times[3]
            self.t1b = 2*self.t1-self.t0
            self.t2b = 2*self.t2-self.T


    def __call__(self, tt):
        """ Overload () operator """
        assert tt >= self.t0 and tt <= self.T, "Input tt out of bounds [t0, T]"
        if tt <= self.t0 + 1e-16: return 0.0
        if tt >= self.T - 1e-16:    return 0.0
        if tt <= self.t1:   
            return np.exp(-1./((tt-self.t0)*(self.t1b-tt)))/np.exp(-1./(self.t1-self.t0)**2)
        if tt >= self.t2:   
            return np.exp(-1./((tt-self.t2b)*(self.T-tt)))/np.exp(-1./(self.T-self.t2)**2)
        return 1.0


    def evaluate(self, times):
        """ vectorized verstion of __call__ """
        vectcall = np.vectorize(self.__call__)
        return vectcall(times)


    def plot(self, ndt=1000):
        """ Plot the shape of the filter along with its fft """
        tt = np.linspace(self.t0, self.T, ndt)
        xx = self.evaluate(tt)
        ff = np.fft.fft(xx) # fft(time-domain)
        ffn = np.sqrt(ff.real**2 + ff.imag**2)
        ffxi = np.fft.fftfreq(len(xx), d=tt[1]-tt[0])
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.plot(tt, xx)
        ax2 = fig.add_subplot(122)
        ax2.plot(np.fft.fftshift(ffxi), np.fft.fftshift(ffn))
        return fig


###########################################################
###########################################################

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
        self.diffv = self.diff.vector()
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
        self.diffv[:] = uin - udin
        return 0.5 * (self.W*self.diffv).inner(self.diffv)

    def assemble_rhsadj(self, uin, udin, outp, bc):
        arearrays(uin, udin)
        isFunction(outp)
        self.diffv[:] = uin - udin
        outp.vector()[:] = - (self.W * self.diffv).array()
        bc.apply(outp.vector())

    def incradj(self, uin):
        isFunction(uin)
        return self.hessian(uin.vector())

    # Make non-array version of cost:
    def costfct_F(self, uin, udin):
        isFunction(uin)
        isFunction(udin)
        setfct(self.diff, uin.vector()-udin.vector())
        return 0.5 * (self.W*self.diffv).inner(self.diffv)

    def grad(self, uin, udin):
        isFunction(uin)
        isFunction(udin)
        setfct(self.diff, uin.vector() - udin.vector())
        return self.W * self.diffv

    def hessian(self, uin):
        isVector(uin)
        return self.W * uin



##################
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
            out.axpy(uin[ii], bb)
        return out.array()

    #@profile
    def BTdotvec(self, uin, outvect):
        """ Compute B^T.uin """
        isarray(uin)
        outvect.zero()
        for ii, bb in enumerate(self.B):
            outvect.axpy(uin[ii], bb)

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



##################
class TimeObsPtwise():
    """
    Create time-dependent pointwise observation operator
    Arguments to assemble:
        paramObsPtwise = parameters used to create ptwise observation operator
        timefilter = [t0, t1, t2, T], times to initialize time-filtering function
        mycomm = MPI communicator
    """

    def __init__(self, paramObsPtwise, timefilter=None, mycomm=None):
        self.PtwiseObs = ObsPointwise(paramObsPtwise, mycomm)
        u = Function(self.PtwiseObs.V)
        self.outvec = u.vector()
        self.st = TimeFilter(timefilter)

    def obs(self, uin):
        """ return result from pointwise observation w/o time-filtering """
        return  self.PtwiseObs.Bdot(uin)

    def costfct(self, uin, udin, times):
        """ compute cost functional
                int_0^T s(t) | B u - udin |^2 dt
        uin, udin must be in np.array format of shape 'recv x time'
        times should be array containg values of t0, t1,..., T, 
        and time-steps should be all equal (!!)
        """
        assert uin.shape == udin.shape, "uin and udin must have same shape"
        assert uin.shape[0] == len(times) or uin.shape[1] == len(times), \
        "must have as many time steps in uin as in times"
        if uin.shape[0] == len(times):  diffuinudinsq = (uin.T - udin.T)**2
        else:   diffuinudinsq = (uin - udin)**2
        #
        factors = np.ones(len(times))
        factors[0], factors[-1] = 0.5, 0.5
        Dt = times[1] - times[0]
        diff = diffuinudinsq*factors*(self.st.evaluate(times))
        return 0.5*Dt*(diff.sum().sum())

    def assemble_rhsadj(self, uin, udin, times, bcadj):
        """ Assemble data for rhs of adj eqn 
            uin, udin = observations and data @ receivers
            times = time steps of solution
            bcadj = boundary conditions
        uin.shape must be 'recv x time' """
        assert uin.shape == udin.shape, "uin and udin must have same shape"
        assert uin.shape[0] == len(times) or uin.shape[1] == len(times), \
        "must have as many time steps in uin as in times"
        #
        if uin.shape[0] == len(times):  
            self.diff = uin.T - udin.T
        else:   
            self.diff = uin - udin
        self.times = times
        self.bcadj = bcadj

    #@profile
    def ftimeadj(self, tt):
        """ Evaluate source term for adj eqn at time tt """
        try:
            index = int(np.where(isequal(self.times, tt, 1e-14))[0])
        except:
            print 'Error in ftimeadj at time {}'.format(tt)
            print np.min(np.abs(self.times-tt))
            sys.exit(0)
        dd = self.diff[:, index]
        self.PtwiseObs.BTdotvec(dd, self.outvec)
        if not self.bcadj == None:  self.bcadj.apply(self.outvec.vector())
        return self.outvec*(-1.0*self.st(tt))

    #@profile
    def incradj(self, uhat, tt):
        """ Compute B^T B uhat """
        self.PtwiseObs.BTdotvec(self.obs(uhat), self.outvec)
        return self.outvec*self.st(tt)
