import abc
import numpy as np

from dolfin import TestFunction, TrialFunction, Function, GenericVector, \
assemble, inner, nabla_grad, dx, ds, LUSolver, sqrt, \
PointSource, Point, Constant
from miscfenics import isFunction, isVector


class PDESolver():
    """Defines abstract class to compute solution of a PDE"""
    __metaclass__ = abc.ABCMeta

    def __init__(self, functionspaces_V):
        self.readV(functionspaces_V)

    @abc.abstractmethod
    def readV(self, functionspaces_V):  return None

    def readoptions(self, options):  return None

    @abc.abstractmethod
    def update(self, parameters_m): return None

    @abc.abstractmethod
    def solve(self, rhs):   return None

    def setfct(self, fct, value):
        if isinstance(value, np.ndarray):
            fct.vector()[:] = value
        elif isinstance(value, Function):
            fct.assign(value)
        elif isinstance(value, float):
            fct.vector()[:] = value
        elif isinstance(value, int):
            fct.vector()[:] = float(value)



############################################################################
# Derived classes
############################################################################

class Wave(PDESolver):

    def update(self, parameters_m):
        self.setfct(self.lam, parameters_m['lambda'])
        if self.elastic == True:    self.setfct(self.mu, parameters_m['mu'])
        self.K = assemble(self.weak_k)
        self.D = assemble(self.weak_d)

        if parameters_m.has_key('rho'):
            #TODO: lump mass matrix
            self.setfct(self.rho, parameters_m['rho'])
            self.M = assemble(self.weak_m)
            self.solverM = LUSolver()
            self.solverM.parameters['reuse_factorization'] = True
            self.solverM.parameters['symmetric'] = True
            self.solverM.set_operator(self.M)

        # Time options:
        if parameters_m.has_key('t0'):   self.t0 = parameters_m['t0'] 
        if parameters_m.has_key('tf'):   self.tf = parameters_m['tf'] 
        if parameters_m.has_key('Dt'):   self.Dt = parameters_m['Dt'] 
        #TODO: Add option for fwd or adj pb


    def solve(self, rhs):
        # u0:
        self.setfct(self.u_nm1, 0.0)
        tt = self.t0 
        # u1:
        self.solverM.solve(self.u_n.vector(), 0.5*self.Dt**2*self.src(tt))
        tt += self.Dt
        # Iteration
        while tt < self.tf:
            self.setfct(self.u_np1, 0.0)
            self.u_np1.vector().axpy(2.0, self.u_n.vector())
            self.u_np1.vector().axpy(-1.0, self.u_nm1.vector())
            self.u_np1.vector().axpy(self.Dt, \
            self.rhs(self.src(tt), self.u_n, self.u_nm1, self.Dt))
            # Advance time by Dt:
            self.setfct(self.u_nm1, self.u_n)
            self.setfct(self.u_n, self.u_np1)
            tt += self.Dt


    def rhs(self, f, un, unm1, Dt):
        """Compute rhs for wave equation
        where f = Vector(V) and u = Function(V)"""
        rhs = self.D * (unm1.vector() - un.vector())
        rhs.axpy(Dt, f - self.K*un.vector())
        out = Function(self.V)
        self.solverM.solve(out.vector(), rhs)
        return out.vector()


    def src(self, time):
        """Compute f(x,t) at given time"""
        return self.ftime(time) * self.f


    def readV(self, functionspaces_V):
        # Solutions:
        self.V = functionspaces_V['V']
        self.test = TestFunction(self.V)
        self.trial = TrialFunction(self.V)
        self.u_nm1 = Function(self.V)    # u(t-Dt)
        self.u_n = Function(self.V)     # u(t)
        self.u_np1 = Function(self.V)    # u(t+Dt)
        # Parameters:
        self.Vl = functionspaces_V['Vl']
        self.lam = Function(self.Vl)
        self.Vr = functionspaces_V['Vr']
        self.rho = Function(self.Vr)
        if functionspaces_V.has_key('Vm'):
            self.Vm = functionspaces_V['Vm']
            self.mu = Function(self.Vm)
            self.elastic = True
            assert(False)   # TODO: Define elastic case
        else:   
            self.elastic = False
            self.weak_k = self.lam * inner(nabla_grad(self.test), \
            nabla_grad(self.trial)) *dx
            self.weak_d = sqrt(self.lam*self.rho) * self.test*self.trial*ds
            self.weak_m = self.rho * self.test*self.trial*dx


    def definesource(self, inputf, timestamp):
        """
        inputf can be either: Vector(V) or dict containing a keyword (delta or
        ricker) and location of source term (+ frequency for ricker)
        timestamp is a function of time
        """
        if isinstance(inputf, GenericVector):
            ff = Function(self.V)
            ff.vector()[:] = inputf.array()
            self.f = ff.vector()
        elif isinstance(inputf, dict):
            if inputf['type'] == 'delta':
                assert inputf.has_key('point')
                f = Constant('0')
                L = f*self.test*dx
                self.f = assemble(L)
                delta = PointSource(self.V, self.list2point(inputf['point']))
                delta.apply(self.f)
                self.f[:] = self.PointSourcecorrection(self.f)
            elif inputf['type'] == 'ricker':
                # TODO
                assert False
            else:   assert False
        else:   assert False

        self.ftime = timestamp


    #TODO: create separate class for point source
    def list2point(self, list_in):
        """Turn a list of coord into a Fenics Point
        list_in = list containing coordinates of the Point"""
        dim = np.size(list_in)
        return Point(dim, np.array(list_in, dtype=float))


    def PointSourcecorrection(self, b):
        """Correct PointSource in parallel"""
        # TODO: TO BE TESTED!!
        scale = b.array().sum()
        if abs(scale) > 1e-12:  
            return b.array()/scale
        else:   return b.array()
        


