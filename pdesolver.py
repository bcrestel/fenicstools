import abc

from dolfin import TestFunction, TrialFunction, Function, \
assemble, div, dx, LUSolver
from miscfenics import isFunction, isVector


class PDESolver():
    """Defines abstract class to compute solution of a PDE"""
    __metaclass__ = abc.ABCMeta

    def __init__(self, functionspaces_V, options=None):
        self.readV(functionspaces_V)
        self.readoptions(options)

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
        self.setfct(self.lda, parameters_m['lambda'])
        if self.elastic == True:    self.setfct(self.mu, parameters_m['mu'])
        self.A = assemble(self.weak_a)

        if parameters_m.has_key('rho'):
            #TODO: lump mass matrix
            self.setfct(self.rho, parameters['rho'])
            self.M = assemble(self.weak_m)
            self.solverM = LUSolver()
            self.solverM.parameters['reuse_factorization'] = True
            self.solverM.parameters['symmetric'] = True
            self.solverM.set_operator(self.M)


    def solve(self, rhs):
        #TODO: Check src term and time
        # u0:
        self.setfct(self.u_nm2, self.u0)
        tt = self.t0 
        # u1:
        self.setfct(self.u_nm1, self.u_nm2)
        self.u_nm1.vector().axpy(self.Dt, self.du0.vector())
        self.u_nm1.vector().axpy(0.5*self.Dt**2, self.rhs(self.src(tt), self.u_nm2))
        tt += self.Dt
        # Iteration
        while tt < self.tf:
            self.setfct(self.u_n, 0.0)
            self.u_n.vector().axpy(2.0, self.u_nm1.vector())
            self.u_n.vector().axpy(-1.0, self.u_nm2.vector())
            self.u_n.vector().axpy(self.Dt**2, self.rhs(self.src(tt), self.u_nm1))
            # Advance time by Dt:
            self.setfct(self.u_nm2, self.u_nm1)
            self.setfct(self.u_nm1, self.u_n)
            tt += self.Dt


    def rhs(self, f, u):
        """Compute M^{-1}(f - A * u)
        where f = Vector(V) and u = Function(V)"""
        f.axpy(-1.0, self.A*u.vector())
        out = Function(self.V)
        self.solverM.solve(out.vector(), f)
        return out.vector()


    def src(self, tt):
        """Compute f(x,t) at time tt"""
        return self.ftime(tt) * self.f


    def readV(self, functionspaces_V):
        # Solutions:
        self.V = functionspaces_V['V']
        self.test = TestFunction(self.V)
        self.trial = TrialFunction(self.V)
        self.u_n = Function(self.V)     # u(t)
        self.u_nm1 = Function(self.V)    # u(t-Dt)
        self.u_nm2 = Function(self.V)    # u(t-2Dt)
        # Parameters:
        self.Vl = functionspaces_V['Vl']
        self.lda = Function(self.Vl)
        self.Vr = functionspaces_V['Vr']
        self.rho = Function(self.Vr)
        if functionspaces_V.has_key('Vm'):
            self.Vm = functionspaces_V['Vm']
            self.mu = Function(self.Vm)
            self.elastic = True
        else:   
            self.elastic = False
            #TODO: Define absorbing BCs
            self.weak_a = self.lda * div(self.test)*div(self.trial)*dx
            self.weak_m = self.rho * self.test*self.trial*dx


    def readoptions(self, options):
        if options.has_key('t0'):   self.t0 = options['t0'] # initial time
        else:   self.t0 = 0.0
        self.tf = options['tf'] # final time
        self.Dt = options['Dt'] # time step size
        if options.has_key('u0'):   self.u0 = options['u0'] # IC u(x,0)
        else:   self.u0 = Function(self.V)
        if options.has_key('du0'):   self.du0 = options['du0']  # IC du/dt(x,0)
        else:   self.du0 = Function(self.V)
        self.f = options['f']   # spatial component of source term
        self.ftime = options['ftime']   # ftime(t) = time component of src term
        #TODO: Add option for fwd or adj pb

        isFunction(self.u0)
        isFunction(self.du0)
        isVector(self.f)        


