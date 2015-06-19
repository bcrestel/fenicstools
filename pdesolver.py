import abc

from dolfin import TestFunction, TrialFunction, Function, \
assemble, div, dx


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

    def readV(self, functionspaces_V):
        self.V = functionspaces_V['V']
        self.test = TestFunction(self.V)
        self.trial = TrialFunction(self.V)
        self.u_n = Function(self.V)
        self.u_n1 = Function(self.V)
        self.u_n2 = Function(self.V)

        self.Vl = functionspaces_V['Vl']
        self.lda = Function(self.Vl)
        self.Vr = functionspaces_V['Vr']
        self.rho = Function(self.Vr)
        if len(functionspaces_V) > 3:
            self.Vm = functionspaces_V['Vm']
            self.mu = Function(self.Vm)
            self.elastic = True
        else:   
            self.elastic = False
            # TODO: Define absorbing BCs
            self.weak_a = self.lda * div(self.test)*div(self.trial)*dx
            self.weak_m = self.rho * self.test*self.trial*dx


    def readoptions(self, options):
        if options.has_key('t0'):   self.t0 = options['t0'] # initial time
        if options.has_key('tf'):   self.tf = options['tf'] # final time
        if options.has_key('Dt'):   self.Dt = options['Dt'] # time step size
        if options.has_key('u0'):   self.u0 = options['u0'] # IC u(x,0)
        if options.has_key('du0'):   self.du0 = options['du0']  # IC du/dt(x,0)


    def update(self, parameters_m):
        self.setfct(self.lda, parameters_m['lambda']
        if self.elastic = True: self.setfct(self.mu, parameters_m['mu'])
        self.A = assemble(self.weak_a)

        if parameters_m.has_key('rho'):
            self.setfct(self.rho, parameters['rho'])
            self.M = assemble(self.weak_m)
            # TODO: lump mass matrix
            # TODO: set up solver for M


    def solve(self, rhs):
        tt = self.t0
        self.setfct(self.u_n, self.u0)

        while tt < self.tf:

            tt += self.Dt
