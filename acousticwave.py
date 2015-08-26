import abc
import numpy as np

from dolfin import TestFunction, TrialFunction, Function, GenericVector, \
assemble, inner, nabla_grad, dx, ds, LUSolver, sqrt, \
PointSource, Point, Constant, FacetFunction, Measure
from miscfenics import isFunction, isVector, setfct


class AcousticWave():

    def __init__(self, functionspaces_V):
        self.readV(functionspaces_V)
        self.verbose = False    # print info
        self.exact = None   # exact solution at final time
        self.utinit = None
        self.u1init = None
        self.bc = None
        self.abc = False
        self.ftime = lambda x: 0.0  # ftime(tt) = source term at time tt (in np.array())


    def readV(self, functionspaces_V):
        # Solutions:
        self.V = functionspaces_V['V']
        self.test = TestFunction(self.V)
        self.trial = TrialFunction(self.V)
        self.u0 = Function(self.V)    # u(t-Dt)
        self.u1 = Function(self.V)     # u(t)
        self.u2 = Function(self.V)    # u(t+Dt)
        self.rhs = Function(self.V)
        self.sol = Function(self.V)
        # Parameters:
        self.Vl = functionspaces_V['Vl']
        self.lam = Function(self.Vl)
        self.Vr = functionspaces_V['Vr']
        self.rho = Function(self.Vr)
        if functionspaces_V.has_key('Vm'):
            self.Vm = functionspaces_V['Vm']
            self.mu = Function(self.Vm)
            self.elastic = True
            assert(False)
        else:   
            self.elastic = False
            self.weak_k = inner(self.lam*nabla_grad(self.trial), \
            nabla_grad(self.test))*dx
            self.weak_m = inner(self.rho*self.trial,self.test)*dx


    def set_abc(self, mesh, class_bc_abc):
        self.abc = True # False means zero-Neumann all-around
        abc_boundaryparts = FacetFunction("size_t", mesh)
        class_bc_abc.mark(abc_boundaryparts, 1)
        self.ds = Measure("ds")[abc_boundaryparts]
        self.weak_d = inner(self.trial, self.test)*self.ds(1)
        #TODO: Define for general values of lambda and rho (right now,
        # definition below does not work (returns zero matrix).
        # Maybe this has to do with function spaces: lambda and rho are defined
        # in Vl and Vr while test and trial are defined in V.
        #self.weak_d = inner(sqrt(self.lam*self.rho)*self.trial, \
        #self.test)*self.ds(0)


    def update(self, parameters_m):
        setfct(self.lam, parameters_m['lambda'])
        if self.verbose: print 'lambda updated '
        if self.elastic == True:    
            setfct(self.mu, parameters_m['mu'])
            if self.verbose: print 'mu updated'
        if self.verbose: print 'assemble K',
        self.K = assemble(self.weak_k)
        if self.verbose: print ' -- K assembled'
        if self.abc == True:    
            if self.verbose:    print 'assemble D',
            self.D = assemble(self.weak_d)
            if self.verbose:    print ' -- D assembled'
        else:   self.D = 0.0
        # Mass matrix:
        if parameters_m.has_key('rho'):
            #TODO: lump mass matrix
            setfct(self.rho, parameters_m['rho'])
            if self.verbose: print 'rho updated\nassemble M',
            self.M = assemble(self.weak_m)
            if not self.bc == None: self.bc.apply(self.M)
            self.solverM = LUSolver()
            self.solverM.parameters['reuse_factorization'] = True
            self.solverM.parameters['symmetric'] = True
            self.solverM.set_operator(self.M)
            if self.verbose: print ' -- M assembled'
        # Time options:
        if parameters_m.has_key('t0'):   self.t0 = parameters_m['t0'] 
        if parameters_m.has_key('tf'):   self.tf = parameters_m['tf'] 
        if parameters_m.has_key('Dt'):   self.Dt = parameters_m['Dt'] 
        # Initial conditions:
        if parameters_m.has_key('u0init'):   self.u0init = parameters_m['u0init']
        if parameters_m.has_key('utinit'):   self.utinit = parameters_m['utinit']
        if parameters_m.has_key('u1init'):   self.u1init = parameters_m['u1init']
        #TODO: Add option for fwd or adj pb


    def solve(self):
        if self.verbose:    print 'Compute solution'
        solout = [] # Store computed solution
        # u0:
        tt = self.t0 
        if self.verbose:    print 'Compute solution -- time {}'.format(tt)
        self.u0 = self.u0init
        solout.append([self.u0.vector().array(), tt])
        # u1:
        if not self.u1init == None: self.u1 = self.u1init
        else:
            assert(not self.utinit == None)
            self.rhs.vector()[:] = self.ftime(tt) - \
            (self.D*self.utinit.vector()).array() - \
            (self.K*self.u0.vector()).array()
            if not self.bc == None: self.bc.apply(self.rhs.vector())
            self.solverM.solve(self.sol.vector(), self.rhs.vector())
            self.u1.vector()[:] = self.u0.vector().array() + \
            self.Dt*self.utinit.vector().array() + \
            0.5*self.Dt**2*self.sol.vector().array()
        tt += self.Dt
        if self.verbose:    print 'Compute solution -- time {}'.format(tt)
        solout.append([self.u1.vector().array(), tt])
        # Iteration
        while tt + self.Dt < self.tf*(1.0 + 1e-14):
            self.rhs.vector()[:] = self.Dt*(self.ftime(tt) - \
            (self.K*self.u1.vector()).array()) - \
            (self.D*(self.u1.vector()-self.u0.vector())).array()
            if not self.bc == None: self.bc.apply(self.rhs.vector())
            self.solverM.solve(self.sol.vector(), self.rhs.vector())
            self.u2.vector()[:] = 2*self.u1.vector().array() - \
            self.u0.vector().array() + self.Dt*self.sol.vector().array()
            # Advance to next time step
            self.u0.vector()[:] = self.u1.vector().array()
            self.u1.vector()[:] = self.u2.vector().array()
            tt += self.Dt
            if self.verbose:    print 'Compute solution -- time {}'.format(tt)
            solout.append([self.u1.vector().array(),tt])
        timeerror = abs(tt - self.tf)/self.tf
        if timeerror > 1e-12:
            raise RuntimeError('Final time is {} instead of {}'.format(tt, \
            self.tf), 'Relative error is {}'.format(timeerror))
        return solout, self.computeerror()


    def computeerror(self):
        if not self.exact == None:
            MM = assemble(inner(self.trial, self.test)*dx)
            norm_ex = np.sqrt((MM*self.exact.vector()).inner(self.exact.vector()))
            diff = self.exact.vector() - self.u1.vector()
            if norm_ex > 1e-16: return np.sqrt((MM*diff).inner(diff))/norm_ex
            else:   return np.sqrt((MM*diff).inner(diff))
        else:   return []


    #TODO: create separate class for point source
#    def list2point(self, list_in):
#        """Turn a list of coord into a Fenics Point
#        list_in = list containing coordinates of the Point"""
#        dim = np.size(list_in)
#        return Point(dim, np.array(list_in, dtype=float))
#
#
#    def PointSourcecorrection(self, b):
#        """Correct PointSource in parallel"""
#        # TODO: TO BE TESTED!!
#        scale = b.array().sum()
#        if abs(scale) > 1e-12:  
#            return b.array()/scale
#        else:   return b.array()
#        
#    def definesource(self, inputf, timestamp):
#        """
#        inputf can be either: Vector(V) or dict containing a keyword (delta or
#        ricker) and location of source term (+ frequency for ricker)
#        timestamp is a function of time
#        """
#        if isinstance(inputf, GenericVector):
#            ff = Function(self.V)
#            ff.vector()[:] = inputf.array()
#            self.f = ff.vector()
#        elif isinstance(inputf, dict):
#            if inputf['type'] == 'delta':
#                if self.verbose: print 'Create Delta source term'
#                f = Constant('0')
#                L = f*self.test*dx
#                self.f = assemble(L)
#                delta = PointSource(self.V, self.list2point(inputf['point']))
#                delta.apply(self.f)
#                self.f[:] = self.PointSourcecorrection(self.f)
#            elif inputf['type'] == 'ricker':
#                # TODO: Implement Ricker wavelet
#                assert False
#            else:   assert False
#        else:   assert False
#        self.ftime = timestamp


