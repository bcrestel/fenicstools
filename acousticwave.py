import sys
import numpy as np

from dolfin import TestFunction, TrialFunction, Function, GenericVector, \
assemble, inner, nabla_grad, dx, ds, sqrt, \
PETScLUSolver, PETScKrylovSolver, \
PointSource, Point, Constant, FacetFunction, Measure

from miscfenics import setfct, isequal
from linalg.lumpedmatrixsolver import LumpedMatrixSolverS, LumpedMassPreconditioner


class AcousticWave():
    """
    Solution of forward and adjoint equations for acoustic inverse problem
    a*p'' - div(b*grad(p)) = f
    note: proper acoustic model has a=1/lambda, and b=1/rho, with
    lambda = bulk modulus, rho = ambient density, lambda = rho c^2
    """

    def __init__(self, functionspaces_V, parameters_in=[]):
        """
        Input:
            functionspaces_V = dict containing functionspaces
                V, for state/adj
                Vm, for a and b medium parameters
        """
        self.parameters = {}
        self.parameters['print']        = False
        self.parameters['lumpM']        = False
        self.parameters['lumpD']        = False
        self.parameters['timestepper']  = 'centered'
        self.parameters['abc']          = False
        self.parameters.update(parameters_in)

        self.readV(functionspaces_V)
        self.exact = None   # exact solution at final time
        self.u0init = None  # provides u(t=t0)
        self.utinit = None  # provides u_t(t=t0)
        self.u1init = None  # provides u1 = u(t=t0+/-Dt)
        self.bc = None
        self.set_fwd()  # default is forward problem


    def copy(self):
        """(hard) copy constructor"""
        newobj = self.__class__({'V':self.V, 'Vm':self.Vm}, self.parameters)
        newobj.exact = self.exact
        newobj.utinit = self.utinit
        newobj.u1init = self.u1init
        newobj.bc = self.bc
        if self.abc == True:
            newobj.set_abc(self.V.mesh(), self.class_bc_abc, self.lumpD)
        newobj.ftime = self.ftime
        newobj.update({'a':self.a, 'b':self.b, \
        't0':self.t0, 'tf':self.tf, 'Dt':self.Dt, \
        'u0init':self.u0init, 'utinit':self.utinit, 'u1init':self.u1init})
        return newobj


    def readV(self, functionspaces_V):
        # solutions:
        self.V = functionspaces_V['V']
        self.test = TestFunction(self.V)
        self.trial = TrialFunction(self.V)
        self.u0 = Function(self.V)    # u(t-Dt)
        self.u1 = Function(self.V)     # u(t)
        self.u2 = Function(self.V)    # u(t+Dt)
        self.rhs = Function(self.V)
        self.sol = Function(self.V)
        # medium parameters:
        self.Vm = functionspaces_V['Vm']
        self.a = Function(self.Vm)
        self.b = Function(self.Vm)
        self.weak_k = inner(self.b*nabla_grad(self.trial), nabla_grad(self.test))*dx
        self.weak_m = inner(self.a*self.trial,self.test)*dx


    def set_abc(self, mesh, class_bc_abc, lumpD=False):
        self.parameters['abc'] = True   # False means zero-Neumann all-around
        if lumpD:    self.parameters['lumpD'] = True
        abc_boundaryparts = FacetFunction("size_t", mesh)
        class_bc_abc.mark(abc_boundaryparts, 1)
        self.ds = Measure("ds")[abc_boundaryparts]
        self.weak_d = inner(sqrt(self.a*self.b)*self.trial, self.test)*self.ds(1)
        self.class_bc_abc = class_bc_abc    # to make copies


    def set_fwd(self):  
        self.fwdadj = 1.0
        self.ftime = None

    def set_adj(self):  
        self.fwdadj = -1.0
        self.ftime = None

    def get_tt(self, nn):
        if self.fwdadj > 0.0:   return self.times[nn]
        else:   return self.times[-nn-1]


    def update(self, parameters_m):
        isprint = self.parameters['print']
        lumpM = self.parameters['lumpM']
        lumpD = self.parameters['lumpD']
        abc = self.parameters['abc']

        # Time options:
        if parameters_m.has_key('t0'):   self.t0 = parameters_m['t0'] 
        if parameters_m.has_key('tf'):   self.tf = parameters_m['tf'] 
        if parameters_m.has_key('Dt'):   self.Dt = parameters_m['Dt'] 
        if parameters_m.has_key('t0') or parameters_m.has_key('tf') or parameters_m.has_key('Dt'):
            self.Nt = int(round((self.tf-self.t0)/self.Dt))
            self.Tf = self.t0 + self.Dt*self.Nt
            self.times = np.linspace(self.t0, self.Tf, self.Nt+1)
            assert isequal(self.times[1]-self.times[0], self.Dt, 1e-16), "Dt modified"
            self.Dt = self.times[1] - self.times[0]
            assert isequal(self.Tf, self.tf, 1e-2), "Final time differs by more than 1%"
            if not isequal(self.Tf, self.tf, 1e-12) and isprint:
                print 'Final time modified from {} to {} ({}%)'.\
                format(self.tf, self.Tf, abs(self.Tf-self.tf)/self.tf)
        # Initial conditions:
        if parameters_m.has_key('u0init'):   self.u0init = parameters_m['u0init']
        if parameters_m.has_key('utinit'):   self.utinit = parameters_m['utinit']
        if parameters_m.has_key('u1init'):   self.u1init = parameters_m['u1init']
        if parameters_m.has_key('um1init'):   self.um1init = parameters_m['um1init']
        # Medium parameters:
        if parameters_m.has_key('b'):
            setfct(self.b, parameters_m['b'])
            if np.amin(self.b.vector().array()) < 1e-14:
                if isprint: print 'negative value for parameter b'
                sys.exit(1)
            if isprint: print 'assemble K',
            self.K = assemble(self.weak_k)
            if isprint: print ' -- K assembled'
        if parameters_m.has_key('a'):
            setfct(self.a, parameters_m['a'])
            if np.amin(self.a.vector().array()) < 1e-14:
                if isprint: print 'negative value for parameter a'
                sys.exit(1)
            # Mass matrix:
            if isprint: print 'assemble M',
            Mfull = assemble(self.weak_m)
            if lumpM:
                self.solverM = LumpedMatrixSolverS(self.V)
                self.solverM.set_operator(Mfull, self.bc)
                self.M = self.solverM
            else:
                self.solverM = PETScKrylovSolver('cg', 'jacobi')
                self.solverM.parameters['report'] = False
                self.solverM.parameters['nonzero_initial_guess'] = True
                if not self.bc == None: self.bc.apply(Mfull)
                self.solverM.set_operator(Mfull)
            if isprint: print ' -- M assembled'
        # Matrix D for abs BC
        if abc == True:    
            if isprint:    print 'assemble D',
            Mfull = assemble(self.weak_m)
            Dfull = assemble(self.weak_d)
            if lumpD:
                self.D = LumpedMatrixSolverS(self.V)
                self.D.set_operator(Dfull, None, False)
                if lumpM:
                    self.solverMplD = LumpedMatrixSolverS(self.V)
                    self.solverMplD.set_operators(Mfull, Dfull, .5*self.Dt, self.bc)
                    self.MminD = LumpedMatrixSolverS(self.V)
                    self.MminD.set_operators(Mfull, Dfull, -.5*self.Dt, self.bc)
            else:
                self.D = Dfull
            if isprint:    print ' -- D assembled'
        else:   self.D = 0.0


    #@profile
    def solve(self):
        timestepper = self.parameters['timestepper']
        isprint = self.parameters['print']

        # Set time-stepper:
        if timestepper == 'backward':
            def iterate(tt):  self.iteration_backward(tt)
        elif timestepper == 'centered':
            def iterate(tt):  self.iteration_centered(tt)
        else:
            if isprint: print "Time stepper not implemented"
            sys.exit(1)

        # Set boundary conditions:
        if self.bc == None:
            self.applybc = self.applybcNone
        else:
            self.applybc = self.applybcD

        if isprint:    print 'Compute solution'
        solout = [] # Store computed solution
        # u0:
        tt = self.get_tt(0)
        if isprint:    print 'Compute solution -- time {}'.format(tt)
        setfct(self.u0, self.u0init)
        solout.append([self.u0.vector().array(), tt])
        # Compute u1:
        if not self.u1init == None: self.u1 = self.u1init
        else:
            assert(not self.utinit == None)
            setfct(self.rhs, self.ftime(tt))
            self.rhs.vector().axpy(-self.fwdadj, self.D*self.utinit.vector())
            self.rhs.vector().axpy(-1.0, self.K*self.u0.vector())
            self.applybc(self.rhs.vector())
            self.sol.vector().zero()
            self.solverM.solve(self.sol.vector(), self.rhs.vector())
            setfct(self.u1, self.u0)
            self.u1.vector().axpy(self.fwdadj*self.Dt, self.utinit.vector())
            self.u1.vector().axpy(0.5*self.Dt**2, self.sol.vector())
        tt = self.get_tt(1)
        if isprint:    print 'Compute solution -- time {}'.format(tt)
        solout.append([self.u1.vector().array(), tt])
        # Iteration
        self.ptru0v = self.u0.vector()    # ptru* = 'pointers' to the u*'s
        self.ptru1v = self.u1.vector()
        self.ptru2v = self.u2.vector()
        config = 1
        for nn in xrange(2, self.Nt+1):
            iterate(tt)
            # Advance to next time step
            if config == 1:
                self.ptru0v = self.u1.vector()
                self.ptru1v = self.u2.vector()
                self.ptru2v = self.u0.vector()
                config = 2
            elif config == 2:
                self.ptru0v = self.u2.vector()
                self.ptru1v = self.u0.vector()
                self.ptru2v = self.u1.vector()
                config = 3
            else:
                self.ptru0v = self.u0.vector()
                self.ptru1v = self.u1.vector()
                self.ptru2v = self.u2.vector()
                config = 1
            tt = self.get_tt(nn)
            if isprint:    
                print 'Compute solution -- time {}'.format(tt)
            solout.append([self.ptru1v.array(),tt])
        if self.fwdadj > 0.0:   
            assert isequal(tt, self.Tf, 1e-16), \
            'tt={}, Tf={}, reldiff={}'.format(tt, self.Tf, abs(tt-self.Tf)/self.Tf)
        else:
            assert isequal(tt, self.t0, 1e-16), \
            'tt={}, t0={}, reldiff={}'.format(tt, self.t0, abs(tt-self.t0))
        return solout, self.computeerror()

    def iteration_centered(self, tt):
        self.rhs.vector().zero()
        self.rhs.vector().axpy(self.Dt*self.Dt, self.ftime(tt))
        self.rhs.vector().axpy(-1.0, self.MminD*self.ptru0v)
        self.rhs.vector().axpy(2.0, self.M*self.ptru1v)
        self.rhs.vector().axpy(-self.Dt**2, self.K*self.ptru1v)
        self.applybc(self.rhs.vector())
        self.solverMplD.solve(self.ptru2v, self.rhs.vector())

    #@profile
    def iteration_backward(self, tt):
        self.rhs.vector().zero()
        self.rhs.vector().axpy(self.Dt, self.ftime(tt))
        self.rhs.vector().axpy(-self.Dt, self.K*self.ptru1v)
        self.rhs.vector().axpy(-1.0, self.D*(self.ptru1v-self.ptru0v))
        self.applybc(self.rhs.vector())
        self.solverM.solve(self.sol.vector(), self.rhs.vector())
        self.ptru2v.zero()
        self.ptru2v.axpy(2.0, self.ptru1v)
        self.ptru2v.axpy(-1.0, self.ptru0v)
        self.ptru2v.axpy(self.Dt, self.sol.vector())


    def applybcNone(self, vect):
        pass

    def applybcD(self, vect):
        self.bc.apply(vect)


    def computeerror(self): 
        return self.computerelativeerror()

    def computerelativeerror(self):
        if not self.exact == None:
            MM = assemble(inner(self.trial, self.test)*dx)
            norm_ex = np.sqrt(\
            (MM*self.exact.vector()).inner(self.exact.vector()))
            diff = self.exact.vector() - self.u1.vector()
            if norm_ex > 1e-16: return np.sqrt((MM*diff).inner(diff))/norm_ex
            else:   return np.sqrt((MM*diff).inner(diff))
        else:   return []

    def computeabserror(self):
        if not self.exact == None:
            MM = assemble(inner(self.trial, self.test)*dx)
            diff = self.exact.vector() - self.u1.vector()
            return np.sqrt((MM*diff).inner(diff))
        else:   return []


