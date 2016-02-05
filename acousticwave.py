import sys
import numpy as np

try:
    from dolfin import TestFunction, TrialFunction, Function, GenericVector, \
    assemble, inner, nabla_grad, dx, ds, LUSolver, KrylovSolver, sqrt, \
    PointSource, Point, Constant, FacetFunction, Measure, MPI, mpi_comm_world
    mycomm = mpi_comm_world()
    myrank = MPI.rank(mycomm)
    mpisize = MPI.size(mycomm)
except:
    from dolfin import TestFunction, TrialFunction, Function, GenericVector, \
    assemble, inner, nabla_grad, dx, ds, LUSolver, KrylovSolver, sqrt, \
    PointSource, Point, Constant, FacetFunction, Measure
    mycomm = None
    myrank = 0
    mpisize = 1

from miscfenics import isFunction, isVector, setfct
from linalg.lumpedmatrixsolver import LumpedMatrixSolverS


class AcousticWave():
    """
    Solution of forward and adjoint equations for acoustic inverse problem
    """

    def __init__(self, functionspaces_V):
        """
        Input:
            functionspaces_V = dict containing functionspaces for state/adj
        ('V') and med param ('Vl' for lambda and 'Vr' for rho)
        """
        self.readV(functionspaces_V)
        self.verbose = False    # print info
        self.lump = False   # Lump the mass matrix
        self.lumpD = False   # Lump the ABC matrix
        self.timestepper = None # 'backward', 'centered'
        self.exact = None   # exact solution at final time
        self.u0init = None  # provides u(t=t0)
        self.utinit = None  # provides u_t(t=t0)
        self.u1init = None  # provides u1 = u(t=t0+/-Dt)
        self.bc = None
        self.abc = False
        self.ftime = lambda t: 0.0  # ftime(tt) = src term @ t=tt (in np.array())
        self.set_fwd()  # default is forward problem


    def copy(self):
        """(hard) copy constructor"""
        newobj = self.__class__({'V':self.V, 'Vl':self.Vl, 'Vr':self.Vr})
        newobj.lump = self.lump
        newobj.timestepper = self.timestepper
        newobj.exact = self.exact
        newobj.utinit = self.utinit
        newobj.u1init = self.u1init
        newobj.bc = self.bc
        if self.abc == True:
            newobj.set_abc(self.V.mesh(), self.class_bc_abc, self.lumpD)
        newobj.ftime = self.ftime
        newobj.update({'lambda':self.lam, 'rho':self.rho, \
        't0':self.t0, 'tf':self.tf, 'Dt':self.Dt, \
        'u0init':self.u0init, 'utinit':self.utinit, 'u1init':self.u1init})
        return newobj


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


    def set_abc(self, mesh, class_bc_abc, lumpD=False):
        self.abc = True # False means zero-Neumann all-around
        if lumpD:    self.lumpD = True
        abc_boundaryparts = FacetFunction("size_t", mesh)
        class_bc_abc.mark(abc_boundaryparts, 1)
        self.ds = Measure("ds")[abc_boundaryparts]
        self.weak_d = inner(sqrt(self.lam*self.rho)*self.trial, 
        self.test)*self.ds(1)
        self.class_bc_abc = class_bc_abc    # to make copies


    def set_fwd(self):  self.fwdadj = 1.0
    def set_adj(self):  
        self.fwdadj = -1.0
        self.ftime = None


    def update(self, parameters_m):
        assert not self.timestepper == None, "You need to set a time stepping method"
        # Time options:
        if parameters_m.has_key('t0'):   self.t0 = parameters_m['t0'] 
        if parameters_m.has_key('tf'):   self.tf = parameters_m['tf'] 
        if parameters_m.has_key('Dt'):   self.Dt = parameters_m['Dt'] 
        # Initial conditions:
        if parameters_m.has_key('u0init'):   self.u0init = parameters_m['u0init']
        if parameters_m.has_key('utinit'):   self.utinit = parameters_m['utinit']
        if parameters_m.has_key('u1init'):   self.u1init = parameters_m['u1init']
        if parameters_m.has_key('um1init'):   self.um1init = parameters_m['um1init']
        # Medium parameters:
        setfct(self.lam, parameters_m['lambda'])
        if self.verbose: print 'lambda updated '
        if self.elastic == True:    
            setfct(self.mu, parameters_m['mu'])
            if self.verbose: print 'mu updated'
        if self.verbose: print 'assemble K',
        self.K = assemble(self.weak_k)
        if self.verbose: print ' -- K assembled'
        if parameters_m.has_key('rho'):
            setfct(self.rho, parameters_m['rho'])
            # Mass matrix:
            if self.verbose: print 'rho updated\nassemble M',
            Mfull = assemble(self.weak_m)
            if self.lump:
                self.solverM = LumpedMatrixSolverS(self.V)
                self.solverM.set_operator(Mfull, self.bc)
                self.M = self.solverM
            else:
                if mpisize == 1:
                    self.solverM = LUSolver()
                    self.solverM.parameters['reuse_factorization'] = True
                    self.solverM.parameters['symmetric'] = True
                else:
                    self.solverM = KrylovSolver('cg', 'amg')
                    self.solverM.parameters['report'] = False
                self.M = Mfull
                if not self.bc == None: self.bc.apply(Mfull)
                self.solverM.set_operator(Mfull)
            if self.verbose: print ' -- M assembled'
        # Matrix D for abs BC
        if self.abc == True:    
            if self.verbose:    print 'assemble D',
            Dfull = assemble(self.weak_d)
            if self.lumpD:
                self.D = LumpedMatrixSolverS(self.V)
                self.D.set_operator(Dfull, None, False)
                if self.lump:
                    self.solverMplD = LumpedMatrixSolverS(self.V)
                    self.solverMplD.set_operators(Mfull, Dfull, .5*self.Dt, self.bc)
                    self.MminD = LumpedMatrixSolverS(self.V)
                    self.MminD.set_operators(Mfull, Dfull, -.5*self.Dt, self.bc)
            else:
                self.D = Dfull
            if self.verbose:    print ' -- D assembled'
        else:   self.D = 0.0


    #@profile
    def solve(self):
        """ Wrapper for default way to solve """
        if self.timestepper == 'backward':  return self.solve_backward()
        elif self.timestepper == 'centered':    return self.solve_centered()
        else:
            print "Time stepper not implemented"
            sys.exit(1)


    def solve_centered(self):
        if self.verbose:    print 'Compute solution'
        solout = [] # Store computed solution
        # u0:
        if self.fwdadj > 0: tt = self.t0 
        else:   tt = self.tf
        if self.verbose:    print 'Compute solution -- time {}'.format(tt)
        setfct(self.u0, self.u0init)
        solout.append([self.u0.vector().array(), tt])
        # Compute u1:
        if not self.u1init == None: self.u1 = self.u1init
        else:
            assert(not self.utinit == None)
            setfct(self.rhs, self.ftime(tt))
            self.rhs.vector().axpy(-self.fwdadj, self.D*self.utinit.vector())
            self.rhs.vector().axpy(-1.0, self.K*self.u0.vector())
            if not self.bc == None: self.bc.apply(self.rhs.vector())
            self.solverM.solve(self.sol.vector(), self.rhs.vector())
            setfct(self.u1, self.u0)
            self.u1.vector().axpy(self.fwdadj*self.Dt, self.utinit.vector())
            self.u1.vector().axpy(0.5*self.Dt**2, self.sol.vector())
        tt += self.fwdadj*self.Dt
        if self.verbose:    print 'Compute solution -- time {}'.format(tt)
        solout.append([self.u1.vector().array(), tt])
        # Iteration
        if self.fwdadj > 0.:    target = self.tf*(1.0 + 1e-12)
        else:   
            if abs(self.t0) < 1e-14:    target = self.t0 - 1e-12
            else:   target = self.t0*(1.0 - 1e-12)
        while self.fwdadj*(tt + self.fwdadj*self.Dt) < target:
            self.rhs.vector()[:] = (self.Dt**2)*self.ftime(tt)
            self.rhs.vector().axpy(-1.0, self.MminD*self.u0.vector())
            self.rhs.vector().axpy(2.0, self.M*self.u1.vector())
            self.rhs.vector().axpy(-self.Dt**2, self.K*self.u1.vector())
            if not self.bc == None: self.bc.apply(self.rhs.vector())
            self.solverMplD.solve(self.u2.vector(), self.rhs.vector())
            # Advance to next time step
            setfct(self.u0, self.u1)
            setfct(self.u1, self.u2)
            tt += self.fwdadj*self.Dt
            if self.verbose:    
                print 'Compute solution -- time {}, rhs {}'.\
                format(tt, np.max(np.abs(self.ftime(tt))))
            solout.append([self.u1.vector().array(),tt])
        if self.fwdadj > 0.:    timeerror = abs(tt - self.tf)/self.tf
        else:    
            if abs(self.t0) < 1e-14:    timeerror = abs(tt - self.t0)
            else:   timeerror = abs(tt - self.t0)/self.t0
        if timeerror > 1e-12:
            raise RuntimeError('Final time is {} instead of {}'.format(tt, \
            self.tf), 'Relative error is {}'.format(timeerror))
        return solout, self.computeerror()


    def solve_backward(self):
        if self.verbose:    print 'Compute solution'
        solout = [] # Store computed solution
        # u0:
        if self.fwdadj > 0: tt = self.t0 
        else:   tt = self.tf
        if self.verbose:    print 'Compute solution -- time {}'.format(tt)
        setfct(self.u0, self.u0init)
        solout.append([self.u0.vector().array(), tt])
        # Compute u1:
        if not self.u1init == None: self.u1 = self.u1init
        else:
            assert(not self.utinit == None)
            setfct(self.rhs, self.ftime(tt))
            self.rhs.vector().axpy(-self.fwdadj, self.D*self.utinit.vector())
            self.rhs.vector().axpy(-1.0, self.K*self.u0.vector())
            if not self.bc == None: self.bc.apply(self.rhs.vector())
            self.solverM.solve(self.sol.vector(), self.rhs.vector())
            setfct(self.u1, self.u0)
            self.u1.vector().axpy(self.fwdadj*self.Dt, self.utinit.vector())
            self.u1.vector().axpy(0.5*self.Dt**2, self.sol.vector())
        tt += self.fwdadj*self.Dt
        if self.verbose:    print 'Compute solution -- time {}'.format(tt)
        solout.append([self.u1.vector().array(), tt])
        # Iteration
        # TODO: re-think about that; should include Dt
        if self.fwdadj > 0.:    target = self.tf*(1.0 + 1e-12)
        else:   
            if abs(self.t0) < 1e-14:    target = self.t0 - 1e-12
            else:   target = self.t0*(1.0 - 1e-12)
        while self.fwdadj*(tt + self.fwdadj*self.Dt) < target:
            #self.rhs.vector()[:] = self.Dt*self.ftime(tt)
            setfct(self.rhs, self.Dt*self.ftime(tt))
            self.rhs.vector().axpy(-self.Dt, self.K*self.u1.vector())
            self.rhs.vector().axpy(-1.0, self.D*(self.u1.vector()-self.u0.vector()))
            if not self.bc == None: self.bc.apply(self.rhs.vector())
            self.solverM.solve(self.sol.vector(), self.rhs.vector())
            self.u2.vector()[:] = 0.0
            self.u2.vector().axpy(2.0, self.u1.vector())
            self.u2.vector().axpy(-1.0, self.u0.vector())
            self.u2.vector().axpy(self.Dt, self.sol.vector())
            # Advance to next time step
            setfct(self.u0, self.u1)
            setfct(self.u1, self.u2)
            tt += self.fwdadj*self.Dt
            if self.verbose:    
                print 'Compute solution -- time {}, max(rhs) {}, min(rhs) {}'.format(\
                tt, np.max(self.ftime(tt)), np.min(self.ftime(tt)))
            solout.append([self.u1.vector().array(),tt])
        if self.fwdadj > 0.:    timeerror = abs(tt - self.tf)/self.tf
        else:    
            if abs(self.t0) < 1e-14:    timeerror = abs(tt - self.t0)
            else:   timeerror = abs(tt - self.t0)/self.t0
        if timeerror > 1e-12:
            raise RuntimeError('Final time is {} instead of {}'.format(tt, \
            self.tf), 'Relative error is {}'.format(timeerror))
        return solout, self.computeerror()


    def computeerror(self):
        if not self.exact == None:
            #MM = assemble(inner(self.trial, self.test)*dx)
            MM = self.M
            norm_ex = np.sqrt(\
            (MM*self.exact.vector()).inner(self.exact.vector()))
            diff = self.exact.vector() - self.u1.vector()
            if norm_ex > 1e-16: return np.sqrt((MM*diff).inner(diff))/norm_ex
            else:   return np.sqrt((MM*diff).inner(diff))
        else:   return []


