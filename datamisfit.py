import abc
import numpy as np

from dolfin import *
from exceptionsfenics import WrongInstanceError
set_log_active(False)

#TODO: Move computations method to a different class (line search, gradient
# check,...)
# Test Hessian

class DataMisfitPart(LinearOperator):
    """
    Provides data misfit, gradient and Hessian information for the data misfit
    part of a time-independent symmetric inverse problem.
    """
    __metaclass__ = abc.ABCMeta

    # Instantiation
    def __init__(self, V, Vm, bc, bcadj, RHSinput=[], B=[], UD=[], R=[], \
    Data=[]):
        # Define test, trial and all other functions
        self.trial = TrialFunction(V)
        self.test = TestFunction(V)
        self.mtrial = TrialFunction(Vm)
        self.mtest = TestFunction(Vm)
        self.rhs = Function(V)
        self.m = Function(Vm)
        self.mcopy = Function(Vm)
        self.srchdir = Function(Vm)
        self.lenm = len(self.m.vector().array())
        self.delta_m = Function(Vm)
        self.MG = Function(Vm)
        self.Grad = Function(Vm)
        self.u = Function(V)
        self.ud = Function(V)
        self.diff = Function(V)
        self.p = Function(V)
        # Define weak forms to assemble A, C and E
        self._wkforma()
        self._wkformc()
        self._wkforme()
        # Store other info:
        self.B = B
        self.Ladj = - inner(self.u - self.ud, self.test)*dx
        self.UD = UD
        self.reset()
        self.Data = Data
        self.GN = 1.0
        # Operators and bc
        LinearOperator.__init__(self, self.delta_m.vector(), \
        self.delta_m.vector()) 
        self.bc = bc
        self.bcadj = bcadj
        self._assemble_solverM(Vm)
        self.assemble_A()
        self.assemble_RHS(RHSinput)
        self._assemble_W()
        self.assemble_R(R)
        # Counters, tolerances and others
        self.nbLS = 12
        self.nbgradcheck = 10
        self.tolgradchk = 1e-6
        self.nbPDEsolves = 0

    def copy(self):
        """Define a copy method"""
        V = self.trial.function_space()
        Vm = self.mtrial.function_space()
        newobj = self.__class__(V, Vm, self.bc, self.bcadj, [], self.B, \
        self.UD, [], self.Data)
        newobj.RHS = self.RHS
        newobj.R = self.R
        newobj.update_m(self.m)
        return newobj

    def obs(self, uin):
        """Apply observation operator based on self.B"""
        assert uin.__class__ == Function
        if self.B == []:   return uin.vector().array()
        else:   return self.B.dot(uin.vector().array())

    def mult(self, x, y):
        """mult(self, x, y): do y = Hessian * x
        member self.GN sets full Hessian (=1.0) or GN Hessian (=0.0)"""
        y[:] = np.zeros(self.lenm)
        for C, E in zip(self.C, self.E):
            # Solve for u_hat
            C.transpmult(x, self.rhs.vector())
            self.bcadj.apply(self.rhs.vector())
            self.solve_A(self.u.vector(), -self.rhs.vector())
            # Solve for phat
            E.transpmult(x, self.rhs.vector())
            self.rhs.vector()[:] *= -1.0 * self.GN
            self.rhs.vector()[:] += -(self.W * self.u.vector()).array()
            self.bcadj.apply(self.rhs.vector())
            self.solve_A(self.p.vector(), self.rhs.vector())
            # Compute Hessian*x:
            y[:] += (C*self.p.vector()).array() + \
            self.GN*(E*self.u.vector()).array()
        y[:] /= len(self.C)
        y[:] += (self.R * x).array()
            #print self.rhs.vector().array()[:5]
            #print self.u.vector().array()[:5]
            #print self.p.vector().array()[:5]

    # Solve
    def costfct(self, uin, udin):
        """Compute cost functional for 2 entries"""
        if self.W == []:
            return 0.5*np.dot(uin, udin)
        else:
            self.diff.vector()[:] = uin - udin
            return 0.5*np.dot(self.diff.vector().array(), \
            (self.W * self.diff.vector()).array())

    def solvefwd(self, cost=False):
        """Solve fwd operators for given RHS"""
        if cost:    self.misfit = 0.0
        for ii, rhs in enumerate(self.RHS):
            self.solve_A(self.u.vector(), rhs)
            u_obs = self.obs(self.u)
            self.U.append(u_obs)
            if cost:
                self.misfit += self.costfct(u_obs, self.UD[ii])
            self.C.append(assemble(self.c))
        if cost:
            self.misfit /= len(self.U)
            self.regul = 0.5 * np.dot(self.m.vector().array(), \
            (self.R * self.m.vector()).array())
            self.cost = self.misfit + self.regul

    def solvefwd_cost(self):
        """Solve fwd operators for given RHS and compute cost fct"""
        self.solvefwd(True)

    def solveadj(self, grad=False):
        """Solve adj operators"""
        self.Nbsrc = len(self.UD)
        if grad:    MG = np.zeros(self.lenm)
        for ii, C in enumerate(self.C):
            self.assemble_rhsadj(self.U[ii], self.UD[ii])
            self.solve_A(self.p.vector(), self.rhs.vector())
            self.E.append(assemble(self.e))
            if grad:    MG += (C*self.p.vector()).array()
        if grad:
            self.MG.vector()[:] = MG/self.Nbsrc + \
            (self.R * self.m.vector()).array()
            self.solverM.solve(self.Grad.vector(), self.MG.vector())

    def solveadj_constructgrad(self):
        """Solve adj operators and assemble gradient"""
        self.solveadj(True)

    # Assembler
    def assemble_A(self):
        """Assemble operator A(m)"""
        self.A = assemble(self.a)
        self.bc.apply(self.A)
        self.set_solver()

    def solve_A(self, b, f):
        """Solve system of the form A.b = f, 
        with b and f in form to be used in solver."""
        self.solver.solve(b, f)
        self.nbPDEsolves += 1

    def assemble_RHS(self, RHSin):
        """Assemble RHS for fwd solve"""
        if RHSin == []: self.RHS = None
        else:
            self.RHS = []
            for rhs in RHSin:
                if isinstance(rhs, Expression):
                    L = rhs*self.test*dx
                    b = assemble(L)
                    self.bc.apply(b)
                    self.RHS.append(b)
                else:   
                    raise WrongInstanceError("rhs should be Expression")

#    def assemble_Ab(self, f):
#        """Assemble operator A(m) and rhs b in symm way"""
#        L = f*self.test*dx
#        return assemble_system(self.a, L, self.bc)

    def assemble_rhsadj(self, U, UD):
        """Assemble rhs for adjoint equation"""
        self.diff.vector()[:] = U - UD
        self.rhs.vector()[:] = - (self.W * self.diff.vector()).array()
        self.bcadj.apply(self.rhs.vector())

    def _assemble_solverM(self, Vm):
        self.MM = assemble(inner(self.mtrial, self.mtest)*dx)
        self.solverM = LUSolver()
        self.solverM.parameters['reuse_factorization'] = True
        self.solverM.parameters['symmetric'] = True
        self.solverM.set_operator(self.MM)

    def _assemble_W(self):
        if self.B == []:
            self.W = assemble(inner(self.trial, self.test)*dx)
            #self.W = assemble(inner(self.trial, self.test)*dx, None, self.bc)
        else:   self.W = []

    def assemble_R(self, R):
        if R == []: self.R = None
        else:
            if isinstance(R, float):
                self.R = assemble(R * inner(nabla_grad(self.mtrial), \
                nabla_grad(self.mtest))*dx)
            else:
                self.R = R

    # Update param
    def update_Data(self, Data):
        """Update Data member"""
        self.Data = Data
        self.assemble_A()
        self.reset()

    def update_m(self, m):
        """Update values of parameter m"""
        if isinstance(m, Function):
            self.m.assign(m)
        elif isinstance(m, np.ndarray):
            self.m.vector()[:] = m
        elif isinstance(m, float):
            self.m.vector()[:] = m
        elif isinstance(m, int):
            self.m.vector()[:] = float(m)
        else:   raise WrongInstanceError('Format for m not accepted')
        self.assemble_A()
        self.reset()

    def backup_m(self):
        self.mcopy.assign(self.m)

    def reset(self):
        """Reset U, C and E"""
        self.U = []
        self.C = []
        self.E = []

    def set_solver(self):
        """Reset solver for fwd operator"""
        self.solver = LUSolver()
        self.solver.parameters['reuse_factorization'] = True
        self.solver.set_operator(self.A)

    def addPDEcount(self, increment):
        """Increase 'nbPDEsolves' by 'increment'"""
        self.nbPDEsolves += increment

    def resetPDEsolves(self):
        self.nbPDEsolves = 0

    # Computations
    def checkgradfd(self):
        """Finite-difference check for the gradient"""
        FDobj = self.copy()
        rnddirc = np.random.randn(self.nbgradcheck, self.lenm)
        H = [1e-5, 1e-4, 1e-3]
        factor = [1.0, -1.0]
        MGdir = rnddirc.dot(self.MG.vector().array())
        for textnb, dirct, mgdir in zip(range(self.lenm), rnddirc, MGdir):
            print 'Gradient check -- direction {0}: MGdir={1:.5e}'\
            .format(textnb+1, mgdir)
            for hh in H:
                cost = []
                for fact in factor:
                    FDobj.update_m(self.m.vector().array() + fact*hh*dirct)
                    FDobj.solvefwd_cost()
                    cost.append(FDobj.cost)
                FDgrad = (cost[0] - cost[1])/(2.0*hh)
                err = abs(mgdir - FDgrad) / abs(FDgrad)
                if err < self.tolgradchk:   
                    print '\th={0:.1e}: FDgrad={1:.5e}, error={2:.2e} -> OK!'\
                    .format(hh, FDgrad, err)
                    break
                else:
                    print '\th={0:.1e}: FDgrad={1:.5e}, error={2:.2e}'\
                    .format(hh, FDgrad, err)

    def set_searchdirection(self, keyword):
        """Set up search direction based on 'keyword'. 
        'keyword' can be: 'sd'."""
        if keyword == 'sd':
            self.srchdir.vector()[:] = -1.0*self.Grad.vector().array()
        self.gradxdir = np.dot(self.srchdir.vector().array(), \
        self.MG.vector().array())
        if self.gradxdir > 0.0: 
            raise ValueError("Search direction is not a descent direction")

    def bcktrcklinesearch(self, alpha_init=1.0, rho=0.5, c=5e-5):
        """Run backtracking line search in 'search_direction'. 
        Default 'search_direction is steepest descent.
        'rho' is multiplicative factor for alpha."""
        if c < 0. or c > 1.:    raise ValueError("c must be between 0 and 1")
        if rho < 0. or rho > 0.99:  
            raise ValueError("rho must be between 0 and 1")
        if alpha_init < 1e-16:    raise ValueError("alpha must be positive")
        self.backup_m()
        cost_mk = self.cost
        LScount = 0
        success = False
        alpha = alpha_init
        srch_dir = self.srchdir.vector().array()
        while LScount < self.nbLS:
            LScount += 1
            self.update_m(self.mcopy.vector().array() + alpha*srch_dir)
            self.solvefwd_cost()
            if self.cost < cost_mk + alpha * c * self.gradxdir: 
                success = True
                break
            alpha *= rho
        return success, LScount, alpha

    # Abstract methods
    @abc.abstractmethod
    def _wkforma(self):
        self.a = []

    @abc.abstractmethod
    def _wkformc(self):
        self.c = []

    @abc.abstractmethod
    def _wkforme(self):
        self.e = []


###########################################################
# Derived Classes
###########################################################

class DataMisfitElliptic(DataMisfitPart):
    """
    Operator for elliptic equation div (m grad u)
    <m grad u, grad v>
    """
    def _wkforma(self):
        self.a = inner(self.m*nabla_grad(self.trial), nabla_grad(self.test))*dx

    def _wkformc(self):
        self.c = inner(self.mtest*nabla_grad(self.u), nabla_grad(self.trial))*dx

    def _wkforme(self):
        self.e = inner(self.mtest*nabla_grad(self.p), nabla_grad(self.trial))*dx

#class OperatorHelmholtz(OperatorPDE):
#    """
#    Operator for Helmholtz equation
#    <grad u, grad v> - k^2 m u v
#    """
#    def _wkforma(self):
#        kk = self.Data['k']
#        self.a = inner(nabla_grad(self.trial), nabla_grad(self.test))*dx -\
#        inner(kk**2*(self.m)*(self.trial), self.test)*dx
