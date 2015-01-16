import abc
import numpy as np

from dolfin import *
from exceptionsfenics import WrongInstanceError
set_log_active(False)

class DataMisfitPart(LinearOperator):
    """
    Provides data misfit, gradient and Hessian information for the data misfit
    part of a time-independent symmetric inverse problem.
    """
    __metaclass__ = abc.ABCMeta

    # Instantiation
    def __init__(self, V, Vm, bc, RHSinput=[], Dr=[], UD=[], R=[], Data=[]):
        # Define test, trial and all other functions
        self.trial = TrialFunction(V)
        self.test = TestFunction(V)
        self.mtrial = TrialFunction(Vm)
        self.mtest = TestFunction(Vm)
        self.rhsadj = Function(V)
        self.m = Function(Vm)
        self.lenm = len(self.m.vector().array())
        self.delta_m = Function(Vm)
        self.MG = Function(Vm)
        self.Grad = Function(Vm)
        self.u = Function(V)
        self.ud = Function(V)
        self.p = Function(V)
        # Define weak forms to assemble A, C and E
        self._wkforma()
        self._wkformc()
        self._wkforme()
        # Store other info:
        self.Dr = Dr
        self.Ladj = - inner(self.u - self.ud, self.test)*dx
        self.UD = UD
        self.reset()
        self.Data = Data
        # Operators and bc
        LinearOperator.__init__(self, self.delta_m.vector(), \
        self.delta_m.vector()) 
        self.bc = bc
        self._assemble_solverM(Vm)
        self.assemble_A()
        self.assemble_RHS(RHSinput)
        self.W = assemble(inner(self.trial, self.test)*dx)
        self.assemble_R(R)
        # Counters, tolerances and others
        self.nbcheck = 10
        self.tolgradchk = 1e-6
        self.nbPDEsolves = 0

    def copy(self):
        """Define a copy method"""
        V = self.trial.function_space()
        Vm = self.mtrial.function_space()
        newobj = self.__class__(V, Vm, self.bc, [], self.Dr, self.UD, [],\
        self.Data)
        newobj.RHS = self.RHS
        newobj.R = self.R
        newobj.update_m(self.m)
        return newobj

    def mult(self, x, y):
        y[:] = np.zeros(self.lenm)
        for C in self.C:
            C.transpmult(x, self.rhsadj.vector())
            print self.rhsadj.vector().array()[:5]
            self.solve_A(self.u.vector(), -self.rhsadj.vector())
            print self.u.vector().array()[:5]
            self.solve_A(self.p.vector(), -(self.W * self.u.vector()))
            print self.p.vector().array()[:5]
            y[:] += (C*self.p.vector()).array()
        y[:] += (self.R * x).array()

    # Solve
    def solvefwd(self):
        """Solve fwd operators for given RHS"""
        for rhs in self.RHS:
            self.solve_A(self.u.vector(), rhs)
            if self.Dr == []:
                self.U.append(self.u.vector().array())
            else:
                self.U.append(self.Dr.dot(self.u.vector().array()))
            self.C.append(assemble(self.c))

    def solvefwd_cost(self):
        """Solve fwd operators for given RHS and compute cost"""
        self.misfit = 0.0
        for rhs, ud in zip(self.RHS, self.UD):
            self.solve_A(self.u.vector(), rhs)
            if self.Dr == []:
                u_vec_arr = self.u.vector().array()
                self.U.append(u_vec_arr)
                # WARNING: self.ud is here used for another meaning
                self.ud.vector()[:] = u_vec_arr - ud   
                self.misfit += np.dot(self.ud.vector().array(), \
                (self.W * self.ud.vector()).array())
            else:
                #self.U.append(self.Dr.dot(self.u.vector().array()))
                assert False
            self.C.append(assemble(self.c))
        self.misfit *= 0.5        
        self.regul = 0.5 * np.dot(self.m.vector().array(), \
        (self.R * self.m.vector()).array())
        self.cost = self.misfit + self.regul

    def solveadj_constructgrad(self):
        """Solve adj operators"""
        self.Nbsrc = len(self.UD)
        MG = np.zeros(self.lenm)
        for ii, C in enumerate(self.C):
            self.assemble_rhsadj(self.U[ii], self.UD[ii])
            self.solve_A(self.p.vector(), self.rhsadj.vector())
            self.E.append(assemble(self.e))
            MG += (C*self.p.vector()).array()
        self.MG.vector()[:] = MG/self.Nbsrc
        self.solverM.solve(self.Grad.vector(), self.MG.vector())

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
        if self.Dr == []:
            self.u.vector()[:] = U
            self.ud.vector()[:] = UD
        else:
            self.u.vector()[:] = self.Dr.transpose.dot(U)
            self.ud.vector()[:] = self.Dr.transpose.dot(UD)
        rhs = assemble(self.Ladj)
        self.bc.apply(rhs)
        self.rhsadj.vector()[:] = rhs.array()

    def _assemble_solverM(self, Vm):
        self.MM = assemble(inner(self.mtrial, self.mtest)*dx)
        self.solverM = LUSolver()
        self.solverM.parameters['reuse_factorization'] = True
        self.solverM.parameters['symmetric'] = True
        self.solverM.set_operator(self.MM)

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

    # Checks
    def checkgradfd(self):
        """Finite-difference check for the gradient"""
        FDobj = self.copy()
        rnddirc = np.random.randn(self.nbcheck, self.lenm)
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
                print '\th={0:.1e}: FDgrad={1:.5e}, error={2:.2e}'\
                .format(hh, FDgrad, err)
                if err < self.tolgradchk:   break

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
#class OperatorMass(OperatorPDE):
#    """
#    Operator for Mass matrix <u, v>
#    """
#    def _wkforma(self):
#        self.a = inner(self.trial, self.test)*dx 

class DataMisfitElliptic(DataMisfitPart):
    """
    Operator for elliptic equation div (m grad u)
    <m grad u, grad v>
    """
    def _wkforma(self):
        self.a = inner(self.m*nabla_grad(self.test), nabla_grad(self.trial))*dx

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
