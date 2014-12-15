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
    def __init__(self, V, Vm, bc, RHS, Dr=[], UD=[], R=[], Data=[]):
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
        self.RHS = RHS
        self.reset()
        self.Data = Data
        # Operators and bc
        LinearOperator.__init__(self, self.delta_m.vector(), self.delta_m.vector()) 
        self.bc = bc
        self._assemble_solverM(Vm)
        self.assemble_A()
        self.W = assemble(inner(self.trial, self.test)*dx)
        self.assemble_R(R)

    def mult(self, x, y):
        y[:] = np.zeros(self.lenm)
        for C in self.C:
            C.transpmult(x, self.rhsadj.vector())
            print self.rhsadj.vector().array()[:5]
            self.solver.solve(self.u.vector(), -self.rhsadj.vector())
            print self.u.vector().array()[:5]
            self.solver.solve(self.p.vector(), -(self.W * self.u.vector()))
            print self.p.vector().array()[:5]
            y[:] += (C*self.p.vector()).array()
        y[:] += (self.R * x).array()

    # Solve
    def solvefwd(self):
        """Solve fwd operators for given RHS"""
        for rhs in self.RHS:
            self.solver.solve(self.u.vector(), rhs)
            if self.Dr == []:
                self.U.append(self.u.vector().array())
            else:
                self.U.append(self.Dr.dot(self.u.vector().array()))
            self.C.append(assemble(self.c))

    def solveadj_constructgrad(self):
        """Solve adj operators"""
        self.Nbsrc = len(self.UD)
        MG = np.zeros(self.lenm)
        for ii, C in enumerate(self.C):
            self.assemble_rhsadj(self.U[ii], self.UD[ii])
            self.solver.solve(self.p.vector(), self.rhsadj.vector())
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

    def assemble_Ab(self, f):
        """Assemble operator A(m) and rhs b in symm way"""
        L = f*self.test*dx
        return assemble_system(self.a, L, self.bc)

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
        if not R == []:
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
        else:   raise WrongInstanceError('m should be Function or ndarray')
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
