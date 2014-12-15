import abc
import numpy as np

from dolfin import *
from exceptionsfenics import WrongInstanceError
set_log_active(False)

class DataMisfitPart:
    """
    Provides data misfit, gradient and Hessian information for the data misfit
    part of a time-independent symmetric inverse problem.
    """
    __metaclass__ = abc.ABCMeta

    # Instantiation
    def __init__(self, V, Vm, bc, RHS, Dr=[], UD=[], Data=[]):
        # parameter & bc
        self.m = Function(Vm)
        self.lenm = len(self.m.vector().array())
        self.mtest = TestFunction(Vm)
        self.bc = bc
        self.MG = Function(Vm)
        self.Grad = Function(Vm)
        self._assemble_solverM(Vm)
        # Define test and trial functions
        self.trial = TrialFunction(V)
        self.test = TestFunction(V)
        self.rhsadj = Function(V)
        # Define u and p for C and E
        self.u = Function(V)
        self.ud = Function(V)
        self.p = Function(V)
        # Store other info:
        self.Dr = Dr
        self.Ladj = - inner(self.u - self.ud, self.test)*dx
        self.UD = UD
        self.RHS = RHS
        # Add pb specific data
        self.Data = Data
        # Define weak forms to assemble A, C and E
        self._wkforma()
        self._wkformc()
        self._wkforme()
        # Assemble PDE operator A 
        self.assemble_A()
        # initialize other members:
        self.reset()

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
        mtrial = TrialFunction(Vm)
        self.MM = assemble(inner(mtrial, self.mtest)*dx)
        self.solverM = LUSolver()
        self.solverM.parameters['reuse_factorization'] = True
        self.solverM.parameters['symmetric'] = True
        self.solverM.set_operator(self.MM)

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
