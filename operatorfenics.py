import abc
import numpy as np

from dolfin import *
from exceptionsfenics import WrongInstanceError
set_log_active(False)

class OperatorPDE:
    """
    Defines a linear operator A(m) in the Fenics module
    Methods:
        A = operator A(m)
        m = parameter
    """
    __metaclass__ = abc.ABCMeta

    # Instantiation
    def __init__(self, V, Vm, bc, Data=[]):
        # parameter & bc
        self.m = Function(Vm)
        self.bc = bc
        # Define test and trial functions
        self.trial = TrialFunction(V)
        self.test = TestFunction(V)
        # Add pb specific data
        self.Data = Data
        # Define weak form to assemble A
        self._wkforma()
        # Assemble PDE operator A 
        self.assemble_A()

    @abc.abstractmethod
    def _wkforma(self):
        self.a = []

    # Update param
    def update_Data(self, Data):
        self.Data = Data
        self.assemble_A()

    def update_m(self, m):
        if isinstance(m, Function):
            self.m.assign(m)
        elif isinstance(m, np.ndarray):
            self.m.vector()[:] = m
        else:   raise WrongInstanceError('m should be Function or ndarray')
        self.assemble_A()

    def assemble_A(self):
        self.A = assemble(self.a)
        self.bc.apply(self.A)

    def assemble_Ab(self, f):
        L = f*self.test*dx
        return assemble_system(self.a, L, self.bc)

    def assemble_b(self, f):
        L = f*self.test*dx
        b = assemble(L)
        self.bc.apply(b)
        return b


###########################################################
# Derived Classes
###########################################################
class OperatorMass(OperatorPDE):
    """
    Operator for Mass matrix <u, v>
    """
    def _wkforma(self):
        self.a = inner(self.trial, self.test)*dx 

class OperatorElliptic(OperatorPDE):
    """
    Operator for elliptic equation div (m grad u)
    <m grad u, grad v>
    """
    def _wkforma(self):
        self.a = inner(self.m*nabla_grad(self.trial), nabla_grad(self.test))*dx

class OperatorHelmholtz(OperatorPDE):
    """
    Operator for Helmholtz equation
    <grad u, grad v> - k^2 m u v
    """
    def _wkforma(self):
        kk = self.Data['k']
        self.a = inner(nabla_grad(self.trial), nabla_grad(self.test))*dx -\
        inner(kk**2*(self.m)*(self.trial), self.test)*dx
