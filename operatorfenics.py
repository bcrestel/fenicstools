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
    def __init__(self, V, Vm, bc, Data=[])
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
        self.update_A()

    @abc.abstractmethod
    def _set_Data(self, Data):
        self.Data = Data

    @abc.abstractmethod
    def _wkforma(self):
        self.a = []

    # Update param
    def update_Data(self, Data):
        self._set_Data(Data)
        self.update_A()

    def update_m(self, m):
        if isinstance(m, Function):
            self.m.assign(m)
        elif isinstance(m, np.ndarray):
            self.m.vector()[:] = m
        elif:   raise WrongInstanceError('m should be Function or ndarray')
        self.update_A()

    def update_A(self):
        self.A = assemble(self.a)
        (self.bc).apply(self.A)


###########################################################
# Derived Classes
###########################################################
class OperatorMass(OperatorPDE):
    """
    Operator A for Mass matrix
    Derived from class OperatorA
    """
    def _set_Data(self, Data):
        self.Data = Data

    def _wkforma(self):
        self.a = inner(self.trial, self.test)*dx 


###########################################################
class OperatorHelmholtz(OperatorPDE):
    """
    Operator A for Helmholtz equation
    Derived from class OperatorA
    """
    def _set_Data(self, Data):
        if not Data.has_key('k'):   raise WrongKeyInInputDataError
        self.Data = Data

    def _wkforma(self):
        kk = (self.Data)['k']
        self.a = inner(nabla_grad(self.trial), nabla_grad(self.test))*dx -\
        inner(kk**2*(self.m)*(self.trial), self.test)*dx
