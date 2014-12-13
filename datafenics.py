import numpy as np
import copy

from dolfin import *
from exceptionsfenics import WrongInstanceError


#class PbData:
#    """Store problem-specific data for medium parameter reconstruction
#    Contains:
#        mesh
#        V, Vm
#        UD
#        m0, mexact
#        src & rec locations
#        pb-specific parameters (e.g, freq for Helmholtz pb)
#        RHS for fwd problem
#    """
#    def __init__(self, inputfile):
#    """Instantiate object from specially written input file"""


class CompData:
    """Stores elements needed to solve medium parameter reconstruction in
    PDE-based formulation.
    Contains:
        U, UD = current solution (with given parameter m) and target data
        m = current medium parameter
        C, E = matrices containing linearized version of the Jacobian wrt fwd
        state variables u_i and adj variables p_i
        solver = used to solve fwd (and adj in case of symm operator) A(m) u = f
    """
    def __init__(self, m = [], Vm = []):
        """If m_in is not [] but Vm == [], then m_in must be defined as a 
        Function(Vm)"""
        if isinstance(Vm, FunctionSpace):
            if not isinstance(m, np.ndarray):
                raise WrongInstanceError("m must be a numpy.ndarray")
            self.m = Function(Vm)
            self.m.vector()[:] = m
        elif Vm == []:
            if isinstance(m, Function):
                self.m = m.copy(deepcopy=True)
            elif m == []: self.m = []
            else:
                raise WrongInstanceError("m must be a Function")
        else:
            raise WrongInstanceError("Vm must be a FunctionSpace")
        self.C = []
        self.E = []
        self.U = []

    def copy(self):
        """Deep copy of an other object CompData"""
        outobj = self.__class__(self.m)
        outobj.U = copy.deepcopy(self.U)
        return outobj
