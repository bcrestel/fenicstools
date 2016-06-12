"""
General functions for Fenics
"""

import numpy as np
from numpy import sqrt
from numpy.linalg import norm
from numpy.random import randn

try:
    from dolfin import Function, GenericVector, mpi_comm_world, MPI
except:
    from dolfin import Function, GenericVector
from exceptionsfenics import WrongInstanceError

def apply_noise(UD, noisepercent, mycomm=None):
    """ WARNING: SUPERCEDED BY CLASS OBSERVATIONOPERATOR
    Apply Gaussian noise to data.
    noisepercent = 0.02 => 2% noise level, i.e.,
    || u - ud || / || ud || = || noise || / || ud || = 0.02 """
    UDnoise = []
    objnoise = 0.0
    for ud in UD:
        noisevect = randn(len(ud))
        # Get norm of entire random vector:
        try:
            normrand = sqrt(MPI.sum(mycomm, norm(noisevect)**2))
        except:
            normrand = norm(noisevect)
        noisevect /= normrand
        # Get norm of entire vector ud (not just local part):
        try:
            normud = sqrt(MPI.sum(mycomm, norm(ud)**2))
        except:
            normud = norm(ud)
        noisevect *= noisepercent * normud
        objnoise += norm(noisevect)**2
        UDnoise.append(ud + noisevect)

    return UDnoise, objnoise


# Checkers
def isFunction(m_in):
    if not isinstance(m_in, Function):
     raise WrongInstanceError("input should be a Dolfin Function")

def isVector(m_in):
    if not isinstance(m_in, GenericVector):
     raise WrongInstanceError("input should be a Dolfin Generic Vector")

def isarray(uin):
    if not isinstance(uin, np.ndarray):
     raise WrongInstanceError("input should be a Numpy array")

def arearrays(uin, udin):
    if not (isinstance(uin, np.ndarray) and isinstance(udin, np.ndarray)):
     raise WrongInstanceError("inputs should be Numpy arrays")

def setfct(fct, value):
    isFunction(fct)
    if isinstance(value, np.ndarray):
        fct.vector()[:] = value
    elif isinstance(value, GenericVector):
        fct.vector().zero()
        fct.vector().axpy(1.0, value)
    elif isinstance(value, Function):
        setfct(fct, value.vector())
    elif isinstance(value, float):
        fct.vector()[:] = value
    elif isinstance(value, int):
        fct.vector()[:] = float(value)


def checkdt(Dt, h, r, c_max, Mlump):
    #TODO: To think about: make it a warning and check at every iteration of inversion?
    """ Checks if Dt is sufficiently small based on some numerical tests 
        Dt = time step size
        h = grid size
        r = polynomial order
        c_max = max wave speed in medium
        Mlump = bool value (lumped mass matrix)
    """
    if Mlump:   alpha = 3.
    else:   alpha = 4.
    upbnd = h/(r*alpha*c_max)
    assert Dt <= upbnd, 'Error: You need to choose Dt < {}'.format(upbnd)

def checkdt_abc(Dt, h, r, c_max, Mlump, Dlump, timestepper):
    """ Checks if Dt is sufficiently small based on some numerical tests 
        Dt = time step size
        h = grid size
        r = polynomial order
        c_max = max wave speed in medium
        Mlump = bool value (lumped mass matrix)
        Dlump = bool value (lumped damping matrix)
        timestepper = type of time stepping scheme
    """
    if Mlump:
        if Dlump:
            if timestepper == 'centered':    alpha = 3.
            else:   alpha = 4.
        else:   alpha = 3.
    else:   alpha = 5.
    assert Dt <= h/(r*alpha*c_max), "Error: You need to choose a smaller Dt"


def isequal(a, b, rtol=1e-14):
    """ Checks if 2 values are equal w/ relative tolerance """
    if abs(b) > 1e-16:  return np.abs(a-b) <= rtol*np.abs(b)
    else:   return np.abs(a-b) <= rtol

class ZeroRegularization():

    def cost(self, m_in):
        return 0.0

    def costab(self, ma_in, mb_int):  
        return self.cost(ma_in)

    def grad(self, m_in):
        out = m_in.copy(deepcopy=True)
        out.vector().zero()
        return out.vector()

    def gradab(self, ma_in, mb_in):  
        self.Va = ma_in.function_space()
        self.Vb = mb_in.function_space()
        u = Function(self.Va*self.Vb)
        u.vector().zero()
        return u.vector()

    def assemble_hessian(self, m_in):
        pass

    def hessian(self, mhat):
        out = mhat.copy()
        out.zero()
        return out

    def hessianab(self, ahat, bhat):
        u = Function(self.Va*self.Vb)
        u.vector().zero()
        return u.vector()
