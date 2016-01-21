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


def checkdt(Dt, h, q, c_max, Mlump):
    """ Checks if Dt is sufficiently small based on some numerical tests """
    if Mlump:   alpha = 3.
    else:   alpha = 4.
    assert Dt <= h/(q*alpha*c_max), "Error: You need to choose a smaller Dt"

def checkdt_abc(Dt, h, q, c_max, Mlump, Dlump, timestepper):
    """ Checks if Dt is sufficiently small based on some numerical tests """
    if Mlump:
        if Dlump:
            if timestepper == 'centered':    alpha = 3.
            else:   alpha = 4.
        else:   alpha = 3.
    else:   alpha = 5.
    assert Dt <= h/(q*alpha*c_max), "Error: You need to choose a smaller Dt"
