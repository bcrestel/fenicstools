from numpy import sqrt
from numpy.linalg import norm
from numpy.random import randn

try:
    from dolfin import Function, GenericVector, mpi_comm_world, MPI
except:
    from dolfin import Function, GenericVector
from exceptionsfenics import WrongInstanceError

def apply_noise(UD, noisepercent, mycomm=None):
    """Apply Gaussian noise to data.
    noisepercent = 0.02 => 2% noise level, i.e.,
    || u - ud || / || ud || = || noise || / || ud || = 0.02"""
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
     raise WrongInstanceError("m_in should be a Dolfin Function")

def isVector(m_in):
    if not isinstance(m_in, GenericVector):
     raise WrongInstanceError("m_in should be a Dolfin Generic Vector")
