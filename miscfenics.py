from numpy import sqrt
from numpy.linalg import norm
from numpy.random import randn

from dolfin import Function, GenericVector, mpi_comm_world, MPI
from exceptionsfenics import WrongInstanceError

def apply_noise(UD, noisepercent, mycomm=mpi_comm_world()):
    """Apply Gaussian noise to data.
    noisepercent = 0.02 => 2% noise level, i.e.,
    || u - ud || / || ud || = || noise || / || ud || = 0.02"""
    UDnoise = []
    objnoise = 0.0
    for ud in UD:
        noisevect = randn(len(ud))
        # Get norm of entire random vector:
        normrand = sqrt(MPI.sum(mycomm, norm(noisevect)**2))
        noisevect /= normrand
        # Get norm of entire vector ud (not just local part):
        normud = sqrt(MPI.sum(mycomm, norm(ud)**2))
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
