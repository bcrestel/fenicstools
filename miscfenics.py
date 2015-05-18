from numpy.linalg import norm
from numpy.random import randn

from dolfin import Function, GenericVector
from exceptionsfenics import WrongInstanceError

def apply_noise(UD, noisepercent):
    """Apply Gaussian noise to data.
    noisepercent = 0.02 => 2% noise level, i.e.,
    || u - ud || / || ud || = || noise || / || ud || = 0.02"""
    UDnoise = []
    objnoise = 0.0
    for ud in UD:
        noisevect = randn(len(ud))
        noisevect = noisevect / norm(noisevect)
        noisevect *= noisepercent * norm(ud)
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
