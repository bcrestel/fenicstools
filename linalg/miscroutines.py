import numpy as np
from dolfin import as_backend_type, PETScVector

def get_diagonal(M):
    """ Extract diagonal of a square Matrix M and return a PETScVector """

    return PETScVector(as_backend_type(M).mat().getDiagonal())
