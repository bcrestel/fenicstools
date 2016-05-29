import numpy as np
from dolfin import as_backend_type, PETScVector

def get_diagonal(M):
    """ Get diagonal of a square Matrix M and return a np.array"""

    return PETScVector(as_backend_type(M).mat().getDiagonal())
