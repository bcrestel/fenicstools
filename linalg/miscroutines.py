import numpy as np
from dolfin import as_backend_type

def get_diagonal(M):
    """ Get diagonal of a square Matrix M and return a np.array"""

    return as_backend_type(M).mat().getDiagonal().array
#    assert M.size(0) == M.size(1), "M is not a square Matrix"
#
#    outp = np.zeros(M.size(0))
#    for ii in range(M.size(0)):
#        indices, values = M.getrow(ii)
#        try:
#            index = np.where(indices == ii)
#            outp[ii] = values[index]
#        except:
#            outp[ii] = 0.0
#    return outp
