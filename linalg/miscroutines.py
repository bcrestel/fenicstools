import numpy as np

#TODO: To be tested
def get_diagonal(M):
    """ Get diagonal of a square Matrix M and return a np.array"""

    assert M.size(0) == M.size(1), "M is not a square Matrix"

    outp = np.zeros(M.size(0))
    for ii in range(M.size(0)):
        indices, values = M.getrow(ii)
        index = np.where(indices == ii)
        outp[ii] = values[index]
    return outp
