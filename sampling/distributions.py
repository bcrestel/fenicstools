import numpy as np

def SampleMultivariateNormal(m, C_cholesky):
    """ Sample from a multivariate normal distribution
    Inputs:
        m = mean of distribution
        C_cholesky = Cholesky factor of the covariance matrix, i.e., if we
        sample from N(m, C), we must have C = C_cholesky.C_cholesky^T.
    Outputs:
        sample = sample from N(m, C_cholesky.C_cholesky^T).
    If Z ~ N(0, I), then we return:
            sample = m + C_cholesky.Z
    One can check that E[sample] = m, and 
    E[(sample-m)(sample-m)^T] = E[C_cholesky.Z.Z^T.C_cholesky^T] = C """
    paramdim, col = m.shape
    assert paramdim == 1 or col == 1, [paramdim, col]
    dim_m = paramdim*col
    m_out = m.reshape((dim_m,1))
    Z = np.random.randn(dim_m).reshape((dim_m,1))
    return m_out + C_cholesky.dot(Z)
