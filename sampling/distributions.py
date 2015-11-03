import numpy as np

######################### Multivariate Gaussian ##############################
def SampleMultivariateNormal(m, C_cholesky, nbsamples=1):
    """ Sample from a multivariate normal distribution
    Inputs:
        m = mean of distribution
        C_cholesky = Cholesky factor of the covariance matrix, i.e., if we
        sample from N(m, C), we must have C = C_cholesky.C_cholesky^T.
        nbsamples = number of samples from distribution
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
    Z = np.random.randn(dim_m*nbsamples).reshape((dim_m, nbsamples))
    return m_out.dot(np.ones(nbsamples).reshape((1, nbsamples))) + C_cholesky.dot(Z)


################################ Wishart #####################################
def Wishart(m, C_cholesky):
    """ Sample from Wishart distribution with paramters (m, Sigma),
    where Sigma = C_cholesky . C_cholesky^T.
    Inputs:
        m, C_cholesky = parameters of distribution where Sigma = C_ch . C_ch^T
    Outputs:
        sample = sample from W(m, C_ch.C_ch^T).
    A sample from W(m, C_ch.C_ch^T) is 
        sum_{i=1}^m z_i z_i^T, where z_i ~ N(0, C_h) 
    One can check that E[sample] = m C_ch.C_ch^T """
    d = C_cholesky.shape[0]
    assert m > d, "You need to have m >= d+1"
    Gauss_samples = SampleMultivariateNormal(np.zeros((d,1)), C_cholesky, m)
    return Gauss_samples.dot(Gauss_samples.T)

def Wishart_fromSigmaInverse(m, Sigma_inv):
    """ Sample from Wishart distribution W(m, Sigma) where Sigma_inv = inv(Sigma)
    Inputs:
        m = parameter
        Sigma_inv = inv(Sigma)
    Ouputs:
        sample = sample from Wishart distribution W(m, inv(Sigma_inv)) """
    U, Ssq, UT = np.linalg.svd(Sigma_inv)
    checkvalue = np.linalg.norm(U.T-UT)
    assert checkvalue < 1e-14, 'Check if Sigma_inv is SPD (err={0})'.format(checkvalue)
    C_ch = U.dot(np.diag(1/np.sqrt(Ssq)))
    return Wishart(m, C_ch)

