import numpy as np
from fenicstools.sampling.distributions import SampleMultivariateNormal

dim_m = 4
m = np.linspace(1, dim_m, dim_m).reshape((dim_m, 1))
D = np.random.randn(dim_m**2).reshape((dim_m, dim_m))
U,S,VT = np.linalg.svd(D)
C = U.dot(np.diag(S**2)).dot(U.T)
C_cholesky = U.dot(np.diag(S))
assert np.linalg.norm(C - C_cholesky.dot(C_cholesky.T))/np.linalg.norm(C) < 1e-16

sample_size = 100000
m_emp = np.zeros((dim_m, 1))
C_emp = np.zeros((dim_m, dim_m))
for ii in range(sample_size):
    sample = SampleMultivariateNormal(m, C_cholesky).reshape((dim_m, 1))
    m_emp += sample
    C_emp += (sample-m).dot((sample-m).T)

m_emp /= (sample_size-1)
C_emp /= (sample_size-1)
print "m, m_emp, m-m_emp: "
print np.concatenate((m, m_emp, m-m_emp), axis=1)
print np.sqrt(np.sum((m_emp-m)**2))/np.sqrt(np.sum(m**2))

print "C, C_emp, C-C_emp: "
print C 
print C_emp 
print C-C_emp
print np.linalg.norm(C-C_emp)/np.linalg.norm(C)
