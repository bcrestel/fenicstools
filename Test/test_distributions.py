import numpy as np
from fenicstools.sampling.distributions import SampleMultivariateNormal, \
Wishart, Wishart_fromSigmaInverse

dim_m = 4
m = np.linspace(1, dim_m, dim_m).reshape((dim_m, 1))
D = np.random.randn(dim_m**2).reshape((dim_m, dim_m))
U,S,VT = np.linalg.svd(D)
C = U.dot(np.diag(S**2)).dot(U.T)
invC = U.dot(np.diag(1/S**2)).dot(U.T)
C_cholesky = U.dot(np.diag(S))
checkvalue = np.linalg.norm(C - C_cholesky.dot(C_cholesky.T))/np.linalg.norm(C)
assert checkvalue < 2e-16, checkvalue
checkvalue2 = np.linalg.norm(C.dot(invC) - np.eye(dim_m))
assert checkvalue2 < 1e-12, checkvalue2

####################### Multivariate Gaussian ##################
sample_size = 100000
samples = SampleMultivariateNormal(m, C_cholesky, sample_size)
m_emp = samples.sum(axis=1).reshape((dim_m,1))/sample_size
m_one = m.dot(np.ones((1,sample_size)))
C_emp = ((samples-m_one).dot((samples-m_one).T))/sample_size
print "Test Multivariate Gaussian:"
print "m, m_emp, m-m_emp: "
print np.concatenate((m, m_emp, m-m_emp), axis=1)
print np.sqrt(np.sum((m_emp-m)**2))/np.sqrt(np.sum(m**2))
print "C, C_emp, C-C_emp: "
print C 
print C_emp 
print C-C_emp
print np.linalg.norm(C-C_emp)/np.linalg.norm(C)

########################## Wishart #############################
sample_size = 10000
sample = np.zeros((dim_m, dim_m))
n = dim_m+10
for ii in range(sample_size):
    sample += Wishart(n, C_cholesky)
C_emp = sample/sample_size
print "\nTest Wishart:"
print C_emp
print n*C
print C_emp - n*C
print np.linalg.norm(n*C-C_emp)/np.linalg.norm(n*C)

sample = np.zeros((dim_m, dim_m))
for ii in range(sample_size):
    sample += Wishart_fromSigmaInverse(n, invC)
C_emp = sample/sample_size
print "\nTest Wishart_fromSigmaInverse:"
print C_emp
print n*C
print C_emp - n*C
print np.linalg.norm(n*C-C_emp)/np.linalg.norm(n*C)
