""" 1D example of image denoising using FD discretization of blurring operator
"""

import numpy as np
import matplotlib.pyplot as plt

# Grid
N = 100 + 1 # nb of grid points
#N = 10 + 1 # nb of grid points
xx = np.linspace(0.,1.,N)
h = xx[1] - xx[0]
# Blurring parameters
gamma = 0.03
C = 1/(gamma*np.sqrt(2*np.pi))
noise = 0.2
dd = np.array([[ii-jj for jj in xrange(N)] for ii in xrange(N)])
K = h*C*np.exp(-(h*dd)**2/(2*gamma**2))
# Target data:
f = 0.75*(xx>=.1)*(xx<=.25)
f += (xx>=0.28)*(xx<=0.3)*(15*xx-15*0.28)
f += (xx>0.3)*(xx<0.33)*0.3
f += (xx>=0.33)*(xx<=0.35)*(-15*xx+15*0.35)
f += (xx>=.5)*(xx-.5)**2*(xx-1.0)**2/.25**4
# data:
d = K.dot(f)
# noisy data:
sigma = noise * np.linalg.norm(d)/np.sqrt(N)
eta = sigma*np.random.randn(N)
dn = d + eta
print np.linalg.norm(eta)/np.linalg.norm(d), sigma**2
# Plot:
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xx, f, label='target')
ax.plot(xx, d, label='data')
ax.plot(xx, dn, '--', label='noisy data ('+str(noise)+')')
ax.legend(loc='best')
#
fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.plot(xx, f, label='target')
ax.plot(xx, np.linalg.solve(K,d), label='data')
#ax.plot(xx, np.linalg.solve(K,dn), '--', label='noisy data')
ax.legend(loc='best')
plt.show()
