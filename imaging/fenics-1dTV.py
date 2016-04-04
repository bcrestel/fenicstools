""" Fenics implementation of 1D denoising problem 
Note that we don't assemble a true kernel operator due to Fenics limitations,
instead we assemble some sort of an hybrid between a finite-difference
approximation and a weak form.
"""

import numpy as np
import dolfin as dl
import matplotlib.pyplot as plt

from fenicstools.imagedenoising import *


N = 100
mesh = dl.UnitIntervalMesh(N)
gamma = 0.03
CC = 1/(gamma*np.sqrt(2*np.pi))
k_e = dl.Expression('C*exp(-pow(x[0]-t,2)/(2*pow(g,2)))', t=0., C=CC, g=gamma)
denoise = ObjectiveImageDenoising1D(mesh, k_e, 'TV')
denoise.generatedata(0.05)
denoise.g = .5*np.abs(np.sin(4*np.pi*denoise.xx))\
*(denoise.xx>=.25)*(denoise.xx<=.75)
#denoise.g = .5*np.sin(np.pi*denoise.xx)
fig = denoise.plot()
plt.show()
denoise.update_reg(10**(-1.5))
#denoise.test_gradient(denoise.g)
#denoise.test_hessian(denoise.g)
denoise.solve()
fig = denoise.plot()
plt.show()
