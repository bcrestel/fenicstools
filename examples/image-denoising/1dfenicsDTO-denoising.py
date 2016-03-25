""" Fenics implementation of 1D denoising problem 
Note that we don't assemble a true kernel operator due to Fenics limitations,
instead we assemble some sort of an hybrid between a finite-difference
approximation and a weak form.
"""

import numpy as np
import dolfin as dl

from fenicstools.imagedenoising import *


N = 100
mesh = dl.UnitIntervalMesh(N)
gamma = 0.03
CC = 1/(gamma*np.sqrt(2*np.pi))
k_e = dl.Expression('C*exp(-pow(x[0]-t,2)/(2*pow(g,2)))', t=0., C=CC, g=gamma)
denoise = ObjectiveImageDenoising1D(mesh, k_e)
denoise.generatedata(0.2)
denoise.g = .5*np.abs(np.sin(np.pi*denoise.xx*2))
#denoise.test_gradient(g)
#denoise.test_hessian(g)
denoise.solve()
denoise.printout()
fig = denoise.plot()
plt.show()
