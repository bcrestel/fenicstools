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
denoise = ObjectiveImageDenoising1D(mesh, k_e)
denoise.generatedata(0.2)
#g = .5*np.abs(np.sin(np.pi*denoise.xx*2))
#denoise.test_gradient(g)
#denoise.test_hessian(g)
"""
GAMMA = 10**(-np.linspace(0.0, 5.0, 100))
RMM = []
COST = []
for gamma in GAMMA:
    print 'gamma={:.2e}'.format(gamma),
    denoise.update_reg(gamma)
    denoise.solve()
    denoise.printout()
    RMM.append(denoise.relmedmisfit)
    COST.append(denoise.cost)
fig = plt.figure(); ax=fig.add_subplot(111); ax.semilogx(GAMMA, RMM);
ax.set_label('gamma'); ax.set_ylabel('rel med misfit')
fig2 = plt.figure(); ax=fig2.add_subplot(111); ax.semilogx(GAMMA, COST);
ax.set_label('gamma'); ax.set_ylabel('objective (=misfit+reg)')
"""
denoise.update_reg(10**(-1.5))
denoise.solve()
denoise.printout()
fig3 = denoise.plot()
plt.show()
