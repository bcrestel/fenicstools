"""
Plot cost functional for joint inversion when varying magnitude of medium
perturbations for a and b
"""

import sys
from os.path import splitext, isdir
from shutil import rmtree
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import dolfin as dl
dl.set_log_active(False)

from fenicstools.objectiveacoustic import ObjectiveAcoustic
from fenicstools.jointregularization import Tikhonovab
from parametersinversion import parametersinversion

# parameters
a_target_fn, a_initial_fn, b_target_fn, b_initial_fn, wavepde, obsop = parametersinversion()
Vm = wavepde.Vm
V = wavepde.V
lenobspts = obsop.PtwiseObs.nbPts

# define objective function:
regul = Tikhonovab({'Vm':Vm,'gamma':5e-4,'beta':1e-8, 'm0':[1.0,1.0], 'cg':1.0})
waveobj = ObjectiveAcoustic(wavepde, 'ab', regul)
waveobj.obsop = obsop

# noisy data
print 'generate noisy data'
waveobj.solvefwd()
dd = waveobj.Bp.copy()
nbobspt, dimsol = dd.shape
noiselevel = 0.1   # = 10%
sigmas = np.sqrt((dd**2).sum(axis=1)/dimsol)*noiselevel
np.random.seed(11)
rndnoise = np.random.randn(nbobspt*dimsol).reshape((nbobspt, dimsol))
waveobj.dd = dd + sigmas.reshape((len(sigmas),1))*rndnoise

# vary medium and compute cost functional
na, nb = 43, 61
perturbalpha = np.linspace(-0.5, 0.9, na)
perturbbeta = np.linspace(-0.5, 1.5, nb)
COST = np.zeros((na, nb))
for indexa, aa in enumerate(perturbalpha):
    a_med = dl.Expression(\
    '1.0 + jj*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)', jj=aa)
    a_med_fn = dl.interpolate(a_med, Vm)
    waveobj.update_PDE({'a':a_med_fn})
    for indexb, bb in enumerate(perturbbeta):
        b_med = dl.Expression(\
        '1.0 + ii*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)', ii=bb)
        b_med_fn = dl.interpolate(b_med, Vm)
        waveobj.update_PDE({'b':b_med_fn})
        waveobj.solvefwd_cost()
        COST[indexa, indexb] = waveobj.cost_misfit

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(COST, aspect='equal')
ax.axis('equal')
fig.colorbar(im)
fig.savefig('inversionab-localminima.pdf')
np.savetxt('inversionab-localminima.txt', COST)
