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

from fenicstools.jointregularization import crossgradient


def createExpression(center, size):
    x1, x2 = center[0]-0.5*size[0], center[0]+0.5*size[0]
    y1, y2 = center[1]-0.5*size[1], center[1]+0.5*size[1]
    scaling = (0.5*size[0])**2 * (0.5*size[1])**2
    mystring = '1.0+(x2-x[0])*(x[0]-x1)*(y2-x[1])*(x[1]-y1)*(x[0]<=x2)*(x[0]>=x1)*(x[1]<=y2)*(x[1]>=y1)/scaling'
    return dl.Expression(mystring, x1=x1, x2=x2, y1=y1, y2=y2, scaling=scaling)

# parameters
mesh = dl.UnitSquareMesh(100, 100)
Vm = dl.FunctionSpace(mesh, 'Lagrange', 1)

cg = crossgradient({'Vm':Vm})

a_med = createExpression([0.5,0.5], [0.4, 0.4])
a_med_fn = dl.interpolate(a_med, Vm)

# translate medium
location = np.linspace(0.0, 1.0, 50)
cgtranslate = np.zeros(50)
for index, ii in enumerate(location):
    b_med = createExpression([ii, 0.5], [0.4, 0.4])
    b_med_fn = dl.interpolate(b_med, Vm)
    cgtranslate[index] = cg.costab(a_med_fn, b_med_fn)
fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(cgtranslate)

# stretch medium
size = np.linspace(0.05, 0.9, 50)
cgstretch = np.zeros(50)
for index, ii in enumerate(location):
    b_med = createExpression([0.5,0.5], [ii, ii])
    b_med_fn = dl.interpolate(b_med, Vm)
    cgtranslate[index] = cg.costab(a_med_fn, b_med_fn)
ax2 = fig.add_subplot(212)
ax2.plot(cgtranslate)

fig.savefig('inversionab-crossgradient.pdf')
