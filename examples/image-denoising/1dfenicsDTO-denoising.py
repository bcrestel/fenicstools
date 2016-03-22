""" Fenics implementation of 1D denoising problem 
Note that we don't assemble a true kernel operator due to Fenics limitations,
instead we assemble some sort of an hybrid between a finite-difference
approximation and a weak form.
"""

import numpy as np
import dolfin as dl

# Grid
N = 100
mesh = dl.UnitIntervalMesh(N)
V = dl.FunctionSpace(mesh, 'Lagrange', 1)
# kernel operator
gamma = 0.03
CC = 1/(gamma*np.sqrt(2*np.pi))
k_e = dl.Expression('C*exp(-pow(x[0]-t,2)/(2*pow(g,2)))', t=0., C=CC, g=gamma)
# data
xx = V.dofmap().tabulate_all_coordinates(mesh)
f = 0.75*(xx>=.1)*(xx<=.25)
f += (xx>=0.28)*(xx<=0.3)*(15*xx-15*0.28)
f += (xx>0.3)*(xx<0.33)*0.3
f += (xx>=0.33)*(xx<=0.35)*(-15*xx+15*0.35)
f += (xx>=.5)*(xx-.5)**2*(xx-1.0)**2/.25**4
ff, gg = dl.Function(V), dl.Function(V)
ff.vector()[:] = f
# operator K
test = dl.TestFunction(V)
kernel = dl.inner(k_e, test)*dl.dx
K = np.zeros((V.dim(),V.dim()))
for ii, tt in enumerate(xx):
    k_e.t = tt
    K[ii,:] = dl.assemble(kernel).array()
gg.vector()[:] = K.dot(ff.vector().array())
dl.plot(ff)
dl.plot(gg)
dl.interactive()
