import sys
from os.path import splitext, isdir
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt

from fenicstools.plotfenics import PlotFenics
from fenicstools.acousticwave import AcousticWave
from dolfin import UnitSquareMesh, FunctionSpace, Constant, DirichletBC, \
interpolate, Expression, Function, SubDomain

NN = np.array((20, 50, 100, 200))
ERROR = []

tf = 0.5 # Final time
direction = 0
u0_expr = Expression(\
'100*pow(x[i]-.25,2)*pow(x[i]-0.75,2)*(x[i]<=0.75)*(x[i]>=0.25)', i=direction)
exact_expr = Expression(\
'50*pow(x[i]+.25,2)*pow(x[i]-0.25,2)*(x[i]<=0.25) + ' + \
'50*pow(x[i]-.75,2)*pow(x[i]-1.25,2)*(x[i]>=0.75)', i=direction)
class LeftRight(SubDomain):
    def inside(self, x, on_boundary):
        return (x[direction] < 1e-16 or x[direction] > 1.0 - 1e-16) \
        and on_boundary

for Nxy in NN:
    h = 1./Nxy
    print '\n\th = {}'.format(h)
    mesh = UnitSquareMesh(Nxy, Nxy, "crossed")
    q = 1   # Polynomial order
    V = FunctionSpace(mesh, 'Lagrange', q)
    Dt = h/(q*5.)

    Wave = AcousticWave({'V':V, 'Vl':V, 'Vr':V})
    Wave.set_abc(mesh, LeftRight())
    Wave.exact = interpolate(exact_expr, V)
    Wave.update({'lambda':1.0, 'rho':1.0, 't0':0.0, 'tf':tf, 'Dt':Dt,\
    'u0init':interpolate(u0_expr, V), 'utinit':Function(V)})
    sol, error = Wave.solve()
    ERROR.append(error)
    print 'relative error = {:.5e}'.format(error)

# Order of convergence:
CONVORDER = []
for ii in range(len(ERROR)-1):
    CONVORDER.append(np.log(ERROR[ii+1]/ERROR[ii])/np.log((1./NN[ii+1])/(1./NN[ii])))
print '\n\norder of convergence:', CONVORDER

# Save plots:
filename, ext = splitext(sys.argv[0])
if isdir(filename + '/'):   rmtree(filename + '/')
myplot = PlotFenics(filename)
myplot.set_varname('p')
plotp = Function(V)
for index, pp in enumerate(sol):
    plotp.vector()[:] = pp[0]
    myplot.plot_vtk(plotp, index)
myplot.gather_vtkplots()
# convergence plot:
fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(1./NN, ERROR, '-o')
ax.set_xlabel('h')
ax.set_ylabel('error')
fig.savefig(filename + '/convergence.eps')

