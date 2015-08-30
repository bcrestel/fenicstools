import sys
from os.path import splitext, isdir
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt

from fenicstools.plotfenics import PlotFenics
from fenicstools.acousticwave import AcousticWave
from dolfin import UnitSquareMesh, FunctionSpace, Constant, DirichletBC, \
interpolate, Expression, Function

NN = np.array((25, 50, 100, 200))
ERROR = []

tf = 1./(8.*np.sqrt(2))  # Final time
u0_expr = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])')
def u0_boundary(x, on_boundary):
    return on_boundary
ubc = Constant("0.0")

for Nxy in NN:
    h = 1./Nxy
    print '\n\th = {}'.format(h)
    mesh = UnitSquareMesh(Nxy, Nxy, "crossed")
    q = 1   # Polynomial order
    V = FunctionSpace(mesh, 'Lagrange', q)
    Dt = h/(q*8.*np.sqrt(2))

    Wave = AcousticWave({'V':V, 'Vl':V, 'Vr':V})
    Wave.exact = Function(V)
    Wave.bc = DirichletBC(V, ubc, u0_boundary)
    Wave.update({'lambda':8.0, 'rho':2.0, 't0':0.0, 'tf':tf, 'Dt':Dt,\
    'u0init':interpolate(u0_expr, V), 'utinit':Function(V)})
    sol, error = Wave.solve()
    ERROR.append(error)
    print 'relative error = {:.5e}'.format(error)
CONVORDER = []
for ii in range(len(ERROR)-1):
    CONVORDER.append(np.log(ERROR[ii+1]/ERROR[ii])/np.log((1./NN[ii+1])/(1./NN[ii])))
print '\n\norder of convergence:', CONVORDER

# Save plots:
try:
    boolplot = int(sys.argv[1])
except:
    boolplot = 0
if boolplot > 0:
    filename, ext = splitext(sys.argv[0])
    if isdir(filename + '/'):   rmtree(filename + '/')
    myplot = PlotFenics(filename)
    myplot.set_varname('p')
    plotp = Function(V)
    for index, pp in enumerate(sol):
        plotp.vector()[:] = pp[0]
        myplot.plot_vtk(plotp, index)
    myplot.gather_vtkplots()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog(1./NN, ERROR, '-o')
    ax.set_xlabel('h')
    ax.set_ylabel('error')
    fig.savefig(filename + '/convergence.eps')

