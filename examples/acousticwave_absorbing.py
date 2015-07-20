"""
Solve forward acoustic wave equation with dashpot absorbing boundary conditions
"""

import sys
from os.path import splitext, isdir
from shutil import rmtree
import numpy as np

from dolfin import UnitSquareMesh, FunctionSpace, Expression, interpolate, \
TestFunction, TrialFunction, assemble, dx, Function
from fenicstools.pdesolver import Wave
from fenicstools.plotfenics import PlotFenics

invh = 200
h = 1./invh
Dt = h/10. # Safeguard for CFL condition
tf = 600*Dt
print 'h={}, Dt={}, tf={}'.format(h, Dt, tf)
mesh = UnitSquareMesh(invh, invh)
V = FunctionSpace(mesh, 'Lagrange', 1)  # space for solution
Vl = FunctionSpace(mesh, 'Lagrange', 1) # space for medium param lambda and rho
# Define exact solution:
test = TestFunction(V)
trial = TrialFunction(V)
MM = assemble(test*trial*dx)
diff = Function(V)
def exact(tt, p):
    pex_exp = \
    Expression('(pow(pow(x[0],2)+pow(x[1],2),0.5)=={})/(4*pi*pow(pow(x[0],2)+pow(x[1],2),0.5))'.format(tt))
    pex = interpolate(pex_exp, V)
    normpex = np.sqrt((MM*pex.vector()).inner(pex.vector()))
    diff.vector()[:] = (pex.vector() - p.vector()).array()
    return np.sqrt((MM*diff.vector()).inner(diff.vector()))/normpex

# Define problem and solve:
PWave = Wave({'V':V, 'Vl':Vl, 'Vr':Vl})
PWave.verbose = True
PWave.update({'lambda':1.0, 'rho':1.0, 't0':0.0, 'tf':tf, 'Dt':Dt})
PWave.definesource({'type':'delta', 'point':[.5,.5]}, \
lambda t: float(np.abs(t)<1e-14))
#PWave.exact = exact
Pout = PWave.solve(np.linspace(0.0, tf, 11))

# Plot solutions
filename, ext = splitext(sys.argv[0])
if isdir(filename + '/'):   rmtree(filename + '/')
myplot = PlotFenics(filename + '/Fwd/')
myplot.set_varname('p_n')
plotp = Function(V)
for index, pp in enumerate(Pout):
    plotp.vector()[:] = pp[1]
    myplot.plot_vtk(plotp, index)
myplot.gather_vtkplots()
# Plot source
myplot.set_varname('f')
plotp.vector()[:] = PWave.f.array()
myplot.plot_vtk(plotp)
# Plot medium parameters
plotpar = Function(Vl)
myplot.set_varname('lam')
plotpar.vector()[:] = PWave.lam.vector().array()
myplot.plot_vtk(plotpar)
myplot.set_varname('rho')
plotpar.vector()[:] = PWave.rho.vector().array()
myplot.plot_vtk(plotpar)
