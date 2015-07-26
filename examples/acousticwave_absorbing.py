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
import matplotlib.pyplot as plt

invh = 200
h = 1./invh
Dt = h/10. # Safeguard for CFL condition
tf = 200*Dt
print 'h={}, Dt={}, tf={}'.format(h, Dt, tf)
mesh = UnitSquareMesh(invh, invh)
V = FunctionSpace(mesh, 'Lagrange', 1)  # space for solution
Vl = FunctionSpace(mesh, 'Lagrange', 1) # space for med param lambda and rho
# Define exact solution:
test = TestFunction(V)
trial = TrialFunction(V)
MM = assemble(test*trial*dx)
diff = Function(V)

# Define problem and solve:
timestamp = lambda t: float(np.abs(t)<.1)*(t*(0.1-t)/0.0025)
srcloc = [0.5,0.5]
PWave = Wave({'V':V, 'Vl':Vl, 'Vr':Vl})
PWave.verbose = True
PWave.update({'lambda':1.0, 'rho':1.0, 't0':0.0, 'tf':tf, 'Dt':Dt}) # c=1.0
PWave.definesource({'type':'delta', 'point':srcloc}, timestamp)
# Define exact solution at final time tf:
pex_Exp = Expression(\
'(fabs(Tf-sqrt(pow(x[0]-SRCx,2)+pow(x[1]-SRCy,2)))<0.1)'\
+'*(Tf-sqrt(pow(x[0]-SRCx,2)+pow(x[1]-SRCy,2)))'\
+'*(0.1-(Tf-sqrt(pow(x[0]-SRCx,2)+pow(x[1]-SRCy,2))))/0.025'\
+' * 1 / (4*pi*sqrt(pow(x[0]-0.5,2)+pow(x[1]-0.5,2)))',\
Tf=tf, SRCx=srcloc[0], SRCy=srcloc[1])
pexact = interpolate(pex_Exp, V)
PWave.exact = pexact
outtime = np.linspace(0.0, tf, 101)
Pout, normerr = PWave.solve(outtime)
print normerr
#print 'Relative error at time={}: {}',format(tf, normerr)

######## PLOTS #########
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
# Plot exact solution
myplot.set_varname('p_ex')
plotp.vector()[:] = pexact.vector().array()
myplot.plot_vtk(plotp)
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
# Plot timestamp
tt = np.linspace(0.0,tf,int(tf/Dt))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tt, map(timestamp,tt))
fig.savefig(filename + '/Fwd/timestamp.eps')
