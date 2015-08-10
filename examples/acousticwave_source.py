import sys
from os.path import splitext, isdir
from shutil import rmtree
import numpy as np

from fenicstools.plotfenics import PlotFenics
from dolfin import *

Nxy = 100
h = 1./Nxy
mesh = UnitSquareMesh(Nxy, Nxy, "crossed")
Dt = h/20.
tf = 5e-2

V = FunctionSpace(mesh, 'Lagrange', 1)
test = TestFunction(V)
trial = TrialFunction(V)
K = assemble(inner(nabla_grad(test), nabla_grad(trial))*dx)
M = assemble(inner(test, trial)*dx)
# BCs:
def u0_boundary(x, on_boundary):
    return (x[0] < 1e-16 or x[0] > 1.0-1e-16) and on_boundary
ubc = Constant("0.0")
bc = DirichletBC(V, ubc, u0_boundary)
bc.apply(M)
solver = LUSolver()
solver.parameters['reuse_factorization'] = True
solver.parameters['symmetric'] = True
solver.set_operator(M)
sol = []
foriginal = assemble(Constant('2')*test*dx)
f = Function(V)
def sourcef(tt):
    return tt*foriginal.array()
# Define plots:
filename, ext = splitext(sys.argv[0])
if isdir(filename + '/'):   rmtree(filename + '/')
myplot = PlotFenics(filename)
#myplot.set_varname('f')
# IC:
tt = 0.0
p0 = Function(V)
sol.append([p0.vector().array(), tt])
print 'time={}, max(p)={}, max(pex)={}'.format(tt, \
p0.vector().array().max(), 0.0)
# t1:
tt+=Dt
p1 = interpolate(Expression('x[0]*(1-x[0])*t', t=tt), V)
sol.append([p1.vector().array(), tt])
print 'time={}, max(p)={}, max(pex)={}'.format(tt, \
p1.vector().array().max(), (.5**2)*tt)
# iterate:
p2 = Function(V)
rhs = Function(V)
Kp = Function(V)
while tt < tf:
    f.vector()[:] = sourcef(tt)
    #myplot.plot_vtk(f, int(tt/Dt))
    Kp.vector()[:] = f.vector().array() - (K*p1.vector()).array()
    bc.apply(Kp.vector())
    solver.solve(rhs.vector(), Kp.vector())  # M^-1*K*p0
    p2.vector()[:] = 2*p1.vector().array() - p0.vector().array() + (Dt**2)*rhs.vector().array()
    tt+=Dt
    sol.append([p2.vector().array(), tt])
    print 'time={}, max(p)={}, max(pex)={}'.format(tt, \
    p2.vector().array().max(), (.5**2)*tt)
    # Update solutions:
    p0.vector()[:] = p1.vector().array()
    p1.vector()[:] = p2.vector().array()
#myplot.gather_vtkplots()
# Plot solutions:
myplot.set_varname('p')
plotp = Function(V)
for index, pp in enumerate(sol):
    plotp.vector()[:] = pp[0]
    myplot.plot_vtk(plotp, index)
myplot.gather_vtkplots()
