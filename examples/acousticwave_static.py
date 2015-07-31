import sys
from os.path import splitext, isdir
from shutil import rmtree
import numpy as np

from fenicstools.plotfenics import PlotFenics
from dolfin import *

Nxy = 100
h = 1./Nxy
mesh = UnitSquareMesh(Nxy, Nxy, "crossed")
Dt = h/(4.*np.sqrt(2))
tf = 1./(2*np.sqrt(2))

V = FunctionSpace(mesh, 'Lagrange', 1)
test = TestFunction(V)
trial = TrialFunction(V)
K = assemble(inner(nabla_grad(test), nabla_grad(trial))*dx)
M = assemble(inner(test, trial)*dx)
# BCs:
def u0_boundary(x, on_boundary):
    return on_boundary
ubc = Constant("0.0")
bc = DirichletBC(V, ubc, u0_boundary)
bc.apply(M)
solver = LUSolver()
solver.parameters['reuse_factorization'] = True
solver.parameters['symmetric'] = True
solver.set_operator(M)
sol = []
# IC:
tt = 0.0
p0 = interpolate(Expression('sin(pi*x[0])*sin(pi*x[1])'), V)
sol.append([p0.vector().array(), tt])
print 'time={}, max(p)={}, min(p)={}'.format(tt, \
p0.vector().array().max(), p0.vector().array().min())
# t1:
tt+=Dt
p1 = Function(V)
p1.vector()[:] = p0.vector().array()
rhs = Function(V)
Kp = K * p0.vector()   # K*p0
bc.apply(Kp)
solver.solve(rhs.vector(), Kp)  # M^-1*K*p0
p1.vector().axpy(-.5*Dt**2, rhs.vector())   
sol.append([p1.vector().array(), tt])
print 'time={}, max(p)={}, min(p)={}'.format(tt, \
p1.vector().array().max(), p1.vector().array().min())
# iterate:
p2 = Function(V)
while tt < tf:
    tt+=Dt
    p2.vector()[:] = 2*p1.vector().array() - p0.vector().array()
    Kp = K * p1.vector()   # K*p1
    bc.apply(Kp)
    solver.solve(rhs.vector(), Kp)  # M^-1*K*p0
    p2.vector().axpy(-Dt**2, rhs.vector())   
    sol.append([p2.vector().array(), tt])
    print 'time={}, max(p)={}, min(p)={}'.format(tt, \
    p2.vector().array().max(), p2.vector().array().min())
    # Update solutions:
    p0.vector()[:] = p1.vector().array()
    p1.vector()[:] = p2.vector().array()
# Define plots:
filename, ext = splitext(sys.argv[0])
if isdir(filename + '/'):   rmtree(filename + '/')
myplot = PlotFenics(filename)
myplot.set_varname('p')
plotp = Function(V)
for index, pp in enumerate(sol):
    plotp.vector()[:] = pp[0]
    myplot.plot_vtk(plotp, index)
myplot.gather_vtkplots()
