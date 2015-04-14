import numpy as np
from dolfin import *
from fenicstools.objectivefunctional import ObjFctalElliptic
from fenicstools.plotfenics import PlotFenics
from fenicstools.optimsolver import checkgradfd, checkhessfd, bcktrcklinesearch

# Domain
mesh = UnitSquareMesh(12,12)
# Finite element spaces
V = FunctionSpace(mesh, 'Lagrange', 2)
Vm = FunctionSpace(mesh, 'Lagrange', 1)
Vme = FunctionSpace(mesh, 'Lagrange', 5)
# Boundary conditions
def u0_boundary(x, on_boundary):
    return on_boundary
u0 = Constant("0.0")
bc = DirichletBC(V, u0, u0_boundary)

# Compute target data
mtrue_exp = Expression('1 + 7*(pow(pow(x[0] - 0.5,2) +' + \
' pow(x[1] - 0.5,2),0.5) > 0.2)')
mtrue = interpolate(mtrue_exp, Vme)
normmtrue = norm(mtrue)
f = Expression("1.0")
goal = ObjFctalElliptic(V, Vme, bc, bc, [f])
goal.update_m(mtrue)
goal.solvefwd()
UD = goal.U
# Add noise
# TO BE DONE

# Set up optimization 
gamma =1e-10
InvPb = ObjFctalElliptic(V, Vm, bc, bc, [f], [], UD, gamma)
InvPb.update_m(1.0)
InvPb.solvefwd_cost()
cost, misfit, regul = InvPb.getcost()
print ('{:2s} {:12s} {:12s} {:12s} {:10s} {:6s} {:12s} {:8s} {:10s} {:10s}')\
.format('iter', 'cost', 'datamisfit', 'regul', 'medmisfit', 'rel', \
'||grad||', 'rel', 'angle', 'alpha')
medmisfit = errornorm(InvPb.m, mtrue, 'l2', 1)
print ('{:2d} {:12.5e} {:12.5e} {:12.5e} {:10.2e} {:6.3f}').format(0, \
cost, misfit, regul, medmisfit, medmisfit/normmtrue)
maxiter = 100 
alpha_init = 1e3
nbcheck = 4
nbLS = 20

# Iteration
for it in range(1, maxiter+1):
    InvPb.solveadj_constructgrad()
    InvPb.mult(InvPb.Grad.vector(), InvPb.delta_m.vector())
    if it == 1 or it % 20 == 0: 
        checkgradfd(InvPb, nbcheck)
        checkhessfd(InvPb, nbcheck)
    InvPb.set_searchdirection('sd')
    LSsuccess, LScount, alpha = bcktrcklinesearch(InvPb, nbLS, alpha_init)
    # Print results
    gradnorm = np.sqrt(np.dot(InvPb.getGradarray(), \
    InvPb.getMGarray()))
    if it == 1:   gradnorm_init = gradnorm
    gradnormrel = gradnorm/gradnorm_init
    srchdirnorm = np.sqrt(np.dot(InvPb.getsearchdirarray(), \
    (InvPb.MM*InvPb.getsearchdirarray())))
    medmisfit = errornorm(InvPb.getm(), mtrue, 'l2', 1)
    cost, misfit, regul = InvPb.getcost()
    print ('{:2d} {:12.5e} {:12.5e} {:12.5e} {:10.2e} {:6.3f} {:12.5e} ' + \
    '{:8.2e} {:10.3e} {:10.3e}').format(it, cost, misfit, regul, \
    medmisfit, medmisfit/normmtrue, gradnorm, \
    gradnormrel, InvPb.gradxdir/(gradnorm*srchdirnorm), alpha)
    # Stopping criteria:
    if not LSsuccess:
        print 'Line Search failed after {0} counts'.format(LScount)
        break
    if gradnormrel < 1e-10: 
        print "Optimization converged!"
        break
    # Set up next iteration
    if LScount == 1:    alpha_init = 10.*alpha
    elif LScount < 5:   alpha_init = 4.*alpha
    else:   alpha_init = alpha

if it == maxiter:   print "Max nb of iterations reached."
# Plot results
myplot = PlotFenics('poisson_SD/')
myplot.set_varname('mtrue')
myplot.plot_vtk(mtrue)
myplot.set_varname('m')
myplot.plot_vtk(InvPb.m)
