"""
Medium parameter reconstruction example for Poisson pb with zero-Dirichlet
boundary conditions.
We solve
arg min_m 1/2||u - u_e||^2 + R(m)
where - Delta u = f, on Omega
with u = 0 on boundary.
"""

import numpy as np
from dolfin import *
from fenicstools.objectivefunctional import ObjFctalElliptic
from fenicstools.observationoperator import ObsEntireDomain
from fenicstools.regularization import TikhonovH1
from fenicstools.optimsolver import checkgradfd, checkhessfd, \
bcktrcklinesearch, compute_searchdirection
from fenicstools.miscfenics import apply_noise
from fenicstools.postprocessor import PostProcessor

# Domain, f-e spaces and boundary conditions:
mesh = UnitSquareMesh(20,20)
V = FunctionSpace(mesh, 'Lagrange', 2)  # space for state and adjoint variables
Vm = FunctionSpace(mesh, 'Lagrange', 1) # space for medium parameter
Vme = FunctionSpace(mesh, 'Lagrange', 5)    # sp for target med param
# Define zero Boundary conditions:
def u0_boundary(x, on_boundary):
    return on_boundary
u0 = Constant("0.0")
bc = DirichletBC(V, u0, u0_boundary)
# Define target medium and rhs:
mtrue_exp = Expression('1 + 7*(pow(pow(x[0] - 0.5,2) +' + \
' pow(x[1] - 0.5,2),0.5) > 0.2)')
mtrue = interpolate(mtrue_exp, Vme)
#normmtrue = norm(mtrue)
f = Expression("1.0")

# Compute target data:
ObsOp = ObsEntireDomain({'V': V})
goal = ObjFctalElliptic(V, Vme, bc, bc, [f], ObsOp, [], [], [], False)
goal.update_m(mtrue)
goal.solvefwd()
UD = goal.U
# Add noise:
noisepercent = 0.00   # e.g., 0.02 = 2% noise level
UDnoise, objnoise = apply_noise(UD, noisepercent)
print 'Noise in data misfit={:.5e}'.format(objnoise*.5/len(UD))

# Solve reconstruction problem:
Regul = TikhonovH1({'Vm':Vm,'gamma':1e-9,'beta':1e-14})
InvPb = ObjFctalElliptic(V, Vm, bc, bc, [f], ObsOp, UDnoise, Regul)
InvPb.update_m(1.0) # Set initial medium
InvPb.solvefwd_cost()
"""
cost, misfit, regul = InvPb.getcost()
print ('{:2s} {:12s} {:12s} {:12s} {:10s} {:6s} {:12s} {:8s} {:10s} {:10s}')\
.format('iter', 'cost', 'datamisfit', 'regul', 'medmisfit', 'rel', \
'||grad||', 'rel', 'angle', 'alpha')
medmisfit = errornorm(InvPb.m, mtrue, 'l2', 1)
print ('{:2d} {:12.5e} {:12.5e} {:12.5e} {:10.2e} {:6.3f}').format(0, \
cost, misfit, regul, medmisfit, medmisfit/normmtrue)
"""
maxiter = 100 
# Choose between steepest descent and Newton's method:
METHODS = ['sd','Newt']
meth = METHODS[1]
if meth == 'sd':    alpha_init = 1e3
elif meth == 'Newt':    alpha_init = 1.0
nbcheck = 4 # Grad and Hessian checks
nbLS = 20   # Max nb of line searches
PP = PostProcessor(meth, mtrue)
PP.getResults(InvPb,None,None,0)    # Get results for index 0 (before first iteration)
# Start iteration:
for it in range(1, maxiter+1):
    InvPb.solveadj_constructgrad()
    #InvPb.mult(InvPb.Grad.vector(), InvPb.delta_m.vector())
    # Check gradient and Hessian:
    if it == 1 or it % 10 == 0: 
        checkgradfd(InvPb, nbcheck)
        checkhessfd(InvPb, nbcheck)
    gradnorm = InvPb.getGradnorm()
    if it == 1:   gradnorm_init = gradnorm
    gradnormrel = gradnorm/max(1.0, gradnorm_init)
    tolcg = min(0.5, np.sqrt(gradnormrel))  # Inexact-CG-Newton's method
    CGresults = compute_searchdirection(InvPb, meth, tolcg)
    LSresults = bcktrcklinesearch(InvPb, nbLS, alpha_init)
    InvPb.plotm(it) # Plot current medium reconstruction
    # Print results
    srchdirnorm = InvPb.getsrchdirnorm()
    medmisfit = errornorm(InvPb.getm(), mtrue, 'l2', 1)
    cost, misfit, regul = InvPb.getcost()
    """
    print ('{:2d} {:12.5e} {:12.5e} {:12.5e} {:10.2e} {:6.3f} {:12.5e} ' + \
    '{:8.2e} {:10.3e} {:10.3e}').format(it, cost, misfit, regul, \
    medmisfit, medmisfit/normmtrue, gradnorm, \
    gradnormrel, InvPb.gradxdir/(gradnorm*srchdirnorm), alpha)
    """
    [LSsuccess, LScount, alpha] = LSresults
    # Stopping criteria:
    if not LSsuccess:
        print 'Line Search failed after {0} counts'.format(LScount)
        break
    if gradnormrel < 1e-10: 
        print "Optimization converged!"
        break
    # Set up next iteration
    if meth == 'sd':
        if LScount == 1:    alpha_init = 10.*alpha
        elif LScount < 5:   alpha_init = 4.*alpha
        else:   alpha_init = alpha
InvPb.gatherm() # Create one plot for all intermediate reconstructions

if it == maxiter:   print "Max nb of iterations reached."
