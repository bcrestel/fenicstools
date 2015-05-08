"""
Medium parameter reconstruction example for Poisson pb with zero-Dirichlet
boundary conditions and continuous observations over the entire domain.
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
# Choose between steepest descent and Newton's method:
METHODS = ['sd','Newt']
meth = METHODS[0]
if meth == 'sd':    alpha_init = 1e3
elif meth == 'Newt':    alpha_init = 1.0
nbcheck = 0 # Grad and Hessian checks
nbLS = 20   # Max nb of line searches
# Prepare results outputs:
PP = PostProcessor(meth, mtrue)
PP.getResults0(InvPb)    # Get results for index 0 (before first iteration)
PP.printResults()
# Start iteration:
maxiter = 100 
for it in range(1, maxiter+1):
    InvPb.solveadj_constructgrad()
    # Check gradient and Hessian:
    if it == 1 or it % 10 == 0: 
        checkgradfd(InvPb, nbcheck)
        checkhessfd(InvPb, nbcheck)
    # Compute search direction:
    if it == 1:   gradnorm_init = InvPb.getGradnorm()
    CGresults = compute_searchdirection(InvPb, meth, gradnorm_init)
    # Compute line search:
    LSresults = bcktrcklinesearch(InvPb, nbLS, alpha_init)
    InvPb.plotm(it) # Plot current medium reconstruction
    # Print results:
    PP.getResults(InvPb, LSresults, CGresults)
    PP.printResults()
    if PP.Stop():   break   # Stopping criterion
    alpha_init = PP.alpha_init()    # Initialize next alpha when using sd
InvPb.gatherm() # Create one plot for all intermediate reconstructions
if it == maxiter:   print "Max nb of iterations reached."
