"""
Medium parameter reconstruction example for Poisson pb with zero-Dirichlet
boundary conditions and continuous observations over the entire domain.
We solve
arg min_m 1/2||u - u_e||^2 + R(m)
where - Delta u = f, on Omega
with u = 0 on boundary.
"""

import numpy as np
try:
    from dolfin import UnitSquareMesh, FunctionSpace, Constant, DirichletBC, \
    Expression, interpolate, MPI, mpi_comm_world
    mycomm = mpi_comm_world()
    myrank = MPI.rank(mycomm)
except:
    from dolfin import UnitSquareMesh, FunctionSpace, Constant, DirichletBC, \
    Expression, interpolate
    mycomm = None
    myrank = 0
from fenicstools.objectivefunctional import ObjFctalElliptic
from fenicstools.observationoperator import ObsEntireDomain
from fenicstools.prior import LaplacianPrior
from fenicstools.optimsolver import checkgradfd, checkhessfd, \
bcktrcklinesearch, compute_searchdirection
from fenicstools.postprocessor import PostProcessor


# Domain, f-e spaces and boundary conditions:
mesh = UnitSquareMesh(150,150)
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
f = Expression("1.0")

print 'p{}: Compute target data'.format(myrank)
noisepercent = 0.00   # e.g., 0.02 = 2% noise level
ObsOp = ObsEntireDomain({'V': V,'noise':noisepercent}, mycomm)
goal = ObjFctalElliptic(V, Vme, bc, bc, [f], ObsOp, [], [], [], False, mycomm)
goal.update_m(mtrue)
goal.solvefwd()
print 'p{}'.format(myrank)
UDnoise = goal.U

print 'p{}: Solve reconstruction problem'.format(myrank)
Regul = LaplacianPrior({'Vm':Vm,'gamma':1e-5,'beta':1e-14})
ObsOp.noise = False
InvPb = ObjFctalElliptic(V, Vm, bc, bc, [f], ObsOp, UDnoise, Regul, [], False, mycomm)
InvPb.update_m(1.0) # Set initial medium
InvPb.solvefwd_cost()
# Choose between steepest descent and Newton's method:
METHODS = ['sd','Newt']
meth = METHODS[1]
if meth == 'sd':    alpha_init = 1e3
elif meth == 'Newt':    alpha_init = 1.0
nbcheck = 0 # Grad and Hessian checks
nbLS = 20   # Max nb of line searches
# Prepare results outputs:
PP = PostProcessor(meth, Vm, mtrue, mycomm)
PP.getResults0(InvPb)    # Get results for index 0 (before first iteration)
PP.printResults(myrank)
# Start iteration:
maxiter = 50
for it in range(1, maxiter+1):
    InvPb.solveadj_constructgrad()
    # Check gradient and Hessian:
    if nbcheck and (it == 1 or it % 10 == 0): 
        checkgradfd(InvPb, nbcheck)
        checkhessfd(InvPb, nbcheck)
    # Compute search direction:
    if it == 1: gradnorm_init = InvPb.getGradnorm()
    if meth == 'Newt':
        if it == 1: maxtolcg = .5
        else:   maxtolcg = CGresults[3]
    else:   maxtolcg = None
    CGresults = compute_searchdirection(InvPb, meth, gradnorm_init, maxtolcg)
    # Compute line search:
    LSresults = bcktrcklinesearch(InvPb, nbLS, alpha_init)
    #InvPb.plotm(it) # Plot current medium reconstruction
    # Print results:
    PP.getResults(InvPb, LSresults, CGresults)
    PP.printResults(myrank)
    if PP.Stop(myrank):   break   # Stopping criterion
    alpha_init = PP.alpha_init()    # Initialize next alpha when using sd
#InvPb.gatherm() # Create one plot for all intermediate reconstructions
if it == maxiter and myrank == 0:   print "Max nb of iterations reached."