"""
Medium parameter reconstruction example for Poisson pb with zero-Dirichlet
boundary conditions and continuous observations over the entire domain.
We solve
arg min_m 1/2||u - u_e||^2 + R(m)
where - Delta u = f, on Omega
with u = 0 on boundary.
"""

import numpy as np
from dolfin import UnitSquareMesh, FunctionSpace, Constant, DirichletBC, \
Expression, interpolate
from fenicstools.objectivefunctional import ObjFctalElliptic
from fenicstools.observationoperator import ObsEntireDomain
from fenicstools.prior import LaplacianPrior
from fenicstools.optimsolver import checkgradfd, checkhessfd, \
bcktrcklinesearch, compute_searchdirection
from fenicstools.miscfenics import apply_noise
from fenicstools.postprocessor import PostProcessor

def run_problem():
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
    f = Expression("1.0")

    # Compute target data:
    noisepercent = 0.05   # e.g., 0.02 = 2% noise level
    ObsOp = ObsEntireDomain({'V': V,'noise':noisepercent})
    goal = ObjFctalElliptic(V, Vme, bc, bc, [f], ObsOp)
    goal.update_m(mtrue)
    goal.solvefwd()
    UDnoise = goal.U

    # Solve reconstruction problem:
    Regul = LaplacianPrior({'Vm':Vm,'gamma':1e-3,'beta':1e-14})
    ObsOp.noise = False
    InvPb = ObjFctalElliptic(V, Vm, bc, bc, [f], ObsOp, UDnoise, Regul, [], False)
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
    PP = PostProcessor(meth, Vm, mtrue)
    PP.getResults0(InvPb)    # Get results for index 0 (before first iteration)
    PP.printResults()
    # Start iteration:
    maxiter = 100 
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
        InvPb.plotm(it) # Plot current medium reconstruction
        # Print results:
        PP.getResults(InvPb, LSresults, CGresults)
        PP.printResults()
        if PP.Stop():   break   # Stopping criterion
        alpha_init = PP.alpha_init()    # Initialize next alpha when using sd
    InvPb.gatherm() # Create one plot for all intermediate reconstructions
    if it == maxiter:   print "Max nb of iterations reached."

if __name__ == "__main__":
    run_problem()
