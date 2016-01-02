import sys
import numpy as np
from linalg.cgsolverSteihaug import CGSolverSteihaug

"""Library to solve optimization problem"""

def checkgradfd(ObjFctal, nbgradcheck=10, tolgradchk=1e-6):
    """
    Finite-difference check for the gradient of an ObjectiveFunctional object
        ObjFctal = object describing objective functional; must have methods:
            - getmarray: return medium parameter in np.array format
            - backup_m: create safe copy of current medium parameter
            - getMGarray: return Mass*gradient in np.array format
            - getmcopyarray: return copy of medium parameter in np.array format
            - update_m: update medium parameter from a np.array
            - solvefwd_cost: solve fwd pb and compute cost
    and member:
            - cost: value of cost function
    """
    lenm = len(ObjFctal.getmarray())
    ObjFctal.backup_m()
    rnddirc = np.random.randn(nbgradcheck, lenm)
    H = [1e-5, 1e-6, 1e-4]
    factor = [1.0, -1.0]
    MGdir = rnddirc.dot(ObjFctal.getMGarray())
    for textnb, dirct, mgdir in zip(range(lenm), rnddirc, MGdir):
        print 'Gradient check -- direction {0}: MGdir={1:.5e}'\
        .format(textnb+1, mgdir)
        for hh in H:
            cost = []
            for fact in factor:
                ObjFctal.update_m(ObjFctal.getmcopyarray() + fact*hh*dirct)
                ObjFctal.solvefwd_cost()
                cost.append(ObjFctal.cost)
            FDgrad = (cost[0] - cost[1])/(2.0*hh)
            err = abs(mgdir - FDgrad) / abs(FDgrad)
            if err < tolgradchk:   
                print '\th={0:.1e}: FDgrad={1:.5e}, error={2:.2e} -> OK!'\
                .format(hh, FDgrad, err)
                break
            else:
                print '\th={0:.1e}: FDgrad={1:.5e}, error={2:.2e}'\
                .format(hh, FDgrad, err)
    # Restore initial value of m:
    ObjFctal.restore_m()
    ObjFctal.solvefwd_cost()
    ObjFctal.solveadj_constructgrad()


def checkhessfd(ObjFctal, nbhesscheck=10, tolgradchk=1e-6):
    """Finite-difference check for the Hessian of an ObjectiveFunctional
    object"""
    lenm = len(ObjFctal.getmarray())
    ObjFctal.backup_m()
    rnddirc = np.random.randn(nbhesscheck, lenm)
    H = [1e-5, 1e-4, 1e-3]
    factor = [1.0, -1.0]
    hessxdir = ObjFctal.srchdir
    dirfct = ObjFctal.delta_m
    for textnb, dirct in zip(range(lenm), rnddirc):
        # Do computations for analytical Hessian:
        dirfct.vector()[:] = dirct
        ObjFctal.mult(dirfct.vector(), hessxdir.vector())
        normhess = np.linalg.norm(hessxdir.vector().array())
        print 'Hessian check -- direction {0}: ||H.x||={1:.5e}'\
        .format(textnb+1, normhess)
        # Do computations for FD Hessian:
        for hh in H:
            MG = []
            for fact in factor:
                ObjFctal.update_m(ObjFctal.getmcopyarray() + fact*hh*dirct)
                ObjFctal.solvefwd_cost()
                ObjFctal.solveadj_constructgrad()
                MG.append(ObjFctal.getMGarray())
            FDHessx = (MG[0] - MG[1])/(2.0*hh)
            # Compute errors:
            normFDhess = np.linalg.norm(FDHessx)
            err = np.linalg.norm(hessxdir.vector().array() - FDHessx)/\
            normhess
            if err < tolgradchk:   
                print '\th={0:.1e}: ||FDH.x||={1:.5e}, error={2:.2e} -> OK!'\
                .format(hh, np.linalg.norm(FDHessx), err)
                break
            else:
                print '\th={0:.1e}: ||FDH.x||={1:.5e}, error={2:.2e}'\
                .format(hh, np.linalg.norm(FDHessx), err)
    # Restore initial value of m:
    ObjFctal.restore_m()
    ObjFctal.solvefwd_cost()
    ObjFctal.solveadj_constructgrad()


def compute_searchdirection(ObjFctal, keyword, gradnorm_init=None, maxtolcg=0.5):
    """Compute search direction for Line Search based on keyword.
    keyword can be 'sd' (steepest descent) or 'Newt' (Newton's method).
    Whether we use full Hessian or GN Hessian in Newton's method depend on
    parameter ObjFctal.GN
    Inputs:
        ObjFctal = object for objective functional; should contain methods:
            - setsrchdir: set search direction from np.array
            - setgradxdir: set value gradient times search direction
            - getsrchdirarray: return search direction in np.array
            - getMGarray: return Mass*Gradient in np.array
            for Newton's method, should also contain methods:
            - getGradnorm: return norm of gradient
            - mult: matrix operation A.x from fenics class LinearOperator
            - getprecond: return preconditioner for computation Newton's step
            and members:
            - srchdir: search direction
            - MG: Mass*Gradient
        keyword = 'sd' or 'Newt'
        gradnorminit = norm of gradient at first step of iteration
        maxtolcg = max value for tol CG
    """
    if keyword == 'sd':
        ObjFctal.setsrchdir(-1.0*ObjFctal.getGradarray())
    elif keyword == 'Newt':
        # Compute tolcg for Inexact-CG Newton's method:
        gradnorm = ObjFctal.getGradnorm()
        gradnormrel = gradnorm/max(1.0, gradnorm_init)
        tolcg = min(maxtolcg, np.sqrt(gradnormrel))
        # Define solver
        solver = CGSolverSteihaug()
        solver.set_operator(ObjFctal)
        solver.set_preconditioner(ObjFctal.getprecond())
        solver.parameters["rel_tolerance"] = tolcg
        solver.parameters["zero_initial_guess"] = True
        solver.parameters["print_level"] = -1
        solver.solve(ObjFctal.srchdir.vector(), -ObjFctal.MG.vector())
    else:
        raise ValueError("Wrong keyword")
    ObjFctal.setgradxdir( np.dot(ObjFctal.getsrchdirarray(), \
    ObjFctal.getMGarray()) )
    if ObjFctal.getgradxdir() > 0.0: 
        raise ValueError("Search direction is not a descent direction")
        sys.exit(1)
    if keyword == 'Newt':
        return [solver.iter, solver.final_norm, solver.reasonid, tolcg]


def bcktrcklinesearch(ObjFctal, nbLS, alpha_init=1.0, rho=0.5, c=5e-5):
    """Run backtracking line search in 'search_direction'. 
    Default 'search_direction is steepest descent.
    'rho' is multiplicative factor for alpha.
    ObjFctal should contain methods:
        - backup_m
        - getcost
        - getsrchdirarray
        - getmcopyarray
        - update_m
        - solvefwd_cost
        - getgradxdir
    """
    # Check input parameters are correct:
    if c < 0. or c > 1.:    raise ValueError("c must be between 0 and 1")
    if rho < 0. or rho > 0.99:  
        raise ValueError("rho must be between 0 and 1")
    if alpha_init < 1e-16:    raise ValueError("alpha must be positive")
    # Prelim steps:
    ObjFctal.backup_m()
    cost_mk = ObjFctal.getcost()[0]
    LScount = 0
    success = False
    alpha = alpha_init
    srch_dir = ObjFctal.getsrchdirarray()
    # Start Line Search:
    while LScount < nbLS:
        LScount += 1
        ObjFctal.update_m(ObjFctal.getmcopyarray() + alpha*srch_dir)
        ObjFctal.solvefwd_cost()
        if ObjFctal.getcost()[0] < (cost_mk + alpha*c*ObjFctal.getgradxdir()):
            success = True
            break
        alpha *= rho
    return [success, LScount, alpha]

