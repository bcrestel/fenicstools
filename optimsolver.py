import numpy as np
import objectivefunctional
from pylib.cgsolverSteihaug import CGSolverSteihaug
from dolfin import *

"""Contains functions used to solve optimization problem"""

def checkgradfd(ObjFctal, nbgradcheck=10, tolgradchk=1e-6):
    """Finite-difference check for the gradient of an ObjectiveFunctional
    object"""
    FDobj = ObjFctal.copy()
    lenm = len(FDobj.getmarray())
    ObjFctal.backup_m()
    rnddirc = np.random.randn(nbgradcheck, lenm)
    H = [1e-5, 1e-4, 1e-3]
    factor = [1.0, -1.0]
    MGdir = rnddirc.dot(ObjFctal.getMGarray())
    for textnb, dirct, mgdir in zip(range(lenm), rnddirc, MGdir):
        print 'Gradient check -- direction {0}: MGdir={1:.5e}'\
        .format(textnb+1, mgdir)
        for hh in H:
            cost = []
            for fact in factor:
                FDobj.update_m(ObjFctal.getmcopyarray() + fact*hh*dirct)
                FDobj.solvefwd_cost()
                cost.append(FDobj.cost)
            FDgrad = (cost[0] - cost[1])/(2.0*hh)
            err = abs(mgdir - FDgrad) / abs(FDgrad)
            if err < tolgradchk:   
                print '\th={0:.1e}: FDgrad={1:.5e}, error={2:.2e} -> OK!'\
                .format(hh, FDgrad, err)
                break
            else:
                print '\th={0:.1e}: FDgrad={1:.5e}, error={2:.2e}'\
                .format(hh, FDgrad, err)

def checkhessfd(ObjFctal, nbhesscheck=10, tolgradchk=1e-6):
    """Finite-difference check for the Hessian of an ObjectiveFunctional
    object"""
    FDobj = ObjFctal.copy()
    lenm = len(FDobj.getmarray())
    ObjFctal.backup_m()
    rnddirc = np.random.randn(nbhesscheck, lenm)
    H = [1e-5, 1e-4, 1e-3]
    factor = [1.0, -1.0]
    hessxdir = FDobj.srchdir
    dirfct = FDobj.delta_m
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
                FDobj.update_m(ObjFctal.getmarray() + fact*hh*dirct)
                FDobj.solvefwd_cost()
                FDobj.solveadj_constructgrad()
                MG.append(FDobj.getMGarray())
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

def compute_searchdirection(ObjFctal, keyword, tolcg=None):
    """Compute search direction for Line Search based on keyword.
    keyword can be 'sd' (steepest descent) or 'Newt' (Newton's method).
    Whether we use full Hessian or GN Hessian in Newton's method depend on
parameter ObjFctal.GN

    ObjFctal = object from class ObjectiveFunctional
    keyword = 'sd' or 'Newt'
    """
    if keyword == 'sd':
        ObjFctal.setsrchdir(-1.0*ObjFctal.getGradarray())
    elif keyword == 'Newt':
        solver = CGSolverSteihaug()
        solver.set_operator(ObjFctal)
        solver.set_preconditioner(ObjFctal.getprecond())
        solver.parameters["rel_tolerance"] = tolcg
        solver.parameters["zero_initial_guess"] = True
        #solver.parameters["print_level"] = print_level-1
        solver.solve(ObjFctal.srchdir.vector(), -ObjFctal.MG.vector())
    else:
        raise ValueError("Wrong keyword")
    ObjFctal.setgradxdir( np.dot(ObjFctal.getsrchdirarray(), \
    ObjFctal.getMGarray()) )
    if ObjFctal.getgradxdir() > 0.0: 
        raise ValueError("Search direction is not a descent direction")
        assert False


def bcktrcklinesearch(ObjFctal, nbLS, alpha_init=1.0, rho=0.5, c=5e-5):
    """Run backtracking line search in 'search_direction'. 
    Default 'search_direction is steepest descent.
    'rho' is multiplicative factor for alpha."""
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
        if ObjFctal.getcost()[0] \
        < cost_mk + alpha * c * ObjFctal.getgradxdir():
            success = True
            break
        alpha *= rho
    return success, LScount, alpha

