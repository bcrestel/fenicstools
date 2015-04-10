import numpy as np
import objectivefunctional

"""Contains functions used to solve optimization problem"""

def checkgradfd(ObjFctal, nbgradcheck=10, tolgradchk=1e-6):
    """Finite-difference check for the gradient of an ObjectiveFunctional
    object"""
    FDobj = ObjFctal.copy()
    lenm = len(FDobj.getmarray())
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
                FDobj.update_m(ObjFctal.getmarray() + fact*hh*dirct)
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
    srch_dir = ObjFctal.getsearchdirarray()
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

