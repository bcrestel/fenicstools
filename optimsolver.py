"""
Library of functions used to solve optimization problem
"""

import sys
import numpy as np

import dolfin as dl
try:
    from dolfin import MPI, mpi_comm_world
    PARALLEL = True
except:
    mpirank = 0
    PARALLEL = False
from linalg.cgsolverSteihaug import CGSolverSteihaug
from miscfenics import setfct


def checkgradfd(ObjFctal, nbgradcheck=10, tolgradchk=1e-6, H = [1e-5, 1e-6, 1e-4]):
    """
    Finite-difference check for the gradient of an ObjectiveFunctional object
        ObjFctal = object describing objective functional; must have methods:
            - getmcopyarray: return medium parameter in np.array format
            - backup_m: create safe copy of current medium parameter
            - getMGarray: return Mass*gradient in np.array format
            - getmcopyarray: return copy of medium parameter in np.array format
            - update_m: update medium parameter from a np.array
            - solvefwd_cost: solve fwd pb and compute cost
    and member:
            - cost: value of cost function
    """
    lenm = len(ObjFctal.getmcopyarray())
    ObjFctal.backup_m()
    rnddirc = np.random.randn(nbgradcheck, lenm)
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


def checkgradfd_med(ObjFctal, Medium, tolgradchk=1e-6, H=[1e-5, 1e-6,1e-4], \
doublesided=True, mpicomm=mpi_comm_world()):
    """
    Finite-difference check for the gradient of an ObjectiveFunctional object
        ObjFctal = object describing objective functional; must have methods:
            - getmcopyarray: return medium parameter in np.array format
            - backup_m: create safe copy of current medium parameter
            - getMGarray: return Mass*gradient in np.array format
            - update_m: update medium parameter from a np.array
            - solvefwd_cost: solve fwd pb and compute cost
    and member:
            - cost: value of cost function
    """
    if PARALLEL:    mpirank = MPI.rank(mpicomm)
    lenm = len(ObjFctal.getmcopyarray())
    ObjFctal.backup_m()
    if doublesided: factor = [1.0, -1.0]
    else:   factor = [1.0]
    costref = ObjFctal.cost
    MGdirloc = Medium.dot(ObjFctal.getMGarray())
    for textnb, dirct, mgdirloc in zip(range(lenm), Medium, MGdirloc):
        if PARALLEL:    mgdir = MPI.sum(mpicomm, mgdirloc)
        else:   mgdir = mgdirloc
        if mpirank == 0:  
            print 'Gradient check -- direction {0}: MGdir={1:.5e}'\
            .format(textnb+1, mgdir)
        for hh in H:
            cost = []
            for fact in factor:
                ObjFctal.update_m(ObjFctal.getmcopyarray() + fact*hh*dirct)
                ObjFctal.solvefwd_cost()
                cost.append(ObjFctal.cost)
            if doublesided: FDgrad = (cost[0] - cost[1])/(2.0*hh)
            else:   FDgrad = (cost[0] - costref)/hh
            err = abs(mgdir - FDgrad) / abs(FDgrad)
            if err < tolgradchk:   
                if mpirank == 0:
                    print '\th={0:.1e}: FDgrad={1:.5e}, error={2:.2e} -> OK!'\
                    .format(hh, FDgrad, err)
                break
            elif mpirank == 0:
                print '\th={0:.1e}: FDgrad={1:.5e}, error={2:.2e}'\
                .format(hh, FDgrad, err)
    # Restore initial value of m:
    ObjFctal.restore_m()
    ObjFctal.solvefwd_cost()
    ObjFctal.solveadj_constructgrad()


def checkhessfd(ObjFctal, nbhesscheck=10, tolgradchk=1e-6, H = [1e-5, 1e-6, 1e-4]):
    """Finite-difference check for the Hessian of an ObjectiveFunctional
    object"""
    lenm = len(ObjFctal.getmcopyarray())
    ObjFctal.backup_m()
    rnddirc = np.random.randn(nbhesscheck, lenm)
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


def checkhessfd_med(ObjFctal, Medium, tolgradchk=1e-6, \
H = [1e-5, 1e-6, 1e-4], doublesided=True, mpicomm=mpi_comm_world()):
    """Finite-difference check for the Hessian of an ObjectiveFunctional
    object"""
    if PARALLEL:    mpirank = MPI.rank(mpicomm)
    lenm = len(ObjFctal.getmcopyarray())
    ObjFctal.backup_m()
    MGref = ObjFctal.getMGarray()
    if doublesided: factor = [1.0, -1.0]
    else:   factor = [1.0]
    hessxdir = ObjFctal.srchdir
    dirfct = ObjFctal.delta_m
    for textnb, dirct in zip(range(lenm), Medium):
        # Do computations for analytical Hessian:
        setfct(dirfct, dirct)
        ObjFctal.mult(dirfct.vector(), hessxdir.vector())
        normhess = np.sqrt( MPI.sum(mpicomm, np.linalg.norm(hessxdir.vector().array())**2) )
        if mpirank == 0:
            print 'Hessian check -- direction {}: |H.x|={:.5e}'\
            .format(textnb+1, normhess)
        # Do computations for FD Hessian:
        for hh in H:
            MG = []
            for fact in factor:
                ObjFctal.update_m(ObjFctal.getmcopyarray() + fact*hh*dirct)
                ObjFctal.solvefwd_cost()
                ObjFctal.solveadj_constructgrad()
                MG.append(ObjFctal.getMGarray())
            if doublesided: FDHessx = (MG[0] - MG[1])/(2.0*hh)
            else:   FDHessx = (MG[0] - MGref)/hh
            # Compute errors:
            setfct(dirfct, FDHessx)
            err = np.sqrt( MPI.sum(mpicomm, \
            np.linalg.norm(hessxdir.vector().array()-FDHessx)**2) )/normhess
            if mpirank == 0:
                print '\t\th={:.1e}: |FDH.x|={:.5e}, err={:.2e}'\
                .format(hh, np.linalg.norm(FDHessx), err),
            if err < tolgradchk:
                if mpirank == 0:    print '\t =>> OK!'
                break
            elif mpirank == 0:  print ''
    # Restore initial value of m:
    ObjFctal.restore_m()
    ObjFctal.solvefwd_cost()
    ObjFctal.solveadj_constructgrad()


def checkhessabfd_med(ObjFctal, Medium, tolgradchk=1e-6, \
H = [1e-5, 1e-6, 1e-4], doublesided=True, direction='b', \
mpicomm=mpi_comm_world()):
    """Finite-difference check for the Hessian of an ObjectiveFunctional
    object"""
    if PARALLEL:    mpirank = MPI.rank(mpicomm)
    lenm = len(ObjFctal.getmcopyarray())
    ObjFctal.backup_m()
    MGref = ObjFctal.getMGarray()
    if doublesided: factor = [1.0, -1.0]
    else:   factor = [1.0]
    hessxdir = ObjFctal.srchdir
    dirfct = ObjFctal.delta_m
    for textnb, dirct in zip(range(lenm), Medium):
        # Do computations for analytical Hessian:
        setfct(dirfct, dirct)
        ObjFctal.mult(dirfct.vector(), hessxdir.vector())
        ah1, bh1 = hessxdir.split(deepcopy=True)
        normhessa = np.sqrt( MPI.sum(mpicomm, np.linalg.norm(ah1.vector().array())**2) )
        normhessb = np.sqrt( MPI.sum(mpicomm, np.linalg.norm(bh1.vector().array())**2) )
        normhess = np.sqrt( MPI.sum(mpicomm, np.linalg.norm(hessxdir.vector().array())**2) )
        if mpirank == 0:
            print 'Hessian check -- direction {}: |H.x|a={:.5e}, |H.x|b={:.5e}, |H.x|={:.5e}'\
            .format(textnb+1, normhessa, normhessb, normhess)
        # Do computations for FD Hessian:
        for hh in H:
            MG = []
            for fact in factor:
                ObjFctal.update_m(ObjFctal.getmcopyarray() + fact*hh*dirct)
                ObjFctal.solvefwd_cost()
                ObjFctal.solveadj_constructgrad()
                MG.append(ObjFctal.getMGarray())
            if doublesided: FDHessx = (MG[0] - MG[1])/(2.0*hh)
            else:   FDHessx = (MG[0] - MGref)/hh
            # Compute errors:
            setfct(dirfct, FDHessx)
            ah2, bh2 = dirfct.split(deepcopy=True)
            erra = np.sqrt( MPI.sum(mpicomm, \
            np.linalg.norm(ah1.vector().array()-ah2.vector().array())**2) )\
            /normhessa
            errb = np.sqrt( MPI.sum(mpicomm, \
            np.linalg.norm(bh1.vector().array()-bh2.vector().array())**2) )\
            /normhessb
            err = np.sqrt( MPI.sum(mpicomm, \
            np.linalg.norm(hessxdir.vector().array()-FDHessx)**2) )/normhess
            FDHxa = np.sqrt(MPI.sum(mpicomm, np.linalg.norm(ah2.vector().array())**2))
            FDHxb = np.sqrt(MPI.sum(mpicomm, np.linalg.norm(bh2.vector().array())**2))
            FDHx = np.sqrt(MPI.sum(mpicomm, np.linalg.norm(FDHessx)**2))
            if mpirank == 0:
                print '\t\th={:.1e}: |FDH.x|a={:.5e}, |FDH.x|b={:.5e}, |FDH.x|={:.5e}'\
                .format( hh, FDHxa, FDHxb, FDHx)
                print '\t\t\t\terra={:.2e}, errb={:.2e}, err={:.2e}'.format(erra, errb, err),
            if direction == 'a':
                if erra < tolgradchk:
                    if mpirank == 0:    print '\t =>> OK!'
                    break
                elif mpirank == 0:   print ''
            elif direction == 'b':
                if errb < tolgradchk:
                    if mpirank == 0:    print '\t =>> OK!'
                    break
                elif  mpirank == 0:   print ''
            else:
                if err < tolgradchk:
                    if mpirank == 0:    print '\t =>> OK!'
                    break
                elif mpirank == 0:   print ''
    # Restore initial value of m:
    ObjFctal.restore_m()
    ObjFctal.solvefwd_cost()
    ObjFctal.solveadj_constructgrad()


def compute_searchdirection(objfctal, keyword, tolcg = 1e-8):
    """Compute search direction for Line Search based on keyword.
    keyword can be 'sd' (steepest descent) or 'Newt' (Newton's method).
    Whether we use full Hessian or GN Hessian in Newton's method depend on
    parameter objfctal.GN
    Inputs:
        objfctal = object for objective functional; should contain methods:
            - setsrchdir: set search direction from np.array
            for Newton's method, should also contain methods:
            - mult: matrix operation A.x from fenics class LinearOperator
            - getprecond: return preconditioner for computation Newton's step
            and members:
            - srchdir: search direction
            - MGv: Mass*Gradient in vector format
        keyword = 'sd' or 'Newt'
    """
    if keyword == 'sd':
        objfctal.setsrchdir(-1.0*objfctal.getGradarray())
    elif keyword == 'Newt':
        objfctal.assemble_hessian()
        # Define solver
        solver = CGSolverSteihaug()
        solver.set_operator(objfctal)
        solver.set_preconditioner(objfctal.getprecond())
        solver.parameters["rel_tolerance"] = tolcg
        solver.parameters["zero_initial_guess"] = True
        solver.parameters["print_level"] = -1
        solver.solve(objfctal.srchdir.vector(), -objfctal.MGv)
    else:
        raise ValueError("Wrong keyword")
    # check it is a descent direction
    GradxDir = objfctal.MGv.inner(objfctal.srchdir.vector())
    if GradxDir > 0.0: 
        raise ValueError("Search direction is not a descent direction")
        sys.exit(1)
    if keyword == 'Newt':
        return [solver.iter, solver.final_norm, solver.reasonid, tolcg]


def bcktrcklinesearch(objfctal, nbLS, alpha_init=1.0, rho=0.5, c=5e-5):
    """Run backtracking line search in 'search_direction'. 
    Default 'search_direction is steepest descent.
    'rho' is multiplicative factor for alpha.
    objfctal should contain methods:
        - backup_m
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
    objfctal.backup_m()
    cost_mk = objfctal.cost
    LScount = 0
    success = False
    alpha = alpha_init
    srch_dir = objfctal.srchdir.vector().array()
    GradxDir = objfctal.MGv.inner(objfctal.srchdir.vector())
    # Start Line Search:
    while LScount < nbLS:
        LScount += 1
        new_m = objfctal.getmcopyarray() + alpha*srch_dir
        if np.amin(new_m) > 1e-14:
            objfctal.update_m(new_m)
            objfctal.solvefwd_cost()
            if objfctal.cost < (cost_mk + alpha*c*GradxDir):
                success = True
                break
        alpha *= rho
    return [success, LScount, alpha]

