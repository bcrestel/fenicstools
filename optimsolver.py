"""
Library of functions used to solve optimization problem
"""

import sys
import numpy as np

import dolfin as dl
from dolfin import MPI, mpi_comm_world
from linalg.cgsolverSteihaug import CGSolverSteihaug
from miscfenics import setfct


def checkgradfd_med(ObjFctal, Medium, tolgradchk=1e-6, H=[1e-5, 1e-6,1e-4], doublesided=True):
    mpicomm = ObjFctal.MG.vector().mpi_comm()
    mpirank = MPI.rank(mpicomm)

    ObjFctal.backup_m()
    ObjFctal.solvefwd_cost()
    costref = ObjFctal.cost
    ObjFctal.solveadj_constructgrad()
    MG = ObjFctal.getMG().copy()
    normMG = MG.norm('l2')
    if mpirank == 0:
        print 'Norm of gradient: |MG|={:.5e}'.format(normMG)

    if doublesided: factor = [1.0, -1.0]
    else:   factor = [1.0]

    for textnb, med in enumerate(Medium):
        mgdir = MG.inner(med)
        normmed = med.norm('l2')

        if mpirank == 0:  
            print 'Gradient check -- direction {}: |dir|={}, MGdir={:.5e}'\
            .format(textnb+1, normmed, mgdir)

        for hh in H:
            cost = []
            for fact in factor:
                modifparam = ObjFctal.getmcopy().copy()
                modifparam.axpy(fact*hh, med)
                ObjFctal.update_m(modifparam)
                ObjFctal.solvefwd_cost()
                cost.append(ObjFctal.cost)

            if doublesided: FDgrad = (cost[0] - cost[1])/(2.0*hh)
            else:   FDgrad = (cost[0] - costref)/hh

            if abs(mgdir) < 1e-16:
                err = abs(mgdir - FDgrad)
            else:
                err = abs(mgdir - FDgrad) / abs(mgdir)
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



def checkhessabfd_med(ObjFctal, Medium, tolgradchk=1e-6, \
    H = [1e-5, 1e-6, 1e-4], doublesided=True, direction='b'):
    """
    Finite-difference check for the Hessian of an ObjectiveFunctional object
    """
    mpicomm = ObjFctal.MG.vector().mpi_comm()
    mpirank = MPI.rank(mpicomm)

    ObjFctal.backup_m()
    ObjFctal.solveadj_constructgrad()
    MGref = ObjFctal.getMG().copy()
    normMG = MGref.norm('l2')
    if mpirank == 0:
        print 'Norm of gradient: |MG|={:.5e}'.format(normMG)
    HessVec = ObjFctal.srchdir.copy(deepcopy=True)
    HessVecFD = ObjFctal.srchdir.copy(deepcopy=True)

    if doublesided: factor = [1.0, -1.0]
    else:   factor = [1.0]

    for textnb, med in enumerate(Medium):
        ObjFctal.update_m(ObjFctal.getmcopy())
        ObjFctal.solvefwd_cost()
        ObjFctal.solveadj_constructgrad()
        ObjFctal.mult(med, HessVec.vector())
        HessVeca, HessVecb = HessVec.split(deepcopy=True)
        normhessa = HessVeca.vector().norm('l2')
        normhessb = HessVecb.vector().norm('l2')
        normhess = HessVec.vector().norm('l2')
        normmed = med.norm('l2')
        if mpirank == 0:
            print 'Hessian check -- direction {}: '.format(textnb+1),
            print '|med|={}, |H.x|a={:.5e}, |H.x|b={:.5e}, |H.x|={:.5e}'.format(\
            normmed, normhessa, normhessb, normhess)

        for hh in H:
            MG = []
            for fact in factor:
                modifparam = ObjFctal.getmcopy().copy()
                modifparam.axpy(fact*hh, med)
                ObjFctal.update_m(modifparam)
                ObjFctal.solvefwd_cost()
                ObjFctal.solveadj_constructgrad()
                MG.append(ObjFctal.getMG().copy())

            if doublesided: FDHessx = (MG[0] - MG[1])/(2.0*hh)
            else:   FDHessx = (MG[0] - MGref)/hh
            HessVecFD.vector().zero()
            HessVecFD.vector().axpy(1.0, FDHessx)

            if normhess < 1e-16:
                err = (HessVecFD.vector() - HessVec.vector()).norm('l2')
                HessVecaFD, HessVecbFD = HessVecFD.split(deepcopy=True)
                erra = (HessVecaFD.vector() - HessVeca.vector()).norm('l2')
                errb = (HessVecbFD.vector() - HessVecb.vector()).norm('l2')
            else:
                err = (HessVecFD.vector() - HessVec.vector()).norm('l2') / normhess
                HessVecaFD, HessVecbFD = HessVecFD.split(deepcopy=True)
                erra = (HessVecaFD.vector() - HessVeca.vector()).norm('l2') / normhessa
                errb = (HessVecbFD.vector() - HessVecb.vector()).norm('l2') / normhessb

            FDHxa = HessVecaFD.vector().norm('l2')
            FDHxb = HessVecbFD.vector().norm('l2')
            FDHx = HessVecFD.vector().norm('l2')
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


#@profile
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
        solver.solve(objfctal.srchdir.vector(), -objfctal.MGv)  # all cpu time spent here
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










#def checkhessfd_med(ObjFctal, Medium, tolgradchk=1e-6, \
#    H = [1e-5, 1e-6, 1e-4], doublesided=True):
#    """
#    Finite-difference check for the Hessian of an ObjectiveFunctional object
#    """
#    mpicomm = ObjFctal.MG.vector().mpi_comm()
#    mpirank = MPI.rank(mpicomm)
#
#    lenm = len(ObjFctal.getmcopyarray())
#    ObjFctal.backup_m()
#    MGref = ObjFctal.getMGarray()
#    if doublesided: factor = [1.0, -1.0]
#    else:   factor = [1.0]
#    hessxdir = ObjFctal.srchdir
#    dirfct = ObjFctal.delta_m
#    for textnb, dirct in zip(range(lenm), Medium):
#        # Do computations for analytical Hessian:
#        setfct(dirfct, dirct)
#        ObjFctal.mult(dirfct.vector(), hessxdir.vector())
#        normhess = np.sqrt( MPI.sum(mpicomm, np.linalg.norm(hessxdir.vector().array())**2) )
#        if mpirank == 0:
#            print 'Hessian check -- direction {}: |H.x|={:.5e}'\
#            .format(textnb+1, normhess)
#        # Do computations for FD Hessian:
#        for hh in H:
#            MG = []
#            for fact in factor:
#                ObjFctal.update_m(ObjFctal.getmcopyarray() + fact*hh*dirct)
#                ObjFctal.solvefwd_cost()
#                ObjFctal.solveadj_constructgrad()
#                MG.append(ObjFctal.getMGarray())
#            if doublesided: FDHessx = (MG[0] - MG[1])/(2.0*hh)
#            else:   FDHessx = (MG[0] - MGref)/hh
#            # Compute errors:
#            setfct(dirfct, FDHessx)
#            err = np.sqrt( MPI.sum(mpicomm, \
#            np.linalg.norm(hessxdir.vector().array()-FDHessx)**2) )/normhess
#            if mpirank == 0:
#                print '\t\th={:.1e}: |FDH.x|={:.5e}, err={:.2e}'\
#                .format(hh, np.linalg.norm(FDHessx), err),
#            if err < tolgradchk:
#                if mpirank == 0:    print '\t =>> OK!'
#                break
#            elif mpirank == 0:  print ''
#    # Restore initial value of m:
#    ObjFctal.restore_m()
#    ObjFctal.solvefwd_cost()
#    ObjFctal.solveadj_constructgrad()
