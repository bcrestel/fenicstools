""" Implement a few general command for linear algebra in Fenics with PETSc as
the linear algebra backend """

import numpy as np
from dolfin import as_backend_type, PETScVector

import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc


def get_diagonal(M):
    """ Extract diagonal of a square Matrix M and return a PETScVector """

    return PETScVector(as_backend_type(M).mat().getDiagonal())



def setglobalvalue(fct, globalindex, value):
    """ set the globalindex of a Fenics Function fct to value, i.e., 
    fct.vector()[globalindex] = value """

    vec = fct.vector()
    V = fct.function_space()
    dof2globindex = V.dofmap().dofs()
    if vec.owns_index(globalindex):
        localindex = np.array(np.where(dof2globindex == globalindex)[0], \
        dtype=np.intc)
        vec.set_local(value*np.ones(localindex.shape), localindex)
    vec.apply('insert')



def setupPETScmatrix(Vr, Vc, mpicomm):
    """ 
    Set up a PETSc matrix partitioned consistently with Fenics mesh
    Vr, Vc = function spaces for the rows and columns
    """
    
    # extract local to global map for each fct space
    VrDM, VcDM = Vr.dofmap(), Vc.dofmap()
    r_map = PETSc.LGMap().create(VrDM.dofs(), comm=mpicomm)
    c_map = PETSc.LGMap().create(VcDM.dofs(), comm=mpicomm)
    # set up matrix
    PETScMatrix = PETSc.Mat()
    PETScMatrix.create(mpicomm)
    PETScMatrix.setSizes([ [VrDM.local_dimension("owned"), Vr.dim()], \
    [VcDM.local_dimension("owned"), Vc.dim()] ])
    PETScMatrix.setType('aij') # sparse
    PETScMatrix.setUp()
    PETScMatrix.setLGMap(r_map, c_map)
    # compare PETSc and Fenics local partitions:
    Istart, Iend = PETScMatrix.getOwnershipRange()
    assert list(VrDM.dofs()) == range(Istart, Iend)
    return PETScMatrix, VrDM, VcDM
