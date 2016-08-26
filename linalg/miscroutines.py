""" Implement a few general command for linear algebra in Fenics with PETSc as
the linear algebra backend """

import numpy as np
from dolfin import as_backend_type, PETScVector

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

