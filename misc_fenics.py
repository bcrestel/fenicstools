"""
General commands for operations on Fenics objects
"""

from dolfin import *
import numpy as np

def M_innerprod_npndarr(M, v1, v2):
    """
    Compute the discrete M inner-product for two np.ndarray v1 and v2,
    ie v1^T . M . v2
    Inputs:
        M = matrix M in array format
        v1, v2 = numpy.ndarray objects
    """

    return np.dot( M.dot(v1), v2 )


def M_innerprod(M, v1, v2):
    """
    Compute the discrete M inner-product for two Fenics fe functions v1 and v2,
    ie v1^T . M . v2
    Inputs:
        M = matrix M in array format
        v1, v2 = Fenics finite-element function
    """
    v1_arr = v1.vector().array()
    v2_arr = v2.vector().array()

    return M_innerprod_npndarr(M, v1_arr, v2_arr)


def M_norm(M, v1):
    """
    Compute the norm wrt M inner-prodcut
    Inputs:
        M = matrix M in array format
        v1 = Fenics finite-element fct
    """

    return np.sqrt( M_innerprod(M, v1, v1) )


def M_norm_misfit(Mm, m, m_ref):
    """
    Compute misfit ||m - m_ref||_M in M_norm
    Inputs:
        m = current medium parameter in Fenics fct format
        m_ref = ref medium parameter in Fenics fct format
        Mm = mass matrix in array format
    """
    diffm = m.vector().array() - m_ref.vector().array()

    return np.sqrt( M_innerprod_npndarr(Mm, diffm, diffm) )


def M_norm_misfit_rel(Mm, m, m_ref):
    """
    Compute relative medium misfit ||m - m_ref|| / ||m_ref||
    Inputs:
        m = current medium parameter in Fenics fct format
        m_ref = exact medium parameter in Fenics fct format
        Mm = mass matrix in array format
    """

    return M_norm_misfit(Mm, m, m_ref) / M_norm(Mm, m_ref)


def list2point(list_in):
    """
    Turn a list into a Fenics Point
    Inputs:
        list_in = list containing coordinates of the Point
    """
    dim = np.size(list_in)

    return Point(dim, np.array(list_in, dtype=float))

