"""
Assemble matrices, encode sources,...
Not used anymore
"""
#TODO: Check if it is still used

from dolfin import *
import os
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from misc_fenics import list2point


def assembleobsopB(bc, V, Data):
    """
    Assemble observation operator at the sources and receivers along with the
    observation matrix B
    Inputs:
        bc = boundary condition as assembled by DirichletBC
        V = Function space as assembled by FunctionSpace
        Data = Data dictionary as read from input file
    """

    Dr = assembleobsop(bc, V, Data['receivers'])
    Ds = assembleobsop(bc, V, Data['sources'])

    B = (Dr.T).dot(Dr)  # B = Dr.T x Dr

    return Ds, Dr, csr_matrix(B)


def assembleobsop(bc, V, Points):
    """
    Short version of assembleobervationoperator that only assembles the
    observation operator Dr at the receivers
    Inputs:
        bc = boundary condition as assembled by DirichletBC
        V = Function space as assembled by FunctionSpace
        Points = points where observation should be made
    """

    v = TestFunction(V); f = Constant('0'); L = f*v*dx; 
    b = assemble(L); bc.apply(b); 
    # Observation points
    Nr = len(Points)
    Dobs = np.zeros(Nr*b.size(), float) 
    Dobs = Dobs.reshape((Nr, b.size()), order='C')
#    Dobs = lil_matrix((Nr, b.size()), dtype='float')
    ii = 0
    for obs in Points:
        delta = PointSource(V, list2point(obs))
        bs = b.copy(); delta.apply(bs); bs = bs.array()
        Dobs[ii,:] = bs.transpose()
        ii += 1

    return csr_matrix(Dobs)


def assemble_massandregularization(Vm):
    """
    Assemble and return sparse versions of the mass and regularization matrices
    Inputs:
        Vm = fe space function for med param
    """
    m1 = TrialFunction(Vm)
    m2 = TestFunction(Vm)
    a = inner(nabla_grad(m1), nabla_grad(m2))*dx
    b = inner(m1, m2)*dx

    R = assemble(a)
    Rarr = R.array()
    R_sp = csr_matrix(Rarr)
    Mm = assemble(b)
    Mm_sp = csr_matrix(Mm.array())

    return Mm_sp, R_sp


def generate_sources(V, b, Data):
    Bsource = []
    for source in Data['sources']:
            delta = PointSource(V, list2point(source))
            bs = b.copy(); delta.apply(bs); 
            Bsource.append(bs)

    return Bsource


def encode_sources(Bsource, UD, W):
    Ns = np.shape(W)[0]
    Nw = np.shape(W)[1]

    UD_encod = np.dot(UD, W)

    Bsource_encod = []
    for col in range(Nw):
            b = Bsource[0]*W[0,col]
            for row in range(1,Ns):
                    b += Bsource[row]*W[row,col]
            Bsource_encod.append(b)

    return Bsource_encod, UD_encod

