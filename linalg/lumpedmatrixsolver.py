import dolfin as dl
import numpy as np

from miscroutines import get_diagonal
from fenicstools.miscfenics import setfct

class LumpedMatrixSolver(dl.GenericLinearSolver):
    """ Lump matrix by row-sum technique """

    def __init__(self, V):
        u = dl.Function(V)
        u.assign(dl.Constant('1.0'))
        self.one = u.vector()
        self.invMdiag = self.one.copy()


    def set_operator(self, M, bc=None):
        """ set_operator(self, M) sets M as the operator """
        self.Mdiag = M * self.one
        assert self.Mdiag.array().min() > 0., self.Mdiag.array().min()
        if not bc == None:
            indexbc = bc.get_boundary_values().keys()
            self.Mdiag[indexbc] = 1.0
        self.invMdiag[:] = 1./self.Mdiag.array()


    def solve(self, x, b):
        """ solve(self, x, b) solves Ax = b """
        x.zero()
        x.axpy(1.0, self.invMdiag * b)  # entry-wise product


    def __mul__(self, bvector):
        """ overload * operator for MatVec product
        inputs:
            bvector must be a Fenics Vector """
        return self.Mdiag * bvector



class LumpedMatrixSolverS(dl.GenericLinearSolver):
    """ Lump matrix by special lumping technique, i.e.,
    scaling diagonal to preserve total mass """

    def __init__(self, V):
        """ V = FunctionSpace for the matrix """
        u = dl.Function(V)
        u.assign(dl.Constant('1.0'))
        self.one = u.vector()
        self.Mdiag = self.one.copy()
        self.Mdiag2 = self.one.copy()
        self.invMdiag = self.one.copy()


    def set_operator(self, M, bc=None, invMdiag=True):
        """ set_operator(self, M, bc, invMdiag) sets M with boundary conditions
        bc as the operator """
        # Lump matrix:
        self.Mdiag[:] = get_diagonal(M)
        self.ratio = self.one.inner(M*self.one) / self.one.inner(self.Mdiag)
        self.Mdiag = self.ratio * self.Mdiag
        if invMdiag:
            assert self.Mdiag.array().min() > 0., self.Mdiag.array().min()
        if not bc == None:
            indexbc = bc.get_boundary_values().keys()
            self.Mdiag[indexbc] = 1.0
        if invMdiag:    self.invMdiag[:] = 1./self.Mdiag.array()


    def set_operators(self, M, D, coeff, bc=None):
        """ set_operator(self, M, bc, D, coeff) sets M + coeff*D as the operator;
        bc only applies to M """
        # Lump matrix M:
        self.Mdiag[:] = get_diagonal(M)
        self.ratio = self.one.inner(M*self.one) / self.one.inner(self.Mdiag)
        self.Mdiag = self.ratio * self.Mdiag
        assert self.Mdiag.array().min() > 0., self.Mdiag.array().min()
        if not bc == None:
            indexbc = bc.get_boundary_values().keys()
            self.Mdiag[indexbc] = 1.0
        # Lump matrix D:
        self.Mdiag2[:] = get_diagonal(D)
        self.ratio2 = self.one.inner(D*self.one) / self.one.inner(self.Mdiag2)
        self.Mdiag2 = self.ratio2 * self.Mdiag2
        # Assemble M+coeff*D:
        self.Mdiag.axpy(coeff, self.Mdiag2)
        # Compute inverse of (M+coeff*D):
        self.invMdiag[:] = 1./self.Mdiag.array()


    def solve(self, x, b):
        """ solve(self, x, b) solves Ax = b """
        x.zero()
        x.axpy(1.0, self.invMdiag * b)  # entry-wise product


    def __mul__(self, bvector):
        """ overload * operator for MatVec product
        inputs:
            bvector must be a Fenics Vector """
        return self.Mdiag * bvector

class LumpedMassMatrixPrime():
    """ Assemble tensor for the derivative of a lumped weighted mass matrix wrt
    to the weight parameter 
    mass matrix is \int rho phi_i phi_j dx.
    we consider a lumping by diagonal scaling, i.e., corresponding to Lumped MatrixSolverS
    we consider the derivative wrt parameter rho """

    #@profile
    def __init__(self, Vr, Vphi, ratioM=None):
        """ Vr = FunctionSpace for weight-parameter in mass matrix
        Vphi = FunctionSpace for test and trial functions in mass matrix
        ratioM = ratio used for the lumping of mass matrix """
        test, trial = dl.TestFunction(Vphi), dl.TrialFunction(Vphi)
        gradM = dl.Function(Vr)
        self.gradMv = gradM.vector()
        rho = dl.Function(Vr)
        self.ratioM = ratioM
        wkform = dl.inner(rho*test, trial)*dl.dx
        diagM = dl.Function(Vphi)
        self.Mprime = []
        M = dl.assemble(wkform)
        for ii in xrange(Vr.dim()):
            rho.vector().zero()
            rho.vector()[ii] = 1.0
            dl.assemble(wkform, tensor=M)
            #M = dl.assemble(wkform)
            #dM = get_diagonal(M)
            #setfct(diagM, dM)
            setfct(diagM, get_diagonal(M))
            self.Mprime.append(diagM.vector().copy())

    def updater(self, ratioM):  self.ratioM = ratioM

    #@profile
    def get_gradient(self, u, v):
        """ compute gradient of the expression u^T.M.v with respect to weight-parameter
        rho in weighted-mass matrix 
            u, v = Vectors """
        self.gradMv.zero()
        outarr = np.zeros(len(self.gradMv))
        for ii, mp in enumerate(self.Mprime):
#            mpv = mp*v
#            umpv = u.inner(mpv)
#            out = self.ratioM*umpv
#            outarr[ii] = out
            outarr[ii] = self.ratioM*(u.inner(mp*v))
        self.gradMv[:] = outarr
        return self.gradMv
