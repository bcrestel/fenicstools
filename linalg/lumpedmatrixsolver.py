import dolfin as dl
from dolfin import MPI
import numpy as np

from miscroutines import get_diagonal
from fenicstools.miscfenics import setfct
from fenicstools.linalg.miscroutines import setglobalvalue

import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc


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
        self.Mdiag.zero()
        self.Mdiag.axpy(1.0, get_diagonal(M))
        self.ratio = self.one.inner(M*self.one) / self.one.inner(self.Mdiag)
        self.Mdiag = self.Mdiag * self.ratio
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
        self.Mdiag.zero()
        self.Mdiag.axpy(1.0, get_diagonal(M))
        self.ratio = self.one.inner(M*self.one) / self.one.inner(self.Mdiag)
        self.Mdiag = self.ratio * self.Mdiag
        assert self.Mdiag.array().min() > 0., self.Mdiag.array().min()
        if not bc == None:
            indexbc = bc.get_boundary_values().keys()
            self.Mdiag[indexbc] = 1.0
        # Lump matrix D:
        self.Mdiag2.zero()
        self.Mdiag2.axpy(1.0, get_diagonal(D))
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
    mass matrix is \int alpha phi_i phi_j dx.
    we consider a lumping by diagonal scaling, i.e., corresponding to Lumped MatrixSolverS
    we consider the derivative wrt parameter alpha """

    def __init__(self, Va, Vphi, ratioM=None, mpicomm=PETSc.COMM_WORLD):
        """ Va = FunctionSpace for weight-parameter in mass matrix
        Vphi = FunctionSpace for test and trial functions in mass matrix
        ratioM = ratio used for the lumping of mass matrix """

        u = dl.Function(Vphi)
        self.uvector = u.vector()
        # prepare weak form
        test, trial = dl.TestFunction(Vphi), dl.TrialFunction(Vphi)
        alpha = dl.Function(Va)
        self.ratioM = ratioM
        wkform = dl.inner(alpha*test, trial)*dl.dx
        M = dl.assemble(wkform)
        # extract local to global map for each fct space
        VaDM, VphiDM = Va.dofmap(), Vphi.dofmap()
        a_map = PETSc.LGMap().create(VaDM.dofs(), mpicomm)
        phi_map = PETSc.LGMap().create(VphiDM.dofs(), mpicomm)
        # assemble PETSc version of Mprime
        MprimePETSc = PETSc.Mat()
        MprimePETSc.create(mpicomm)
        MprimePETSc.setSizes([ [VaDM.local_dimension("owned"), Va.dim()], \
        [VphiDM.local_dimension("owned"), Vphi.dim()] ])
        MprimePETSc.setType('aij') # sparse
#        MprimePETSc.setPreallocationNNZ(30)
        MprimePETSc.setUp()
        MprimePETSc.setLGMap(a_map, phi_map)
        # compare PETSc and Fenics local partitions:
        Istart, Iend = MprimePETSc.getOwnershipRange()
        assert list(VaDM.dofs()) == range(Istart, Iend)
        # populate the PETSc matrix
        for ii in xrange(Va.dim()):
            alpha.vector().zero()
            setglobalvalue(alpha, ii, 1.0)
            dl.assemble(wkform, tensor=M)
            diagM = get_diagonal(M)
            normdiagM = diagM.norm('l2')
            diagM = diagM.array()
            cols = np.where(np.abs(diagM) > 1e-16*normdiagM)[0]
            for cc, val in zip(cols, diagM[cols]):  
                global_cc = VphiDM.dofs()[cc]
                MprimePETSc[ii, global_cc] = val
        MprimePETSc.assemblyBegin()
        MprimePETSc.assemblyEnd()
        # convert PETSc matrix to PETSc-wrapper in Fenics
        self.Mprime = dl.PETScMatrix(MprimePETSc)


    def updater(self, ratioM):  self.ratioM = ratioM


    def get_gradient(self, u, v):
        """ compute gradient of the expression u^T.M.v 
        with respect to weight-parameter alpha in weighted-mass matrix 
            u, v = Vectors """

        uv = u*v
        return (self.Mprime * uv) * self.ratioM


    def get_incremental(self, ahat, u):
        """ Compute term on rhs of incremental equations
        ahat and u are vectors """

        self.Mprime.transpmult(ahat, self.uvector)
        return (self.uvector * u) * self.ratioM



class LumpedMassPreconditioner(dl.PETScUserPreconditioner):
    """ Define matrix-free preconditioner for mass matrix inversion based on
    lumped mass matrix """

    def __init__(self, V, M, bc):
        self.solverMlumped = LumpedMatrixSolver(V)
        self.solveRMlumped.set_operator(M, bc)

    def solve(self, x, b):
        self.solverMlumped.solver(x, b)
