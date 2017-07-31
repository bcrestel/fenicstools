import dolfin as dl

from hippylib.linalg import Transpose, MatPtAP
from miscroutines import setupPETScmatrix
from fenicstools.miscfenics import createMixedFS, createMixedFSi



class SplitAndAssign():
    """ Class to split (and assign) vectors from MixedFunctionSpace
    into 2 vectors of each FunctionSpace """

    def __init__(self, V1, V2, mpicomm):
        """ WARNING: MixedFunctionSpace is assumed to be V1*V2 
        only works if V1 and V2 are CG of same dimension """

        assert V1.dim() == V2.dim(), "V1, V2 must have same dimension"
        assert (V1.dofmap().dofs() == V2.dofmap().dofs()).prod() == 1, \
        "V1 and V2 must have same dofmap"
        assert V1.mesh().size(0) == V2.mesh().size(0), \
        "V1, V2 must be built on same mesh"

        V1V2 = createMixedFS(V1, V2)

        SplitOperator1PETSc,_,_ = setupPETScmatrix(V1, V1V2, 'aij', mpicomm)
        SplitOperator2PETSc,_,_ = setupPETScmatrix(V2, V1V2, 'aij', mpicomm)

        V1V2_1dofs = V1V2.sub(0).dofmap().dofs()
        V1V2_2dofs = V1V2.sub(1).dofmap().dofs()

        V1dofs = V1.dofmap().dofs()
        V2dofs = V2.dofmap().dofs()

        for ii in xrange(len(V1dofs)):
            SplitOperator1PETSc[V1dofs[ii], V1V2_1dofs[ii]] = 1.0

        for ii in xrange(len(V2dofs)):
            SplitOperator2PETSc[V2dofs[ii], V1V2_2dofs[ii]] = 1.0

        SplitOperator1PETSc.assemblyBegin()
        SplitOperator1PETSc.assemblyEnd()
        SplitOperator2PETSc.assemblyBegin()
        SplitOperator2PETSc.assemblyEnd()

        self.SplitOperator1 = dl.PETScMatrix(SplitOperator1PETSc)
        self.SplitOperator2 = dl.PETScMatrix(SplitOperator2PETSc)

        self.AssignOperator1 = Transpose(self.SplitOperator1)
        self.AssignOperator2 = Transpose(self.SplitOperator2)
        

    def split(self, m1m2):
        """ m1m2 is a Vector() from FunctionSpace V1*V2 """
        return self.SplitOperator1*m1m2, self.SplitOperator2*m1m2

    def assign(self, m1, m2):
        """ m1 (resp. m2) is a Vector() from FunctionSpace V1 (resp. V2) """
        return self.AssignOperator1*m1 + self.AssignOperator2*m2



class SplitAndAssigni():
    """ Class to split (and assign) vectors from MixedFunctionSpace
    into vectors of each FunctionSpace """

    def __init__(self, Vs):
        """
        Arguments:
            Vs = list of function spaces
        """
        Vdim = Vs[0].dim()
        Vdofmapdofs = Vs[0].dofmap().dofs()
        Vmeshsize = Vs[0].mesh().size(0)
        for V in Vs:
            assert Vdim == V.dim(), "Vs must have same dimension"
            assert (Vdofmapdofs == V.dofmap().dofs()).prod() == 1, \
            "Vs must have same dofmap"
            assert Vmeshsize == V.mesh().size(0), \
            "Vs must be built on same mesh"

        VV = createMixedFSi(Vs)

        self.SplitOperator = []
        self.AssignOperator = []
        for ii, V in enumerate(Vs):
            V_dofs = V.dofmap().dofs()
            VV_dofs = VV.sub(ii).dofmap().dofs()
            mpicomm = V.mesh().mpi_comm()
            SplitOperatorPETSc,_,_ = setupPETScmatrix(V, VV, 'aij', mpicomm)
            for jj in xrange(len(V_dofs)):
                SplitOperatorPETSc[V_dofs[jj], VV_dofs[jj]] = 1.0
            SplitOperatorPETSc.assemblyBegin()
            SplitOperatorPETSc.assemblyEnd()

            SplitOperator = dl.PETScMatrix(SplitOperatorPETSc)
            self.SplitOperator.append(SplitOperator)
            self.AssignOperator.append(Transpose(SplitOperator))

    def split(self, mm):
        """ mm is a Vector() from FunctionSpace VV """
        out = []
        for splitoperator in self.SplitOperator:
            out.append(splitoperator*mm)
        return out

    def assign(self, ms):
        """ ms is a list of Vector() from FunctionSpace Vs """
        out = []
        for m, assignoperator in zip(ms, self.AssignOperator):
            out.append(assignoperator*m)
        return sum(out)



class BlockDiagonal():
    """ Class to assemble block diagonal operator in MixedFunctionSpace, 
    given operators for each diagonal block.
    Note: Resulting operator is not 'block-diagonal' from  linear algebra point
    of view, as Fenics mix ordering of MixedFunctionSpace """

    def __init__(self, V1, V2, mpicomm):
        self.saa = SplitAndAssign(V1, V2, mpicomm)


    def assemble(self, A, B):
        """ A is square matrix with test, trial from V1,
        B is square matrix with test, trial from V2 """
        return MatPtAP(A, self.saa.SplitOperator1) + \
        MatPtAP(B, self.saa.SplitOperator2)




class PrecondPlusIdentity():
    """
    Define block-diagonal preconditioner made of a preconditioner and the
    identity matrix, i.e. if param == 'a'
    [   B   |   0   ]
    ----------------
    [   0   |   I   ]
    """

    def __init__(self, precondsolver, param, VV):
        self.solver = precondsolver
        self.param = param
        self.ab = dl.Function(VV)
        self.X = dl.Function(VV)


    def solve(self, sol, rhs):
        """
        Solve A.sol = rhs
        Arguments:
            sol = solution
            rhs = rhs
        """
        self.ab.vector().zero()
        self.ab.vector().axpy(1.0, rhs)
        a, b = self.ab.split(deepcopy=True)
        xa, xb = self.X.split(deepcopy=True)

        if self.param == 'a':
            n_pcg = self.solver.solve(xa.vector(),a.vector())

            xb.vector().zero()
            xb.vector().axpy(1.0, b.vector())
        elif self.param == 'b':
            xa.vector().zero()
            xa.vector().axpy(1.0, a.vector())

            n_pcg = self.solver.solve(xb.vector(),b.vector())

        dl.assign(self.X.sub(0), xa)
        dl.assign(self.X.sub(1), xb)

        sol.zero()
        sol.axpy(1.0, self.X.vector())

        return n_pcg
