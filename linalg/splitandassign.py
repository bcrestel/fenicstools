import dolfin as dl

from hippylib.linalg import Transpose, MatPtAP
from miscroutines import setupPETScmatrix



class SplitAndAssign():
    """ Class to split (and assign) vectors from MixedFunctionSpace
    into 2 vectors of each FunctionSpace """

    def __init__(self, V1, V2, mpicomm):
        """ WARNING: MixedFunctionSpace is assumed to be V1*V2 
        only works if V1 and V2 are CG of same dimension """
        assert V1.dim() == V2.dim(), "V1, V2 must have same dimension"

        V1V2 = V1*V2

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
