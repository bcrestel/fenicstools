from dolfin import GenericLinearSolver, Function, Constant

from miscroutines import get_diagonal

class LumpedMatrixSolver(GenericLinearSolver):
    """ Lump matrix by row-sum technique """

    def __init__(self, V):
        u = Function(V)
        u.assign(Constant('1.0'))
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
        x[:] = 0.0
        x.axpy(1.0, self.invMdiag * b)


    def __mul__(self, bvector):
        """ overload * operator for MatVec product
        inputs:
            bvector must be a Fenics Vector """
        return self.Mdiag * bvector



class LumpedMatrixSolverS(GenericLinearSolver):
    """ Lump matrix by special lumping technique, i.e.,
    scaling diagonal to preserve total mass """

    def __init__(self, V):
        u = Function(V)
        u.assign(Constant('1.0'))
        self.one = u.vector()
        self.Mdiag = self.one.copy()
        self.Mdiag2 = self.one.copy()
        self.invMdiag = self.one.copy()


    def set_operator(self, M, bc=None, invMdiag=True):
        """ set_operator(self, M, bc, invMdiag) sets M with boundary conditions
        bc as the operator """
        # Lump matrix:
        self.Mdiag[:] = get_diagonal(M)
        ratio = self.one.inner(M*self.one) / self.one.inner(self.Mdiag)
        self.Mdiag = ratio * self.Mdiag
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
        ratio = self.one.inner(M*self.one) / self.one.inner(self.Mdiag)
        self.Mdiag = ratio * self.Mdiag
        assert self.Mdiag.array().min() > 0., self.Mdiag.array().min()
        if not bc == None:
            indexbc = bc.get_boundary_values().keys()
            self.Mdiag[indexbc] = 1.0
        # Lump matrix D:
        self.Mdiag2[:] = get_diagonal(D)
        ratio2 = self.one.inner(D*self.one) / self.one.inner(self.Mdiag2)
        self.Mdiag2 = ratio2 * self.Mdiag2
        # Assemble M+coeff*D:
        self.Mdiag.axpy(coeff, self.Mdiag2)
        # Compute inverse of (M+coeff*D):
        self.invMdiag[:] = 1./self.Mdiag.array()


    def solve(self, x, b):
        """ solve(self, x, b) solves Ax = b """
        x[:] = 0.0
        x.axpy(1.0, self.invMdiag * b)


    def __mul__(self, bvector):
        """ overload * operator for MatVec product
        inputs:
            bvector must be a Fenics Vector """
        return self.Mdiag * bvector
