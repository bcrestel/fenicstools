from dolfin import GenericLinearSolver, Function, Constant

class LumpedMatrixSolver(GenericLinearSolver):
    """Lump matrix by row-sum technique"""

    def __init__(self, V):
        u = Function(V)
        u.assign(Constant('1.0'))
        self.one = u.vector()


    def set_operator(self, M):
        """ set_operator(self, M) sets M as the operator """
        self.Mdiag = M * self.one
        assert(self.Mdiag.array().min() > 0)
        self.invMdiag = self.Mdiag.copy()
        self.invMdiag[:] = 1./self.Mdiag.array()


    def solve(self, x, b):
        """ solve(self, x, b) solves Ax = b """
        x[:] = (self.invMdiag * b).array()



class LumpedMatrixSolverS(GenericLinearSolver):
    """Lump matrix by special lumping technique, i.e.,
    scaling diagonal to preserve total mass
    Note: C++ has a get_diagonal command. May need to write that class in C++."""

    def __init__(self, V):
        u = Function(V)
        u.assign(Constant('1.0'))
        self.one = u.vector()
        self.Mdiag = self.one.copy()
        self.invMdiag = self.one.copy()


    def set_operator(self, M):
        """ set_operator(self, M) sets M as the operator """
        # Lump matrix:
        self.Mdiag[:] = M.array().diagonal()
        ratio = self.one.inner(M*self.one) / self.one.inner(self.Mdiag)
        assert(ratio > 1.)
        self.Mdiag = ratio * self.Mdiag
        assert(self.Mdiag.array().min() > 0)
        # Compute inverse action:
        self.invMdiag[:] = 1./self.Mdiag.array()


    def solve(self, x, b):
        """ solve(self, x, b) solves Ax = b """
        x[:] = (self.invMdiag * b).array()
