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
        x[:] = (self.invMdiag * b).array()



class LumpedMatrixSolverS(GenericLinearSolver):
    """ Lump matrix by special lumping technique, i.e.,
    scaling diagonal to preserve total mass """

    def __init__(self, V):
        u = Function(V)
        u.assign(Constant('1.0'))
        self.one = u.vector()
        self.Mdiag = self.one.copy()
        self.invMdiag = self.one.copy()


    def set_operator(self, M, bc=None):
        """ set_operator(self, M) sets M as the operator """
        # Lump matrix:
        self.Mdiag[:] = get_diagonal(M)
        ratio = self.one.inner(M*self.one) / self.one.inner(self.Mdiag)
        self.Mdiag = ratio * self.Mdiag
        assert self.Mdiag.array().min() > 0., self.Mdiag.array().min()
        if not bc == None:
            indexbc = bc.get_boundary_values().keys()
            self.Mdiag[indexbc] = 1.0
        # Compute inverse action:
        self.invMdiag[:] = 1./self.Mdiag.array()


    def solve(self, x, b):
        """ solve(self, x, b) solves Ax = b """
        x[:] = (self.invMdiag * b).array()
