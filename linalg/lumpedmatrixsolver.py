from dolfin import GenericLinearSolver, Function, Constant

class LumpedMatrixSolver(GenericLinearSolver):

    def __init__(self, V):
        u = Function(V)
        u.assign(Constant('1.0'))
        self.one = u.vector()


    def set_operator(self, M):
        """ set_operator(self, M) sets M as the operator """
        self.Mdiag = M * self.one
        self.invMdiag = self.Mdiag.copy()
        self.invMdiag[:] = 1./self.Mdiag.array()


    def solve(self, x, b):
        """ solve(self, x, b) solves Ax = b """
        x[:] = (self.invMdiag * b).array()
