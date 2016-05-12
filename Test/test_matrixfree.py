from dolfin import *

class IdentityMatrix(LinearOperator):

    def __init__(self, V):
        u = Function(V)
        LinearOperator.__init__(self, u.vector(), u.vector())

    def mult(self, x, y):
        y.zero()
        y.axpy(1.0, x)


if __name__ == "__main__":
    print 'linear algebra backend = {}'.format(parameters["linear_algebra_backend"])
    mesh = UnitSquareMesh(10,10)
    V = FunctionSpace(mesh, 'Lagrange', 1)
    Id = IdentityMatrix(V)
    solver = PETScKrylovSolver("cg", "none")    # only work with "none" preconditioner
    #solver = LUSolver("petsc") # does not work. LUSolver requires an actual matrix
    solver.set_operator(Id)
    b = interpolate(Constant(1.0), V)
    x = Function(V)
    solver.solve(x.vector(), b.vector())
