"""
Test how solver handle operator, memory-wise
self.M: 1816 - 1740
no self.M: 1816 - 1741
"""

from dolfin import UnitSquareMesh, FunctionSpace, TestFunction, TrialFunction, \
assemble, inner, dx, KrylovSolver, Function, interpolate, Constant


class Test():

    def __init__(self):
        mesh = UnitSquareMesh(200,200)
        self.V = FunctionSpace(mesh, 'Lagrange', 2)
        test, trial = TestFunction(self.V), TrialFunction(self.V)
        M = assemble(inner(test, trial)*dx)
        #self.M = assemble(inner(test, trial)*dx)
        self.solverM = KrylovSolver("cg", "amg")
        self.solverM.set_operator(M)
        #self.solverM.set_operator(self.M)

@profile
def run():
    obj = Test()
    u = interpolate(Constant(1.0), obj.V)
    v = Function(obj.V)
    obj.solverM.solve(v.vector(), u.vector())

if __name__ == "__main__":
    run()
