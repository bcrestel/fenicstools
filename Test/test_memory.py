"""
Compare memory footprint of different output format
Function: 230Mb (23 bytes)
Vector: 212Mb (21.2 bytes)
Array: 174Mb (17.4 bytes)
Mass matrix: 44Mb
"""

from dolfin import UnitSquareMesh, FunctionSpace, interpolate, Constant, \
TestFunction, TrialFunction, assemble, inner, dx
import numpy as np

@profile
def run():
    mesh = UnitSquareMesh(20,20)
    V = FunctionSpace(mesh,'Lagrange',2)
    u = interpolate(Constant(2.0), V)
    sol = []
    for ii in range(10000):
        #sol.append(u.copy(deepcopy=True))
        #sol.append(u.vector().copy())
        sol.append(u.vector().array())
    test, trial = TestFunction(V), TrialFunction(V)
    M = assemble(inner(test, trial)*dx)
    m1 = np.random.randn(1000000).reshape((1000,1000))
    m2 = []
    for ii in range(1000):
        m2.append(np.random.randn(1000))


if __name__ == "__main__":
    run()
