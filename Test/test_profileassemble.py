import dolfin as dl
import numpy as np
from fenicstools.miscfenics import setfct

#@profile
def run_test(nbrep=100):
    mesh = dl.UnitSquareMesh(100,100)
    V = dl.FunctionSpace(mesh, 'Lagrange', 2)
    test, trial = dl.TestFunction(V), dl.TrialFunction(V)
    k = dl.Function(V)
    wkform = dl.inner(k*dl.nabla_grad(test), dl.nabla_grad(trial))*dl.dx
    for ii in xrange(nbrep):
        setfct(k, float(ii+1))
        dl.assemble(wkform)


if __name__ == "__main__":
    run_test(200)
