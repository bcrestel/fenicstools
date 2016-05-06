import dolfin as dl
import numpy as np
import ffc
from fenicstools.miscfenics import setfct
from fenicstools.linalg.lumpedmatrixsolver import LumpedMassMatrixPrime

#@profile
def run_test(nbrep=100):
#    dl.parameters['form_compiler']['cpp_optimize'] = True
#    dl.parameters['form_compiler']['cpp_optimize_flags'] = '-O3'
#    dl.parameters['form_compiler']['optimize'] = True
#    ffc.parameters.FFC_PARAMETERS['optimize'] = True
#    ffc.parameters.FFC_PARAMETERS['cpp_optimize'] = True
#    ffc.parameters.FFC_PARAMETERS['cpp_optimize_flags'] = '-O3'
    mesh = dl.UnitSquareMesh(100,100)
    V = dl.FunctionSpace(mesh, 'Lagrange', 2)
    test, trial = dl.TestFunction(V), dl.TrialFunction(V)
    k = dl.Function(V)
    wkform = dl.inner(k*dl.nabla_grad(test), dl.nabla_grad(trial))*dl.dx
    v2 = dl.assemble(wkform)
    for ii in xrange(nbrep):
        setfct(k, float(ii+1))
        v1 = dl.assemble(wkform, form_compiler_parameters={\
        'representation':'quadrature', 'quadrature_degree':1})
        setfct(k, 2.0*float(ii+1))
        dl.assemble(wkform, tensor=v2, \
        form_compiler_parameters={'optimize':True, \
        'representation':'quadrature', 'quadrature_degree':1})

@profile
def run_test2(nbrep=100):
    mesh = dl.UnitSquareMesh(40,40)
    V = dl.FunctionSpace(mesh, 'Lagrange', 2)
    Vr = dl.FunctionSpace(mesh, 'Lagrange', 1)
    test = dl.TestFunction(Vr)
    u, v = dl.Function(V), dl.Function(V)
    # weak form
    wkform = dl.inner(test*u, v)*dl.dx
    # assemble tensor
    M = LumpedMassMatrixPrime(Vr, V, 1.0)
    for ii in xrange(nbrep):
        setfct(u, np.random.randn(V.dim()))
        setfct(v, np.random.randn(V.dim()))

        dl.assemble(wkform)
        M.get_gradient(u.vector(), v.vector())


if __name__ == "__main__":
    run_test2(100)
