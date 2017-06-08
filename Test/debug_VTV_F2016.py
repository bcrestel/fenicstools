"""
Solver from preconditioner fails with memory corruption error for
jointregularizations of the VTV family, when using Fenics >= 2016.1.0
"""

from dolfin import *

from fenicstools.jointregularization import VTV
from fenicstools.miscfenics import createMixedFS, setfct

useVTV = False

mesh = UnitSquareMesh(10,10)
Vm = FunctionSpace(mesh, 'CG', 1)
VmVm = FunctionSpace(Vm.mesh(), Vm.ufl_element()*Vm.ufl_element())

if useVTV:
    jointregul = VTV(Vm)

    m = interpolate(Expression('sin(pi*x[0])*sin(pi*x[1])', degree=10), Vm)
    jointregul.assemble_hessianab(m, m)

    PCsolver = jointregul.getprecond()
else:
    testm = TestFunction(VmVm)
    trialm = TrialFunction(VmVm)
    M = assemble(inner(testm, trialm)*dx)

    # pb is from using ml_amg; works with petsc_amg and hypre_amg
    PCsolver = PETScKrylovSolver('cg', 'ml_amg')
    PCsolver.set_operator(M)

x = Function(VmVm)
b = interpolate(Constant(('1.0', '1.0')), VmVm)
print '\tApply preconditioner:'
PCsolver.solve(x.vector(), b.vector())
print '\tSuccess'




"""
Bug report:

from dolfin import *

mesh = UnitSquareMesh(10,10)
Vm = FunctionSpace(mesh, 'CG', 1)
VmVm = FunctionSpace(Vm.mesh(), Vm.ufl_element()*Vm.ufl_element())

testm = TestFunction(VmVm)
trialm = TrialFunction(VmVm)
M = assemble(inner(testm, trialm)*dx)

solver = PETScKrylovSolver('cg', 'ml_amg')
solver.set_operator(M)

x = Function(VmVm)
b = interpolate(Constant(('1.0', '1.0')), VmVm)
solver.solve(x.vector(), b.vector())
"""
