"""
Test parallel computation with fenics
"""

import numpy as np
from dolfin import UnitSquareMesh, FunctionSpace, Constant, DirichletBC, \
Expression, interpolate, parameters, PETScKrylovSolver, \
MPI, mpi_comm_world, \
TrialFunction, TestFunction, assemble, inner, nabla_grad, dx, LUSolver, Function
from fenicstools.objectivefunctional import ObjFctalElliptic
from fenicstools.observationoperator import ObsEntireDomain

# this option does not appear to be working at the moment:
#parameters["num_threads"] = 6 
mycomm = mpi_comm_world()
myrank = MPI.rank(mycomm)

# Domain, f-e spaces and boundary conditions:
mesh = UnitSquareMesh(500,500)
V = FunctionSpace(mesh, 'Lagrange', 2)  # space for state and adjoint variables
Vm = FunctionSpace(mesh, 'Lagrange', 1) # space for medium parameter
Vme = FunctionSpace(mesh, 'Lagrange', 5)    # sp for target med param

# Define zero Boundary conditions:
def u0_boundary(x, on_boundary):
    return on_boundary
u0 = Constant("0.0")
bc = DirichletBC(V, u0, u0_boundary)

# Define target medium and rhs:
mtrue_exp = Expression('1 + 7*(pow(pow(x[0] - 0.5,2) +' + \
' pow(x[1] - 0.5,2),0.5) > 0.2)')
#mtrue = interpolate(mtrue_exp, Vme)
mtrue = interpolate(mtrue_exp, Vm)
f = Expression("1.0")

# Assemble weak form
trial = TrialFunction(V)
test = TestFunction(V)
a_true = inner(mtrue*nabla_grad(trial), nabla_grad(test))*dx
A_true = assemble(a_true)
bc.apply(A_true)
solver = PETScKrylovSolver('cg')    # doesn't work with ilu preconditioner
#solver = LUSolver()    # doesn't work in parallel !?
#solver.parameters['reuse_factorization'] = True
solver.set_operator(A_true)
# Assemble rhs
L = f*test*dx
b = assemble(L)
bc.apply(b)
# Solve:
u_true = Function(V)

"""
solver.solve(u_true.vector(), b)
if myrank == 0: print 'By hand:\n'
print 'P{0}: max(u)={1}\n'.format(myrank, max(u_true.vector().array()))

MPI.barrier(mycomm)

# Same with object
ObsOp = ObsEntireDomain({'V': V})
Goal = ObjFctalElliptic(V, Vme, bc, bc, [f], ObsOp,[],[],[],False)
Goal.update_m(mtrue)
Goal.solvefwd()
if myrank == 0: print 'With ObjFctalElliptic class:\n'
print 'P{0}: max(u)={1}\n'.format(myrank, max(Goal.U[0]))
"""
