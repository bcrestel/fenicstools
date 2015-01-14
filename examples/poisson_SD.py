from dolfin import *
from fenicstools.datamisfit import DataMisfitElliptic

# Domain
mesh = UnitSquareMesh(12,12)
# Finite element spaces
V = FunctionSpace(mesh, 'Lagrange', 2)
Vm = FunctionSpace(mesh, 'Lagrange', 1)
Vme = FunctionSpace(mesh, 'Lagrange', 5)
# Boundary conditions
def u0_boundary(x, on_boundary):
    return on_boundary
u0 = Constant("0.0")
bc = DirichletBC(V, u0, u0_boundary)
# Compute target data
mtrue_exp = Expression('1 + 7*(pow(pow(x[0] - 0.5,2) +' + \
' pow(x[1] - 0.5,2),0.5) > 0.2)')
mtrue = interpolate(mtrue_exp, Vme)
f = Expression("1.0")
goal = DataMisfitElliptic(V, Vme, bc, [f])
goal.update_m(mtrue)
goal.solvefwd()
print goal.u.vector().array()[:20]
UD = goal.U
# Add noise
# TO BE DONE
# Construct gradient at initial state and check it (fd)
InvPb = DataMisfitElliptic(V, Vm, bc, [f], [], UD, 1e-10)
InvPb.update_m(1.0)
InvPb.solvefwd_cost()
print InvPb.u.vector().array()[:20]
print InvPb.misfit, InvPb.regul, InvPb.cost
InvPb.solveadj_constructgrad()
print InvPb.Grad.vector().array()[:20]
InvPb.checkgradfd()
