import sys
from time import time
from dolfin import UnitSquareMesh, FunctionSpace, TrialFunction, TestFunction, \
Function, inner, nabla_grad, dx, ds, assemble, sqrt, FacetFunction, Measure, \
SubDomain, MPI, mpi_comm_world
mpicomm = mpi_comm_world()
mpirank = MPI.rank(mpicomm)
mpisize = MPI.size(mpicomm)

mesh = UnitSquareMesh(100, 100, "crossed")

V = FunctionSpace(mesh, 'Lagrange', 2)
Vl = FunctionSpace(mesh, 'Lagrange', 1)
Vr = FunctionSpace(mesh, 'Lagrange', 1)

trial = TrialFunction(V)
test = TestFunction(V)

lam1 = Function(Vl)
lam2 = Function(Vl)
lamV = Function(V)
rho1 = Function(Vl)
rho2 = Function(Vr)


try:
    myrun = int(sys.argv[1])
except:
    myrun = 2

if myrun == 1:
    weak_1 = lam1*inner(nabla_grad(trial), nabla_grad(test))*dx
    weak_2 = inner(lam2*nabla_grad(trial), nabla_grad(test))*dx
    weak_V = inner(lamV*nabla_grad(trial), nabla_grad(test))*dx

    lam1.vector()[:] = 1.0
    if mpirank == 0: print 'Start assembling K1'
    MPI.barrier(mpicomm)
    t0 = time()
    K1 = assemble(weak_1)
    MPI.barrier(mpicomm)
    t1 = time()
    if mpirank == 0: print 'Time to assemble K1 = {}'.format(t1-t0)

    lam2.vector()[:] = 1.0
    if mpirank == 0: print 'Start assembling K2'
    MPI.barrier(mpicomm)
    t0 = time()
    K2 = assemble(weak_2)
    MPI.barrier(mpicomm)
    t1 = time()
    if mpirank == 0: print 'Time to assemble K2 = {}'.format(t1-t0)

    lamV.vector()[:] = 1.0
    if mpirank == 0: print 'Start assembling KV'
    MPI.barrier(mpicomm)
    t0 = time()
    KV = assemble(weak_V)
    MPI.barrier(mpicomm)
    t1 = time()
    if mpirank == 0: print 'Time to assemble KV = {}'.format(t1-t0)
elif myrun == 2:
    class LeftRight(SubDomain):
        def inside(self, x, on_boundary):
            return (x[0] < 1e-16 or x[0] > 1.0 - 1e-16) \
            and on_boundary
    class_bc_abc = LeftRight()
    abc_boundaryparts = FacetFunction("size_t", mesh)
    class_bc_abc.mark(abc_boundaryparts, 1)
    ds = Measure("ds")[abc_boundaryparts]
    weak_1 = inner(sqrt(lam1*rho1)*nabla_grad(trial), nabla_grad(test))*ds(1)
    weak_2 = inner(sqrt(lam2*rho2)*nabla_grad(trial), nabla_grad(test))*ds(1)

    lam1.vector()[:] = 1.0
    rho1.vector()[:] = 1.0
    print 'Start assembling K2'
    t0 = time()
    K2 = assemble(weak_2)
    t1 = time()
    print 'Time to assemble K2 = {}'.format(t1-t0)

    lam2.vector()[:] = 1.0
    rho2.vector()[:] = 1.0
    print 'Start assembling K1'
    t0 = time()
    K1 = assemble(weak_1)
    t1 = time()
    print 'Time to assemble K1 = {}'.format(t1-t0)

