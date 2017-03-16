"""
Script to test validity of classed defined in linalg/splitandassign.py
"""
import numpy as np
import dolfin as dl

from fenicstools.linalg.splitandassign import SplitAndAssign, BlockDiagonal

#@profile
def testsplitassign():
    mesh = dl.UnitSquareMesh(40,40)
    V1 = dl.FunctionSpace(mesh, "Lagrange", 2)
    V2 = dl.FunctionSpace(mesh, "Lagrange", 2)
    V1V2 = V1*V2
    splitassign = SplitAndAssign(V1, V2, mesh.mpi_comm())

    mpirank = dl.MPI.rank(mesh.mpi_comm())

    u = dl.interpolate(dl.Expression(("x[0]*x[1]", "11+x[0]+x[1]")), V1V2)
    uu = dl.Function(V1V2)
    u1, u2 = u.split(deepcopy=True)
    u1v, u2v = splitassign.split(u.vector())
    u11 = dl.interpolate(dl.Expression("x[0]*x[1]"), V1)
    u22 = dl.interpolate(dl.Expression("11+x[0]+x[1]"), V2)
    a,b,c,d= dl.norm(u1.vector()-u1v), dl.norm(u2.vector()-u2v),\
    dl.norm(u1.vector()-u11.vector()), dl.norm(u2.vector()-u22.vector())
    if mpirank == 0:
        print 'Splitting an interpolated function:', a, b, c, d

    uv = splitassign.assign(u1v, u2v)
    dl.assign(uu.sub(0), u11)
    dl.assign(uu.sub(1), u22)
    a, b = dl.norm(uv-u.vector()), dl.norm(uv-uu.vector())
    if mpirank == 0:
        print 'Putting it back together:', a, b

    for ii in xrange(10):
        u.vector()[:] = np.random.randn(len(u.vector().array()))
        u1, u2 = u.split(deepcopy=True)
        u1v, u2v = splitassign.split(u.vector())
        uv = splitassign.assign(u1v, u2v)
        a, b = dl.norm(u1.vector()-u1v), dl.norm(u2.vector()-u2v)
        c = dl.norm(uv-u.vector())
        if mpirank == 0:
            print 'Splitting random numbers:', a, b
            print 'Putting it back together:', c


def testblockdiagonal():
    mesh = dl.UnitSquareMesh(40,40)
    V1 = dl.FunctionSpace(mesh, "Lagrange", 2)
    test1, trial1 = dl.TestFunction(V1), dl.TrialFunction(V1)
    V2 = dl.FunctionSpace(mesh, "Lagrange", 2)
    test2, trial2 = dl.TestFunction(V2), dl.TrialFunction(V2)
    V1V2 = V1*V2
    test12, trial12 = dl.TestFunction(V1V2), dl.TrialFunction(V1V2)
    bd = BlockDiagonal(V1, V2, mesh.mpi_comm())

    mpirank = dl.MPI.rank(mesh.mpi_comm())

    if mpirank == 0:    print 'mass+mass'
    M1 = dl.assemble(dl.inner(test1, trial1)*dl.dx)
    M2 = dl.assemble(dl.inner(test1, trial2)*dl.dx)
    M12bd = bd.assemble(M1, M2)
    M12 = dl.assemble(dl.inner(test12, trial12)*dl.dx)
    diff = M12bd - M12
    nn = diff.norm('frobenius')
    if mpirank == 0:
        print nn

    if mpirank == 0:    print 'mass+2ndD'
    D2 = dl.assemble(dl.inner(dl.nabla_grad(test1), dl.nabla_grad(trial2))*dl.dx)
    M1D2bd = bd.assemble(M1, D2)
    tt1, tt2 = test12
    tl1, tl2 = trial12
    M1D2 = dl.assemble(dl.inner(tt1, tl1)*dl.dx + dl.inner(dl.nabla_grad(tt2),dl.nabla_grad(tl2))*dl.dx)
    diff = M1D2bd - M1D2
    nn = diff.norm('frobenius')
    if mpirank == 0:
        print nn

    if mpirank == 0:    print 'wM+wM'
    u11 = dl.interpolate(dl.Expression("x[0]*x[1]"), V1)
    u22 = dl.interpolate(dl.Expression("11+x[0]+x[1]"), V2)
    M1 = dl.assemble(dl.inner(u11*test1, trial1)*dl.dx)
    M2 = dl.assemble(dl.inner(u22*test1, trial2)*dl.dx)
    M12bd = bd.assemble(M1, M2)
    ua, ub = dl.interpolate(dl.Expression(("x[0]*x[1]", "11+x[0]+x[1]")), V1V2)
    M12 = dl.assemble(dl.inner(ua*tt1, tl1)*dl.dx + dl.inner(ub*tt2, tl2)*dl.dx)
    diff = M12bd - M12
    nn = diff.norm('frobenius')
    if mpirank == 0:
        print nn



if __name__ == "__main__":
    testsplitassign()
    testblockdiagonal()

