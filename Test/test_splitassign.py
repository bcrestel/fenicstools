"""
Script to test validity of classed defined in linalg/splitandassign.py
"""
import numpy as np
import dolfin as dl

from fenicstools.miscfenics import createMixedFS, createMixedFSi
from fenicstools.linalg.splitandassign import SplitAndAssign, BlockDiagonal, SplitAndAssigni

#@profile
def testsplitassign():
    USEi = True

    mesh = dl.UnitSquareMesh(40,40)
    V1 = dl.FunctionSpace(mesh, "Lagrange", 2)
    V2 = dl.FunctionSpace(mesh, "Lagrange", 2)
    if USEi:
        V1V2 = createMixedFSi([V1, V2])
        splitassign = SplitAndAssigni([V1, V2], mesh.mpi_comm())
    else:
        V1V2 = createMixedFS(V1, V2)
        splitassign = SplitAndAssign(V1, V2, mesh.mpi_comm())

    mpirank = dl.MPI.rank(mesh.mpi_comm())

    u = dl.interpolate(dl.Expression(("x[0]*x[1]", "11+x[0]+x[1]"), degree=10), V1V2)
    uu = dl.Function(V1V2)
    u1, u2 = u.split(deepcopy=True)
    u1v, u2v = splitassign.split(u.vector())
    u11 = dl.interpolate(dl.Expression("x[0]*x[1]", degree=10), V1)
    u22 = dl.interpolate(dl.Expression("11+x[0]+x[1]", degree=10), V2)
    a,b,c,d= dl.norm(u1.vector()-u1v), dl.norm(u2.vector()-u2v),\
    dl.norm(u1.vector()-u11.vector()), dl.norm(u2.vector()-u22.vector())
    if mpirank == 0:
        print '\nSplitting an interpolated function:', a, b, c, d

    if USEi:
        uv = splitassign.assign([u1v, u2v])
    else:
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
        if USEi:
            uv = splitassign.assign([u1v, u2v])
        else:
            uv = splitassign.assign(u1v, u2v)
        a, b = dl.norm(u1.vector()-u1v), dl.norm(u2.vector()-u2v)
        c = dl.norm(uv-u.vector())
        if mpirank == 0:
            print 'Splitting random numbers:', a, b
            print 'Putting it back together:', c



def testsplitassigni():
    mesh = dl.UnitSquareMesh(40,40)
    V1 = dl.FunctionSpace(mesh, "Lagrange", 2)
    V2 = dl.FunctionSpace(mesh, "Lagrange", 2)
    V3 = dl.FunctionSpace(mesh, "Lagrange", 2)
    V1V2 = createMixedFSi([V1, V2, V3])
    splitassign = SplitAndAssigni([V1, V2, V3], mesh.mpi_comm())

    mpirank = dl.MPI.rank(mesh.mpi_comm())

    u = dl.interpolate(dl.Expression(("x[0]*x[1]", "11+x[0]+x[1]", "x[0]*x[0]"), degree=10), V1V2)
    uu = dl.Function(V1V2)
    u1, u2, u3 = u.split(deepcopy=True)
    u1v, u2v, u3v = splitassign.split(u.vector())
    u11 = dl.interpolate(dl.Expression("x[0]*x[1]", degree=10), V1)
    u22 = dl.interpolate(dl.Expression("11+x[0]+x[1]", degree=10), V2)
    u33 = dl.interpolate(dl.Expression("x[0]*x[0]", degree=10), V3)
    a,b,c,d,e,f= dl.norm(u1.vector()-u1v), dl.norm(u2.vector()-u2v),\
    dl.norm(u3.vector()-u3v),\
    dl.norm(u1.vector()-u11.vector()), dl.norm(u2.vector()-u22.vector()),\
    dl.norm(u3.vector()-u33.vector())
    if mpirank == 0:
        print '\nSplitting an interpolated function:', a, b, c, d, e, f

    uv = splitassign.assign([u1v, u2v, u3v])
    dl.assign(uu.sub(0), u11)
    dl.assign(uu.sub(1), u22)
    dl.assign(uu.sub(2), u33)
    a, b = dl.norm(uv-u.vector()), dl.norm(uv-uu.vector())
    if mpirank == 0:
        print 'Putting it back together:', a, b

    for ii in xrange(10):
        u.vector()[:] = np.random.randn(len(u.vector().array()))
        u1, u2, u3 = u.split(deepcopy=True)
        u1v, u2v, u3v = splitassign.split(u.vector())
        uv = splitassign.assign([u1v, u2v, u3v])
        a, b, c = dl.norm(u1.vector()-u1v), dl.norm(u2.vector()-u2v), dl.norm(u3.vector()-u3v)
        d = dl.norm(uv-u.vector())
        if mpirank == 0:
            print 'Splitting random numbers:', a, b, c
            print 'Putting it back together:', d



def testblockdiagonal():
    mesh = dl.UnitSquareMesh(40,40)
    V1 = dl.FunctionSpace(mesh, "Lagrange", 2)
    test1, trial1 = dl.TestFunction(V1), dl.TrialFunction(V1)
    V2 = dl.FunctionSpace(mesh, "Lagrange", 2)
    test2, trial2 = dl.TestFunction(V2), dl.TrialFunction(V2)
    V1V2 = createMixedFS(V1, V2)
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
        print '\nnn={}'.format(nn)

    if mpirank == 0:    print 'mass+2ndD'
    D2 = dl.assemble(dl.inner(dl.nabla_grad(test1), dl.nabla_grad(trial2))*dl.dx)
    M1D2bd = bd.assemble(M1, D2)
    tt1, tt2 = test12
    tl1, tl2 = trial12
    M1D2 = dl.assemble(dl.inner(tt1, tl1)*dl.dx + dl.inner(dl.nabla_grad(tt2),dl.nabla_grad(tl2))*dl.dx)
    diff = M1D2bd - M1D2
    nn = diff.norm('frobenius')
    if mpirank == 0:
        print 'nn={}'.format(nn)

    if mpirank == 0:    print 'wM+wM'
    u11 = dl.interpolate(dl.Expression("x[0]*x[1]", degree=10), V1)
    u22 = dl.interpolate(dl.Expression("11+x[0]+x[1]", degree=10), V2)
    M1 = dl.assemble(dl.inner(u11*test1, trial1)*dl.dx)
    M2 = dl.assemble(dl.inner(u22*test1, trial2)*dl.dx)
    M12bd = bd.assemble(M1, M2)
    ua, ub = dl.interpolate(dl.Expression(("x[0]*x[1]", "11+x[0]+x[1]"),\
    degree=10), V1V2)
    M12 = dl.assemble(dl.inner(ua*tt1, tl1)*dl.dx + dl.inner(ub*tt2, tl2)*dl.dx)
    diff = M12bd - M12
    nn = diff.norm('frobenius')
    if mpirank == 0:
        print 'nn={}'.format(nn)


def testassignsplit():
    """
    Check that assign then splitting does not modify entries

    no differences recorded
    """
    mesh = dl.UnitSquareMesh(60,60)
    mpirank = dl.MPI.rank(mesh.mpi_comm())
    if mpirank == 0:    print '\n'
    V1 = dl.FunctionSpace(mesh, "Lagrange", 2)
    V2 = dl.FunctionSpace(mesh, "Lagrange", 2)
    V1V2 = createMixedFS(V1, V2)
    ab = dl.Function(V1V2)
    add = dl.interpolate(dl.Constant(('1.0', '0.0')), V1V2)
    a, b = dl.Function(V1), dl.Function(V2)
    adda, addb = dl.interpolate(dl.Constant('1.0'), V1), dl.interpolate(dl.Constant('0.0'), V2)

    for ii in range(10):
        a.vector()[:] = np.random.randn(len(a.vector().array()))
        norma = a.vector().norm('l2')
        b.vector()[:] = np.random.randn(len(b.vector().array()))
        normb = b.vector().norm('l2')
        dl.assign(ab.sub(0), a)
        dl.assign(ab.sub(1), b)
        ab.vector().axpy(1.0, add.vector())
        ab.vector().axpy(-1.0, add.vector())
        aba, abb = ab.split(deepcopy=True)
        a.vector().axpy(1.0, adda.vector())
        a.vector().axpy(-1.0, adda.vector())
        b.vector().axpy(1.0, addb.vector())
        b.vector().axpy(-1.0, addb.vector())
        diffa = (a.vector() - aba.vector()).norm('l2') / norma
        diffb = (b.vector() - abb.vector()).norm('l2') / normb
        if mpirank == 0:
            print 'diffa={} (|a|={}), diffb={} (|b|={})'.format(\
            diffa, norma, diffb, normb)




def testassignspliti():
    """
    Check that assign then splitting does not modify entries

    small differences recorded due to round-off error
    """
    mesh = dl.UnitSquareMesh(60,60)
    mpirank = dl.MPI.rank(mesh.mpi_comm())
    if mpirank == 0:    print '\n'
    V1 = dl.FunctionSpace(mesh, "Lagrange", 2)
    V2 = dl.FunctionSpace(mesh, "Lagrange", 2)
    V3 = dl.FunctionSpace(mesh, "Lagrange", 2)
    V1V2 = createMixedFSi([V1, V2, V3])
    ab = dl.Function(V1V2)
    add = dl.interpolate(dl.Constant(('-1.0', '0.0', '1.0')), V1V2)
    a, b, c = dl.Function(V1), dl.Function(V2), dl.Function(V3)
    adda, addb, addc = dl.interpolate(dl.Constant('1.0'), V1),\
    dl.interpolate(dl.Constant('1.0'), V2), dl.interpolate(dl.Constant('1.0'), V3)

    for ii in range(10):
        a.vector()[:] = np.random.randn(len(a.vector().array()))
        norma = a.vector().norm('l2')
        b.vector()[:] = np.random.randn(len(b.vector().array()))
        normb = b.vector().norm('l2')
        c.vector()[:] = np.random.randn(len(c.vector().array()))
        normc = c.vector().norm('l2')
        dl.assign(ab.sub(0), a)
        dl.assign(ab.sub(1), b)
        dl.assign(ab.sub(2), c)
        ab.vector().axpy(1.0, add.vector())
        ab.vector().axpy(-1.0, add.vector())
        aba, abb, abc = ab.split(deepcopy=True)
        a.vector().axpy(1.0, adda.vector())
        a.vector().axpy(-1.0, adda.vector())
        b.vector().axpy(1.0, addb.vector())
        b.vector().axpy(-1.0, addb.vector())
        c.vector().axpy(1.0, addc.vector())
        c.vector().axpy(-1.0, addc.vector())
        diffa = (a.vector() - aba.vector()).norm('l2') / norma
        diffb = (b.vector() - abb.vector()).norm('l2') / normb
        diffc = (c.vector() - abc.vector()).norm('l2') / normc
        if mpirank == 0:
            print 'diffa={} (|a|={}), diffb={} (|b|={}), diffc={} (|c|={})'.format(\
            diffa, norma, diffb, normb, diffc, normc)



if __name__ == "__main__":
    testsplitassign()
    testsplitassigni()
    testblockdiagonal()
    testassignsplit()
    testassignspliti()


#Profiling test on ccgo1 for testsplitassign()
#
#Timer unit: 1e-06 s
#
#Total time: 0.903843 s
#File: test_splitassign.py
#Function: testsplitassign at line 9
#
#Line #      Hits         Time  Per Hit   % Time  Line Contents
#==============================================================
#     9                                           @profile
#    10                                           def testsplitassign():
#    11         1          686    686.0      0.1      mesh = dl.UnitSquareMesh(40,40)
#    12         1        74121  74121.0      8.2      V1 = dl.FunctionSpace(mesh, "Lagrange", 2)
#    13         1        27287  27287.0      3.0      V2 = dl.FunctionSpace(mesh, "Lagrange", 2)
#    14         1        50650  50650.0      5.6      V1V2 = V1*V2
#    15         1       126288 126288.0     14.0      splitassign = SplitAndAssign(V1, V2, mesh.mpi_comm())
#    16                                           
#    17         1            9      9.0      0.0      mpirank = dl.MPI.rank(mesh.mpi_comm())
#    18                                           
#    19         1       117949 117949.0     13.0      u = dl.interpolate(dl.Expression(("x[0]*x[1]", "11+x[0]+x[1]")), V1V2)
#    20         1          264    264.0      0.0      uu = dl.Function(V1V2)
#    21         1        47927  47927.0      5.3      u1, u2 = u.split(deepcopy=True)
#    22         1          489    489.0      0.1      u1v, u2v = splitassign.split(u.vector())
#    23         1        81407  81407.0      9.0      u11 = dl.interpolate(dl.Expression("x[0]*x[1]"), V1)
#    24         1        76929  76929.0      8.5      u22 = dl.interpolate(dl.Expression("11+x[0]+x[1]"), V2)
#    25         1          611    611.0      0.1      a,b,c,d= dl.norm(u1.vector()-u1v), dl.norm(u2.vector()-u2v),\
#    26         1          320    320.0      0.0      dl.norm(u1.vector()-u11.vector()), dl.norm(u2.vector()-u22.vector())
#    27         1            3      3.0      0.0      if mpirank == 0:
#    28         1          102    102.0      0.0          print 'Splitting an interpolated function:', a, b, c, d
#    29                                           
#    30         1          450    450.0      0.0      uv = splitassign.assign(u1v, u2v)
#    31         1         8557   8557.0      0.9      dl.assign(uu.sub(0), u11)
#    32         1         7801   7801.0      0.9      dl.assign(uu.sub(1), u22)
#    33         1          367    367.0      0.0      a, b = dl.norm(uv-u.vector()), dl.norm(uv-uu.vector())
#    34         1            3      3.0      0.0      if mpirank == 0:
#    35         1           52     52.0      0.0          print 'Putting it back together:', a, b
#    36                                           
#    37        11           17      1.5      0.0      for ii in xrange(10):
#    38        10        10341   1034.1      1.1          u.vector()[:] = np.random.randn(len(u.vector().array()))
#    39        10       264388  26438.8     29.3          u1, u2 = u.split(deepcopy=True)
#    40        10         1436    143.6      0.2          u1v, u2v = splitassign.split(u.vector())
#    41        10         1858    185.8      0.2          uv = splitassign.assign(u1v, u2v)
#    42        10         1830    183.0      0.2          a, b = dl.norm(u1.vector()-u1v), dl.norm(u2.vector()-u2v)
#    43        10          927     92.7      0.1          c = dl.norm(uv-u.vector())
#    44        10           16      1.6      0.0          if mpirank == 0:
#    45        10          423     42.3      0.0              print 'Splitting random numbers:', a, b
#    46        10          335     33.5      0.0              print 'Putting it back together:', c
#

