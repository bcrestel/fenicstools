"""
Simple code to test performance of meshing in parallel
Times are real time from Unix-time command, minimum of 3 runs

100x100
Serial (ccgo1):
    UnitSquareMesh = 2.732s
    RectangleMesh = 2.725s
    msrh.Rectangle = 2.789s

Parallel 16 cores (ccgo1):
    UnitSquareMesh = 2.691s
    RectangleMesh = 2.548s
    msrh.Rectangle = 2.886s

1000x1000
Serial (ccgo1):
    UnitSquareMesh = 30.083s
    RectangleMesh = 30.216s
    msrh.Rectangle = 53.257s

Parallel 16 cores (ccgo1):
    UnitSquareMesh = 6.686s
    RectangleMesh = 6.537s
    msrh.Rectangle = 23.670s
"""

import dolfin as dl
from mshr import Rectangle, generate_mesh

#mesh = dl.UnitSquareMesh(1000, 1000)
#mesh = dl.RectangleMesh(dl.Point(0.,0.), dl.Point(1.,1.), 1000, 1000)
domain= Rectangle(dl.Point(0.,0.), dl.Point(1.,1.))
mesh = generate_mesh(domain, 893)

V = dl.FunctionSpace(mesh, 'Lagrange', 3)
test, trial = dl.TestFunction(V), dl.TrialFunction(V)
u = dl.interpolate(dl.Expression('x[0]*x[1]'), V)
M = dl.assemble( dl.inner(test, trial)*dl.dx )
print u.vector().size()    # only to check size of mesh
