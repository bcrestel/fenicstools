"""
Simple code to test performance of meshing in parallel

Serial (ccgo1):
    UnitSquareMesh
    RectangleMesh
    msrh.Rectangle
"""

import dolfin as dl
from mshr import Rectangle, generate_mesh

mesh = dl.UnitSquareMesh(100, 100)
#mesh = dl.RectangleMesh(dl.Point(0.,0.), dl.Point(1.,1.), 100, 100)
#domain= Rectangle(dl.Point(0.,0.), dl.Point(1.,1.))
#mesh = generate_mesh(domain, 90)

V = dl.FunctionSpace(mesh, 'Lagrange', 3)
test, trial = dl.TestFunction(V), dl.TrialFunction(V)
u = dl.interpolate(dl.Expression('x[0]*x[1]'), V)
M = dl.assemble( dl.inner(test, trial)*dl.dx )
#print u.vector().size()    # only to check size of mesh
