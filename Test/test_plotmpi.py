import dolfin as dl
from fenicstools.plotfenics import PlotFenics
from fenicstools.miscfenics import setfct


mesh = dl.UnitSquareMesh(40,40)
V = dl.FunctionSpace(mesh, 'Lagrange', 1)
myplot = PlotFenics()
myplot.set_varname('u')
u = dl.Function(V)
for ii in range(10):
    setfct(u, dl.interpolate(dl.Constant(ii), V))
    myplot.plot_vtk(u, ii)
myplot.gather_vtkplots()
