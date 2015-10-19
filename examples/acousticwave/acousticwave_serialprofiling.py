import sys
from os.path import splitext, isdir
from shutil import rmtree
import numpy as np

from fenicstools.plotfenics import PlotFenics
from fenicstools.acousticwave import AcousticWave
from dolfin import UnitSquareMesh, FunctionSpace, Constant, DirichletBC, \
interpolate, Expression, Function, SubDomain

def RunAcoustic(q):
    Nxy = 100
    tf = 0.1 # Final time
    direction = 0
    u0_expr = Expression(\
    '100*pow(x[i]-.25,2)*pow(x[i]-0.75,2)*(x[i]<=0.75)*(x[i]>=0.25)', i=direction)
    class LeftRight(SubDomain):
        def inside(self, x, on_boundary):
            return (x[direction] < 1e-16 or x[direction] > 1.0 - 1e-16) \
            and on_boundary
    h = 1./Nxy
    mesh = UnitSquareMesh(Nxy, Nxy, "crossed")
    V = FunctionSpace(mesh, 'Lagrange', q)
    Vl = FunctionSpace(mesh, 'Lagrange', 1)
    Dt = h/(q*10.)

    Wave = AcousticWave({'V':V, 'Vl':Vl, 'Vr':Vl})
    Wave.verbose = True
    Wave.lump = True
    Wave.set_abc(mesh, LeftRight())
    Wave.update({'lambda':1.0, 'rho':1.0, 't0':0.0, 'tf':tf, 'Dt':Dt,\
    'u0init':interpolate(u0_expr, V), 'utinit':Function(V)})
    sol, tmp = Wave.solve()


if __name__ == "__main__":
    q = 5
    RunAcoustic(q)
