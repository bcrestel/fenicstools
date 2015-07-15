"""
Solve forward acoustic wave equation with dashpot absorbing boundary conditions
"""

from dolfin import UnitSquareMesh, FunctionSpace
from fenicstools.pdesolver import Wave

mesh = UnitSquareMesh(15,15)
V = FunctionSpace(mesh, 'Lagrange', 2)  # space for solution
Vl = FunctionSpace(mesh, 'Lagrange', 1) # space for medium param lambda and rho

PWave = Wave({'V':V, 'Vl':Vl, 'Vr':Vl})
PWave.update({'lambda':1.0, 'rho':1.0, 't0':0.0, 'tf':1.0, 'Dt':1e-2})
PWave.definesource({'type':'delta', 'point':[.5,.5]}, lambda t: float(t==0.0))
