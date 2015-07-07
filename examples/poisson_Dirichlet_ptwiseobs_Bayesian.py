"""Bayesian inference of diffusion parameter m in Poisson problem
-div(m grad u) = f with zero-Dirichlet boundary conditions.
We use the Bilaplacian as a prior"""

import sys
from os.path import splitext, isdir
from shutil import rmtree
import numpy as np
from dolfin import UnitSquareMesh, FunctionSpace, Constant, DirichletBC, \
Expression, interpolate, Function
mycomm = None
myrank = 0
from fenicstools.objectivefunctional import ObjFctalElliptic
from fenicstools.observationoperator import ObsPointwise
from fenicstools.prior import BilaplacianPrior 
from fenicstools.optimsolver import checkgradfd, checkhessfd, \
bcktrcklinesearch, compute_searchdirection
from fenicstools.miscfenics import apply_noise
from fenicstools.postprocessor import PostProcessor
from fenicstools.plotfenics import PlotFenics


mesh = UnitSquareMesh(20,20)
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
mtrue = interpolate(mtrue_exp, Vme)
f = Expression("1.0")

if myrank == 0: print 'Compute target data'.format(myrank)
obspts = [[ii/5.,jj/5.] for ii in range(1,5) for jj in range(1,5)]
noisepercent = 0.10   # e.g., 0.02 = 2% noise level
ObsOp = ObsPointwise({'V': V, 'Points':obspts,'noise':noisepercent}, mycomm)
goal = ObjFctalElliptic(V, Vme, bc, bc, [f], ObsOp, [], [], [], False, mycomm)
goal.update_m(mtrue)
goal.solvefwd()
UDnoise = goal.U

if myrank == 0: print 'Define prior and sample from prior'
m0 = Function(Vm)
m0.vector()[:] = 1.0
myprior = BilaplacianPrior({'Vm':Vm,'gamma':10.,'beta':10.,'m0':m0})
filename, ext = splitext(sys.argv[0])
if isdir(filename + '/'):   rmtree(filename + '/')
plotprior = PlotFenics(filename + '/Prior/')
plotprior.set_varname('m_prior')
for ii in range(10):
    mypriorsample = myprior.sample()
    plotprior.plot_vtk(mypriorsample, ii)
plotprior.gather_vtkplots()
