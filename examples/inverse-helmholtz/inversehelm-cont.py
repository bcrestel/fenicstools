"""
Medium parameter reconstruction example for Poisson pb with zero-Dirichlet
boundary conditions and pointwise observations over the entire domain.
We solve
    arg min_m 1/2||u - u_e||^2 + R(m)
    where - Delta u = f, on Omega
    with u = 0 on boundary.
Use different regularization functional R(m)
"""
import sys
from os.path import splitext, isdir
from shutil import rmtree
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import dolfin as dl
from dolfin import MPI, mpi_comm_world
dl.set_log_active(False)

from fenicstools.objectivefunctional import ObjFctalHelmholtz
from fenicstools.plotfenics import PlotFenics
from fenicstools.prior import LaplacianPrior
from fenicstools.regularization import TV, TVPD
from fenicstools.observationoperator import ObsEntireDomain
from fenicstools.optimsolver import checkgradfd_med, checkhessfd_med
from fenicstools.miscfenics import setfct
from fenicstools.sourceterms import PointSources

mpicomm = mpi_comm_world()
mpirank = MPI.rank(mpicomm)
mpisize = MPI.size(mpicomm)


#### Set-up
PLOT = True
CHECK = False

nxy = 100 
mesh = dl.UnitSquareMesh(nxy, nxy)
V = dl.FunctionSpace(mesh, 'Lagrange', 2)  # space for state and adjoint variables
Vm = dl.FunctionSpace(mesh, 'Lagrange', 1) # space for medium parameter
Vme = dl.FunctionSpace(mesh, 'Lagrange', 1)    # sp for target med param

def u0_boundary(x, on_boundary):
    return on_boundary
u0 = dl.Constant("0.0")
bc = dl.DirichletBC(V, u0, u0_boundary)
#bc = None

#mtrue_exp = dl.Expression('2 + 7*(pow(pow(x[0] - 0.5,2) + pow(x[1] - 0.5,2),0.5) > 0.25)')
mtrue_exp = dl.Expression('1.0')
mtrue = dl.interpolate(mtrue_exp, Vme) # target medium
mtrueVm = dl.interpolate(mtrue_exp, Vm) # target medium
minit_exp = dl.Expression('1.0')
minit = dl.interpolate(minit_exp, Vm) 
#f = [dl.Expression("1.0")]   # source term
f = PointSources(V, [[0.5, 0.5]]).PtSrc

if PLOT:
    filename, ext = splitext(sys.argv[0])
    if mpirank == 0 and isdir(filename + '/'):   
        rmtree(filename + '/')
    MPI.barrier(mpicomm)
    myplot = PlotFenics(filename)
    MPI.barrier(mpicomm)
    myplot.set_varname('m_target')
    myplot.plot_vtk(mtrue)
    myplot.set_varname('m_targetVm')
    myplot.plot_vtk(mtrueVm)
else:   myplot = None

if mpirank == 0:    print 'Compute noisy data'
ObsOp = ObsEntireDomain({'V': V}, mpicomm)
ObsOp.noise = False
goal = ObjFctalHelmholtz(V, Vme, bc, bc, f, ObsOp, Data={'k':1.0}, plot=False, mycomm=mpicomm)
goal.update_m(mtrue)
goal.solvefwd()
# noise
np.random.seed(11)
noisepercent = 0.02   # e.g., 0.02 = 2% noise level
UD = goal.U[0]
rndnb = np.random.randn(UD.size)
rndnb = rndnb / np.linalg.norm(rndnb)
noiseres = noisepercent*np.linalg.norm(UD)
UDnoise = UD + rndnb*noiseres
if mpirank == 0:    print 'noiseres={}, rndnb*noiseres={}'.format(\
MPI.sum(mpicomm, noiseres), MPI.sum(mpicomm, np.linalg.norm(rndnb*noiseres)))
if PLOT:
    myplot.set_varname('u_target')
    myplot.plot_vtk(goal.u)
    setfct(goal.u, UDnoise)
    myplot.set_varname('d_target')
    myplot.plot_vtk(goal.u)

# Define regularization:
# Tikhonov
Regul = LaplacianPrior({'Vm':Vm,'gamma':1e-2,'beta':1e-2, 'm0':9.0})
# Total Variation
#   full TV w/o primal-dual
#Regul = TV({'Vm':Vm, 'eps':dl.Constant(1.0), 'GNhessian':False})
#   GN Hessian for TV w/o primal-dual
#Regul = TV({'Vm':Vm, 'eps':dl.Constant(1e-4), 'GNhessian':True})
#   full TV w/ primal-dual
#Regul = TVPD({'Vm':Vm, 'eps':dl.Constant(1.0)})

ObsOp.noise = False
InvPb = ObjFctalHelmholtz(V, Vm, bc, bc, f, ObsOp, [UDnoise], Regul, {'k':1.0}, False, mpicomm)
InvPb.regparam = 1.0
InvPb.update_m(mtrueVm)
InvPb.solvefwd_cost()
if mpirank == 0:
    print 'Objective at MAP point: misfit={} ({:.2f}), regularization={}'.format(\
    InvPb.misfit, \
    np.sqrt(InvPb.misfit/ObsOp.costfct(InvPb.UD[0], np.zeros(InvPb.UD[0].shape))), \
    InvPb.regul)


#### check gradient and Hessian against finite-difference ####
nbcheck = 5
MedPert = np.zeros((nbcheck, InvPb.m.vector().local_size()))
for ii in range(nbcheck):
    smoothperturb = dl.Expression('sin(n*pi*x[0])*sin(n*pi*x[1])', n=ii+1)
    smoothperturb_fn = dl.interpolate(smoothperturb, Vm)
    MedPert[ii,:] = smoothperturb_fn.vector().array()

if CHECK:
    if mpirank == 0:    print 'Check gradient and Hessian against finite-difference'

    #InvPb.update_m(mtrueVm)
    #InvPb.update_m(1.0)
    InvPb.update_m(dl.interpolate(dl.Expression("sin(x[0])*sin(x[1])"), Vm))
    InvPb.regparam = 0.0
    InvPb.solvefwd_cost()
    InvPb.solveadj_constructgrad()

    checkgradfd_med(InvPb, MedPert, 1e-6, [1e-2, 1e-3, 1e-4, 1e-5, 1e-6], True, mpicomm)
    print ''
    #checkhessfd_med(InvPb, MedPert, 1e-6, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], False, mpicomm)
    checkhessfd_med(InvPb, MedPert, 1e-6, [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], False, mpicomm)

    sys.exit(0)


#### Solve inverse problem
if mpirank == 0:    print 'Solve inverse problem'
InvPb.inversion(minit, mtrueVm, mpicomm, {'maxnbNewtiter':5000}, myplot=myplot)

"""
InvPb.regparam = 0.0
InvPb.solvefwd_cost()
InvPb.solveadj_constructgrad()

checkgradfd_med(InvPb, MedPert, 1e-6, [1e-4, 1e-5, 1e-6], True, mpicomm)
print ''
checkhessfd_med(InvPb, MedPert, 1e-6, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], False, mpicomm)
"""

