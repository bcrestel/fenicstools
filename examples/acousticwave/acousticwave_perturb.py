"""
Forward problem close to inverse problem sitation. Homogeneous medium with
perturbation in the middle. Neumann boundary conditions all around.
"""

import sys
from os.path import splitext, isdir
from shutil import rmtree

from fenicstools.plotfenics import PlotFenics
from fenicstools.acousticwave import AcousticWave
from fenicstools.sourceterms import PointSources, RickerWavelet
try:
    from dolfin import UnitSquareMesh, FunctionSpace, Constant, DirichletBC, \
    interpolate, Expression, Function, SubDomain, MPI, mpi_comm_world
    mycomm = mpi_comm_world()
    myrank = MPI.rank(mycomm)
except:
    from dolfin import UnitSquareMesh, FunctionSpace, Constant, DirichletBC, \
    interpolate, Expression, Function, SubDomain
    mycomm = None
    myrank = 0


# Inputs:
Nxy = 10
mesh = UnitSquareMesh(Nxy, Nxy, "crossed")
h = 1./Nxy
Vl = FunctionSpace(mesh, 'Lagrange', 1)
r = 2
Dt = 2e-3   #Dt = h/(r*alpha*c_max)
tf = 6.0

# Source term:
fpeak = .4 # .4Hz => up to 10Hz in input signal
Ricker = RickerWavelet(fpeak, 1e-10)

# Boundary conditions:
#class AllFour(SubDomain):
#    def inside(self, x, on_boundary):
#        return on_boundary

V = FunctionSpace(mesh, 'Lagrange', r)
Pt = PointSources(V, [[.5,1.]])
mydelta = Pt[0].array()
def mysrc(tt):
    return Ricker(tt)*mydelta
# Computation:
if myrank == 0: print '\n\th = {}, Dt = {}'.format(h, Dt)
Wave = AcousticWave({'V':V, 'Vl':Vl, 'Vr':Vl})
Wave.verbose = True
Wave.timestepper = 'backward'
Wave.lump = True
#Wave.set_abc(mesh, AllFour(), True)
lambda_target = Expression('1.0 + 0.0*(' \
'(x[0]>=0.3)*(x[0]<=0.7)*(x[1]>=0.3)*(x[1]<=0.7) +' \
'((x[0]-0.2)*10*(x[0]>0.2)*(x[0]<0.3) +' \
'(-x[0]+0.8)*10*(x[0]>0.7)*(x[0]<0.8))*(x[1]>0.2)*(x[1]<0.8) +' \
'((x[1]-0.2)*10*(x[1]>0.2)*(x[1]<0.3) +' \
'(-x[1]+0.8)*10*(x[1]>0.7)*(x[1]<0.8))*(x[0]>0.2)*(x[0]<0.8))')    # square perturbation in the middle
lambda_target_fn = interpolate(lambda_target, Vl)
Wave.update({'lambda':lambda_target_fn, 'rho':1.0, \
't0':0.0, 'tf':tf, 'Dt':Dt, 'u0init':Function(V), 'utinit':Function(V)})
Wave.ftime = mysrc
sol, tmp = Wave.solve()
if not mycomm == None:  MPI.barrier(mycomm)
#TODO: Add observation and check with paraview
#TODO: Add observation along with fading filter

# Plots:
try:
    boolplot = int(sys.argv[1])
except:
    boolplot = 50
if boolplot > 0:
    filename, ext = splitext(sys.argv[0])
    filename = filename + '0'
    if myrank == 0: 
        if isdir(filename + '/'):   rmtree(filename + '/')
    if not mycomm == None:  MPI.barrier(mycomm)
    myplot = PlotFenics(filename)
    myplot.set_varname('p')
    plotp = Function(V)
    for index, pp in enumerate(sol):
        if index%boolplot == 0:
            plotp.vector()[:] = pp[0]
            myplot.plot_vtk(plotp, index)
    myplot.gather_vtkplots()
    # Plot medium
    myplot.set_varname('lambda')
    myplot.plot_vtk(lambda_target_fn)
#TODO: Write python script to take different between homogeneous results (in
#/ac..._perturb0/) and perturbed medium (in /ac.._perturb/).
# Solutions, in vtu files are stored in the last section DataArray.
