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
Dt = 2e-3   #Dt = h/(r*alpha)
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
#TODO: need to define lambda medium with perturbation in the middle
Wave.update({'lambda':1.0, 'rho':1.0, 't0':0.0, 'tf':tf, 'Dt':Dt,\
'u0init':Function(V), 'utinit':Function(V)})
Wave.ftime = mysrc
sol, tmp = Wave.solve()
if not mycomm == None:  MPI.barrier(mycomm)

# Plots:
try:
    boolplot = int(sys.argv[1])
except:
    boolplot = 50
if boolplot > 0:
    filename, ext = splitext(sys.argv[0])
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
