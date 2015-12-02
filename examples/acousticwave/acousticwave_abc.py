import sys
from os.path import splitext, isdir
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt

from fenicstools.plotfenics import PlotFenics
from fenicstools.acousticwave import AcousticWave
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

NN = np.array((10, 20, 30, 40))
ERROR = []

# Medium ppties:
lam = 8.0
rho = 2.0
c = np.sqrt(lam/rho)
tf = 0.5/c # Final time
direction = 0
u0_expr = Expression(\
'100*pow(x[i]-.25,2)*pow(x[i]-0.75,2)*(x[i]<=0.75)*(x[i]>=0.25)', i=direction)
exact_expr = Expression(\
'50*pow(x[i]+.25,2)*pow(x[i]-0.25,2)*(x[i]<=0.25) + ' + \
'50*pow(x[i]-.75,2)*pow(x[i]-1.25,2)*(x[i]>=0.75)', i=direction)
class LeftRight(SubDomain):
    def inside(self, x, on_boundary):
        return (x[direction] < 1e-16 or x[direction] > 1.0 - 1e-16) \
        and on_boundary

q = 3
if myrank == 0: print '\npolynomial order = {}'.format(q)

for Nxy in NN:
    h = 1./Nxy
    mesh = UnitSquareMesh(Nxy, Nxy, "crossed")
    V = FunctionSpace(mesh, 'Lagrange', q)
    Vl = FunctionSpace(mesh, 'Lagrange', 1)
    Dt = h/(q*11.*c)
    if myrank == 0: print '\n\th = {}, Dt = {}'.format(h, Dt)

    Wave = AcousticWave({'V':V, 'Vl':Vl, 'Vr':Vl})
    #Wave.verbose = True
    Wave.timestepper = 'centered'
    Wave.lump = True
    Wave.lumpD = True
    Wave.set_abc(mesh, LeftRight(), True)
    Wave.exact = interpolate(exact_expr, V)
    Wave.update({'lambda':lam, 'rho':rho, 't0':0.0, 'tf':tf, 'Dt':Dt,\
    'u0init':interpolate(u0_expr, V), 'utinit':Function(V)})
    sol, error = Wave.solve()
    ERROR.append(error)
    if myrank == 0: print 'relative error = {:.5e}'.format(error)
    if not mycomm == None:  MPI.barrier(mycomm)

if myrank == 0:
    # Order of convergence:
    CONVORDER = []
    for ii in range(len(ERROR)-1):
        CONVORDER.append(np.log(ERROR[ii+1]/ERROR[ii])/np.log((1./NN[ii+1])/(1./NN[ii])))
    print '\n\norder of convergence:', CONVORDER

# Save plots:
try:
    boolplot = int(sys.argv[1])
except:
    boolplot = 0
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
    # convergence plot:
    if myrank == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.loglog(1./NN, ERROR, '-o')
        ax.set_xlabel('h')
        ax.set_ylabel('error')
        fig.savefig(filename + '/convergence.eps')

