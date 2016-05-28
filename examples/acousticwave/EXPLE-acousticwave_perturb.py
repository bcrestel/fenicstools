"""
Forward problem close to inverse problem situation. Homogeneous medium with
perturbation in the middle. Neumann boundary conditions all around.
"""
PARALLEL = False

import sys
import numpy as np
import matplotlib.pyplot as plt
from os.path import splitext, isdir
from shutil import rmtree

from fenicstools.plotfenics import PlotFenics
from fenicstools.acousticwave import AcousticWave
from fenicstools.sourceterms import PointSources, RickerWavelet
from fenicstools.observationoperator import TimeObsPtwise, TimeFilter
from fenicstools.miscfenics import checkdt, setfct
from dolfin import UnitSquareMesh, FunctionSpace, Constant, DirichletBC, \
interpolate, Expression, Function, SubDomain
if PARALLEL:
    from dolfin import MPI, mpi_comm_world
    mycomm = mpi_comm_world()
    myrank = MPI.rank(mycomm)
else:
    myrank = 0


# Inputs:
fpeak = 4.0 # 4Hz => up to 10Hz in input signal
Nxy = 100
Dt = 5e-4   #Dt = h/(r*alpha*c_max)
tf = 1.0
mytf = TimeFilter([0.0,0.2,0.8,tf])

#fpeak = .4 # 4Hz => up to 10Hz in input signal
#Nxy = 10
#Dt = 1e-4   #Dt = h/(r*alpha*c_max)
#tf = 7.0
#mytf = TimeFilter([0.,1.,6.,tf])

mesh = UnitSquareMesh(Nxy, Nxy)
h = 1./Nxy
Vl = FunctionSpace(mesh, 'Lagrange', 1)
r = 2
checkdt(Dt, h, r, 2.0, False)
# Source term:
Ricker = RickerWavelet(fpeak, 1e-10)

# Boundary conditions:
#class AllFour(SubDomain):
#    def inside(self, x, on_boundary):
#        return on_boundary

V = FunctionSpace(mesh, 'Lagrange', r)
Pt = PointSources(V, [[.5,1.]])
mydelta = Pt[0]
src = Function(V)
srcv = src.vector()
def mysrc(tt):
    srcv.zero()
    srcv.axpy(Ricker(tt), mydelta)
    return srcv
# Computation:
if myrank == 0: print '\n\th = {}, Dt = {}'.format(h, Dt)
Wave = AcousticWave({'V':V, 'Vm':Vl})
#Wave.verbose = True
Wave.timestepper = 'backward'
#Wave.lump = True
#Wave.set_abc(mesh, AllFour(), True)
lambda_target = Expression('1.0 + 3.0*(' \
'(x[0]>=0.3)*(x[0]<=0.7)*(x[1]>=0.3)*(x[1]<=0.7))') 
lambda_target_fn = interpolate(lambda_target, Vl)
Wave.update({'b':lambda_target_fn, 'a':1.0, \
't0':0.0, 'tf':tf, 'Dt':Dt, 'u0init':Function(V), 'utinit':Function(V)})
Wave.ftime = mysrc
sol, tmp = Wave.solve()
if not PARALLEL:
    # Observations
    myObs = TimeObsPtwise({'V':V, 'Points':[[.5,.2], [.5,.8], [.2,.5], [.8,.5]]})
    Bp = np.zeros((4, len(sol)))
    mytimes = np.zeros(len(sol))
    solp = Function(V)
    for index, pp in enumerate(sol):
        setfct(solp, pp[0])
        Bp[:,index] = myObs.obs(solp)
        mytimes[index] = pp[1]
    Bpf = Bp*mytf.evaluate(mytimes)


# Plots:
try:
    boolplot = int(sys.argv[1])
except:
    boolplot = 20
if boolplot > 0:
    filename, ext = splitext(sys.argv[0])
    #filename = filename + '0'
    if myrank == 0: 
        if isdir(filename + '/'):   rmtree(filename + '/')
    if PARALLEL:    MPI.barrier(mycomm)
    myplot = PlotFenics(filename)
    plotp = Function(V)
    myplot.plot_timeseries(sol, 'p', 0, boolplot, plotp)
#    myplot.set_varname('p')
#    for index, pp in enumerate(sol):
#        if index%boolplot == 0:
#            setfct(plotp, pp[0])
#            myplot.plot_vtk(plotp, index)
#    myplot.gather_vtkplots()
    # Plot medium
    myplot.set_varname('lambda')
    myplot.plot_vtk(lambda_target_fn)
    if not PARALLEL:
        # Plot observations
        fig = plt.figure()
        for ii in range(Bp.shape[0]):
            ax = fig.add_subplot(2,2,ii+1)
            ax.plot(mytimes, Bp[ii,:], 'b')
            ax.plot(mytimes, Bpf[ii,:], 'r--')
            ax.set_title('Plot'+str(ii))
        fig.savefig(filename + '/observations.eps')
