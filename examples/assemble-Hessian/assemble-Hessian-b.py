"""
Compute MAP point then assemble the data-misfit part of the Hessian
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

from petsc4py import PETSc

from fenicstools.plotfenics import PlotFenics
from fenicstools.objectiveacoustic import ObjectiveAcoustic
from fenicstools.prior import LaplacianPrior
from fenicstools.miscfenics import checkdt
from fenicstools.acousticwave import AcousticWave
from fenicstools.sourceterms import PointSources, RickerWavelet
from fenicstools.observationoperator import TimeObsPtwise
from fenicstools.linalg.miscroutines import setglobalvalue, setupPETScmatrix

outputdirectory = '/workspace/ben/fenics/assemble-Hessian/'
#outputdirectory = ''

mpicomm = mpi_comm_world()
mpirank = MPI.rank(mpicomm)
mpisize = MPI.size(mpicomm)

# SAVE_MAP = True only computes the MAP point and print to file
# SAVE_MAP = False loads the MAP point and compute the Hessian at that point
SAVE_MAP = False
PLOT = False

# Input data:
Data={
'4.0': [4.0, 50, 5e-4, [0.0, 0.04, 2.0, 2.04]], \
'8.0': [8.0, 50, 5e-4, [0.0, 0.04, 2.0, 2.04]], \
'14.0': [14.0, 140, 5e-4, [0.0, 0.02, 1.5, 1.52]]
}
freq, Nxy, Dt, t0tf = Data['8.0']
t0, t1, t2, tf = t0tf
skip = int(0.1/Dt)
checkdt(Dt, 1./Nxy, 2, np.sqrt(2.0), True)

# mesh
if mpirank == 0:    print 'meshing'
mesh = dl.UnitSquareMesh(Nxy, Nxy)
Vm = dl.FunctionSpace(mesh, 'Lagrange', 1)
V = dl.FunctionSpace(mesh, 'Lagrange', 2)

# target medium:
b_target = dl.Expression(\
'1.0 + 1.0*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
b_target_fn = dl.interpolate(b_target, Vm)
a_target = dl.Expression('1.0')
a_target_fn = dl.interpolate(a_target, Vm)

# set up plots:
filename, ext = splitext(sys.argv[0])
myplot = PlotFenics(outputdirectory + filename + str(freq))
MPI.barrier(mpicomm)
if PLOT:
    myplot.set_varname('b_target')
    myplot.plot_vtk(b_target_fn)
else:
    myplot = None

# observations:
obspts = [[0.0, ii/10.] for ii in range(1,10)] + \
[[1.0, ii/10.] for ii in range(1,10)] + \
[[ii/10., 0.0] for ii in range(1,10)]
#obspts = [[0.9, 0.1]]

# define pde operator:
if mpirank == 0:    print 'define wave pde'
wavepde = AcousticWave({'V':V, 'Vm':Vm})
wavepde.timestepper = 'backward'
wavepde.lump = True

# source:
if mpirank == 0:    print 'sources'
srcloc = [[ii/10., 1.0] for ii in range(1,10,2)]
#srcloc = [[0.1, 0.9]]
Ricker = RickerWavelet(freq, 1e-10)
Pt = PointSources(V, srcloc)
src = dl.Function(V)
srcv = src.vector()
mysrc = [Ricker, Pt, srcv]

# define objective function:
wavepde.update({'a':a_target_fn, 'b':b_target_fn, \
't0':t0, 'tf':tf, 'Dt':Dt, 'u0init':dl.Function(V), 'utinit':dl.Function(V)})
regul = LaplacianPrior({'Vm':Vm,'gamma':1e-4,'beta':1e-4, 'm0':1.0})
waveobj = ObjectiveAcoustic(wavepde, mysrc, 'b', regul)
obsop = TimeObsPtwise({'V':V, 'Points':obspts}, t0tf)
waveobj.obsop = obsop

# Save MAP point, or assemble Hessian at MAP point
if SAVE_MAP:
    # noisy data
    if mpirank == 0:    print 'generate noisy data'
    waveobj.solvefwd()
    DD = waveobj.Bp[:]
    noiselevel = 0.1   # = 10%
    for ii, dd in enumerate(DD):
        np.random.seed(11)
        nbobspt, dimsol = dd.shape
        sigmas = np.sqrt((dd**2).sum(axis=1)/dimsol)*noiselevel
        rndnoise = np.random.randn(nbobspt*dimsol).reshape((nbobspt, dimsol))
        DD[ii] = dd + sigmas.reshape((len(sigmas),1))*rndnoise
    waveobj.dd = DD
    if mpirank == 0:
        np.save(outputdirectory + filename + str(freq) + '/dd.npy', np.array(DD))
    waveobj.solvefwd_cost()
    if mpirank == 0:
        print 'noise misfit={}, regul cost={}, ratio={}'.format(waveobj.cost_misfit, \
        waveobj.cost_reg, waveobj.cost_misfit/waveobj.cost_reg)
    if PLOT:    
        plotindex = len(waveobj.solfwd)/2
        myplot.plot_timeseries(waveobj.solfwd[plotindex], 'pd', 0, skip, dl.Function(V))

    # solve inverse problem
    if mpirank == 0:    print 'Solve inverse problem'
    waveobj.inversion(b_target_fn, b_target_fn, mpicomm, myplot=myplot)
    if PLOT:
        myplot.set_varname('b_MAP')
        myplot.plot_vtk(waveobj.PDE.b)

    # Save MAP point
    fileout = dl.HDF5File(mpicomm, \
    outputdirectory + filename + str(freq) + '/MAP.h5', 'w')
    fileout.write(waveobj.PDE.b, 'b')

else:
    # Load MAP point
    fileout = dl.HDF5File(mpicomm, \
    outputdirectory + filename + str(freq) + '/MAP.h5', 'r')
    binit = dl.Function(Vm)
    fileout.read(binit, 'b')
    waveobj.update_PDE({'b':binit})

    # Load data
    DDarr = np.load(outputdirectory + filename + str(freq) + '/dd.npy')
    DD = []
    for d in DDarr:
        DD.append(d)
    waveobj.dd = DD

    # Compute gradient
    if mpirank == 0:    print 'Compute gradient'
    waveobj.solvefwd_cost()
    waveobj.solveadj_constructgrad()
    # Print output for comparison with MAP computation
    gradnorm = waveobj.MGv.inner(waveobj.Grad.vector())
    diff = waveobj.PDE.b.vector() - b_target_fn.vector()
    medmisfit = diff.inner(waveobj.Mass*diff)
    dtruenorm = b_target_fn.vector().inner(waveobj.Mass*b_target_fn.vector())
    print '\t{:10s} {:12s} {:12s} {:12s} {:10s} \t{:10s} {:12s} {:12s}'.format(\
    'cost', 'misfit', 'reg', '|G|', 'medmisf', 'a_ls', 'tol_cg', 'n_cg')
    print '{:12.4e} {:12.2e} {:12.2e} {:11.4e} {:10.2e} ({:4.2f})'.\
    format(waveobj.cost, waveobj.cost_misfit, waveobj.cost_reg, \
    gradnorm, medmisfit, medmisfit/dtruenorm)

    # Parallelism or hand-made parallelism?
    try:
        myrank = int(sys.argv[1])   # starts at 0
        if myrank > 0:  PLOT = False
        mysize = int(sys.argv[2])   # total nb of processes
        assert mpisize == 1, \
        "cannot run hand-made parallelims and MPI parallelism at the same time"
        assert myrank < mysize

        a = myrank*(Vm.dim()/mysize)
        if myrank+1 < mysize:
            b = (myrank+1)*(Vm.dim()/mysize)+5
        else:
            b = Vm.dim()
        myrange = range(Vm.dim())[a:b]
        print 'Hand-made parallelism: myrank={} (out of {}), myrange=[{}:{}]'.format(\
        myrank, mysize, a, b)
        Hessfilename = outputdirectory + 'Hessian' + str(freq) \
        + '_' + str(a) + '-' + str(b) + '.dat'
    except:
        myrange = xrange(Vm.dim())
        Hessfilename = outputdirectory + 'Hessian' + str(freq) + '.dat'

    # Assemble data Hessian
    if mpirank == 0:    print 'Assemble data misfit part of the Hessian'
    waveobj.alpha_reg = 0.0
    Hei, ei = dl.Function(Vm), dl.Function(Vm)
    Hessian, VrDM, VcDM = setupPETScmatrix(Vm, Vm, 'dense', mpicomm)
    for ii in myrange:
        if ii%100 == 0 and mpirank == 0:    print 'ii={} out of {}'.format(ii, Vm.dim())
        ei.vector().zero()
        setglobalvalue(ei, ii, 1.0)
        waveobj.mult(ei.vector(), Hei.vector())
        normvec = Hei.vector().norm('l2')
        Hei_arr = Hei.vector().array()
        cols = np.where(np.abs(Hei_arr) > 1e-16*normvec)[0]
        for cc, val in zip(cols, Hei_arr[cols]):  
            global_cc = VcDM.dofs()[cc]
            Hessian[ii, global_cc] = val
    Hessian.assemblyBegin()
    Hessian.assemblyEnd()

    # Print Hessian to file
    if mpirank == 0:    print 'Print Hessian to file'
    myviewer = PETSc.Viewer().createBinary(Hessfilename, \
    mode='w', format=PETSc.Viewer.Format.NATIVE, comm=mpicomm)
    myviewer(Hessian)
