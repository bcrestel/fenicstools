"""
Acoustic wave inverse problem with a single low-frequency,
and absorbing boundary conditions on left, bottom, and right.
Check gradient and Hessian for joint inverse problem a and b
"""

import sys
from os.path import splitext, isdir
from shutil import rmtree
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    pass
import time

import dolfin as dl
from dolfin import MPI

from fenicstools.plotfenics import PlotFenics, plotobservations
from fenicstools.acousticwave import AcousticWave
from fenicstools.sourceterms import PointSources, RickerWavelet
from fenicstools.observationoperator import TimeObsPtwise
from fenicstools.objectiveacoustic import ObjectiveAcoustic
from fenicstools.optimsolver import checkgradfd_med, checkhessabfd_med
from fenicstools.prior import LaplacianPrior
from fenicstools.regularization import TVPD, TV
from fenicstools.jointregularization import \
SingleRegularization, V_TVPD, SumRegularization, NuclearNormSVD2D
from fenicstools.mpicomm import create_communicators, partition_work
from fenicstools.miscfenics import createMixedFS, ZeroRegularization, computecfromab

#from fenicstools.examples.acousticwave.mediumparameters0 import \
from fenicstools.examples.acousticwave.mediumparameters1 import \
targetmediumparameters, initmediumparameters, loadparameters

dl.set_log_active(False)


# Create local and global communicators
mpicomm_local, mpicomm_global = create_communicators()
mpiworldrank = MPI.rank(dl.mpi_comm_world())
PRINT = (mpiworldrank == 0)
mpicommbarrier = dl.mpi_comm_world()


# Command-line arguments
try:
    PARAM = sys.argv[1]
except:
    PARAM = 'ab'
try:
    k = float(sys.argv[2])
except:
    k = 7e-6


##############
LARGE = False
NOISE = True
PLOTTS = False

FDGRAD = False
ALL = False
nbtest = 5
##############
Nxy, Dt, fpeak, t0, t1, t2, tf = loadparameters(LARGE)
h = 1./Nxy
if PRINT:
    print 'Nxy={} (h={}), Dt={}, fpeak={}, t0,t1,t2,tf={}'.format(\
    Nxy, h, Dt, fpeak, [t0,t1,t2,tf])


# Define PDE:
# dist is in [km]
mesh = dl.UnitSquareMesh(mpicomm_local, Nxy, Nxy)
X, Y = 1, 1 # shall not be changed
Vl = dl.FunctionSpace(mesh, 'Lagrange', 1)

# Source term:
Ricker = RickerWavelet(fpeak, 1e-6)
r = 2   # polynomial degree for state and adj
V = dl.FunctionSpace(mesh, 'Lagrange', r)
y_src = 1.0 # 1.0->reflection, 0.1->transmission
#Pt = PointSources(V, [[0.1*ii*X-0.05, y_src] for ii in range(1,11)])
Pt = PointSources(V, [[0.1,y_src], [0.25,y_src], [0.4,y_src],\
[0.6,y_src], [0.75,y_src], [0.9,y_src]])
#Pt = PointSources(V, [[0.5, y_src]])
srcv = dl.Function(V).vector()

# Boundary conditions:
class ABCdom(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[1] < Y)

Wave = AcousticWave({'V':V, 'Vm':Vl}, 
{'print':False, 'lumpM':True, 'timestepper':'backward'})
Wave.set_abc(mesh, ABCdom(), lumpD=False)


at, bt, ct,_,_ = targetmediumparameters(Vl, X)
a0, b0,_,_,_ = initmediumparameters(Vl, X)
if 'a' not in PARAM:    a0 = at
if 'b' not in PARAM:    b0 = bt
Wave.update({'b':bt, 'a':at, 't0':0.0, 'tf':tf, 'Dt':Dt,\
'u0init':dl.Function(V), 'utinit':dl.Function(V)})
if PRINT:
    print 'nb of src={}, nb of timesteps={}'.format(len(Pt.src_loc), Wave.Nt)

sources, timesteps = partition_work(mpicomm_local, mpicomm_global, \
len(Pt.src_loc), Wave.Nt)

mpilocalrank = MPI.rank(mpicomm_local)
mpiglobalrank = MPI.rank(mpicomm_global)
mpiworldsize = MPI.size(dl.mpi_comm_world())
print 'mpiworldrank={}, mpiglobalrank={}, mpilocalrank={}, sources={}, timestep=[{},{}]'.format(\
mpiworldrank, mpiglobalrank, mpilocalrank, sources,\
timesteps[0], timesteps[-1])

# observation operator:
obspts = [[ii*float(X)/float(Nxy), Y] for ii in range(1,Nxy)]
#obspts = [[0.0, ii/10.] for ii in range(1,10)] + \
#[[1.0, ii/10.] for ii in range(1,10)] + \
#[[ii/10., 0.0] for ii in range(1,10)] + \
#[[ii/10., 1.0] for ii in range(1,10)]

tfilterpts = [t0, t1, t2, tf]
obsop = TimeObsPtwise({'V':V, 'Points':obspts}, tfilterpts)

# define objective function:
if FDGRAD:
    waveobj = ObjectiveAcoustic(mpicomm_global, Wave, [Ricker, Pt, srcv],\
    sources, timesteps, PARAM)
else:
    # REGULARIZATION:
    eps = 1e-3
    if PARAM == 'ab':
        rega = TVPD({'Vm':Vl, 'eps':eps, 'k':5e-6, 'print':PRINT})
        regb = TVPD({'Vm':Vl, 'eps':eps, 'k':9e-6, 'print':PRINT})

        #regul = V_TVPD(Vl, {'eps':eps, 'k':k, 'PCGN':False, 'print':PRINT})
        regul = SumRegularization(rega, regb, \
        coeff_ncg=k, parameters_ncg={'eps':1e-6}, isprint=PRINT)
        #regul = NuclearNormSVD2D(mesh, {'eps':eps, 'k':k}, isprint=PRINT)
    else:
        reg = TVPD({'Vm':Vl, 'eps':eps, 'k':k, 'print':PRINT})
        regul = SingleRegularization(reg, PARAM, PRINT)

    waveobj = ObjectiveAcoustic(mpicomm_global, Wave, [Ricker, Pt, srcv],\
    sources, timesteps, PARAM, regul)
waveobj.obsop = obsop
#waveobj.GN = True

# Generate synthetic observations
if PRINT:    print 'generate noisy data'
waveobj.solvefwd()
DD = waveobj.Bp[:]
if NOISE:
    np.random.seed(11)
    SNRdB = 20.0   # [dB], i.e, log10(mu/sigma) = SNRdB/10
    # Generate random components for all src (even if not owned)
    RAND = []
    nbobspt = len(obspts)
    nbt = waveobj.PDE.Nt + 1
    for ii in range(len(Pt.src_loc)):
        RAND.append(np.random.randn(nbobspt*nbt).reshape((nbobspt, nbt)))
    RAND = RAND[sources[0]:sources[-1]+1]
    # Add noise
    for ii in range(len(DD)):
        if PRINT:    print 'source {}'.format(ii)
        dd = DD[ii]
        rndnoise = RAND[ii]

        mu = np.abs(dd).mean(axis=1)
        sigmas = mu/(10**(SNRdB/10.))
        #sigmas = np.sqrt((dd**2).sum(axis=1)/nbt)*0.1

        #rndnoise = np.random.randn(nbobspt*nbt).reshape((nbobspt, nbt))
        print 'mpiglobalrank={}, sigmas={}, |rndnoise|={}'.format(\
        mpiglobalrank, sigmas.sum()/len(sigmas), (rndnoise**2).sum().sum())
        DD[ii] = dd + sigmas.reshape((nbobspt,1))*rndnoise
        MPI.barrier(mpicommbarrier)
waveobj.dd = DD
if PLOTTS:
    if PRINT:
        src = int(len(Pt.src_loc)*0.5)
        print 'Plotting source #{}'.format(src)
        fig = plotobservations(waveobj.PDE.times, waveobj.Bp[src], waveobj.dd[src], 9)
        plt.show()
    MPI.barrier(mpicommbarrier)
# check:
waveobj.solvefwd_cost()
costmisfit = waveobj.cost_misfit
#assert costmisfit < 1e-14, costmisfit

# Compute gradient at initial parameters
waveobj.update_PDE({'a':a0, 'b':b0})
waveobj.solvefwd_cost()
if PRINT:    
    print 'misfit at target={:.4e}; at initial state = {:.4e}'.format(\
    costmisfit, waveobj.cost_misfit)
if PLOTTS:
    if PRINT:
        fig = plotobservations(waveobj.PDE.times, waveobj.Bp[src], waveobj.dd[src], 9)
        plt.show()
    MPI.barrier(mpicommbarrier)
    sys.exit(0)



##################################################
# Finite difference check of gradient and Hessian
if FDGRAD:
    if ALL and (PARAM == 'a' or PARAM == 'b') and PRINT:
        print '*** Warning: Single inversion but changing both parameters'
    MPa = [
    dl.Constant('1.0'), 
    dl. Expression('sin(pi*x[0])*sin(pi*x[1])', degree=10),
    dl.Expression('x[0]', degree=10), dl.Expression('x[1]', degree=10), 
    dl.Expression('sin(3*pi*x[0])*sin(3*pi*x[1])', degree=10)]
    MPb = [
    dl.Constant('1.0'), 
    dl. Expression('sin(pi*x[0])*sin(pi*x[1])', degree=10),
    dl.Expression('x[1]', degree=10), dl.Expression('x[0]', degree=10), 
    dl.Expression('sin(3*pi*x[0])*sin(3*pi*x[1])', degree=10)]

    if ALL:
        Medium = []
        tmp = dl.Function(Vl*Vl)
        for ii in range(nbtest):
            tmp.vector().zero()
            dl.assign(tmp.sub(0), dl.interpolate(MPa[ii], Vl))
            dl.assign(tmp.sub(1), dl.interpolate(MPb[ii], Vl))
            Medium.append(tmp.vector().copy())
        if PRINT:    print 'check gradient with FD'
        checkgradfd_med(waveobj, Medium, PRINT, 1e-6, [1e-5, 1e-6, 1e-7], True)
        if PRINT:    print '\ncheck Hessian with FD'
        checkhessabfd_med(waveobj, Medium, PRINT, 1e-6, [1e-5, 1e-6, 1e-7], True, 'all')
    else:
        Mediuma, Mediumb = [], []
        tmp = dl.Function(Vl*Vl)
        for ii in range(nbtest):
            tmp.vector().zero()
            dl.assign(tmp.sub(0), dl.interpolate(MPa[ii], Vl))
            Mediuma.append(tmp.vector().copy())
            tmp.vector().zero()
            dl.assign(tmp.sub(1), dl.interpolate(MPb[ii], Vl))
            Mediumb.append(tmp.vector().copy())
        if PRINT:    print 'check a-gradient with FD'
        if 'a' in PARAM:
            checkgradfd_med(waveobj, Mediuma, PRINT, 1e-6, [1e-5, 1e-6, 1e-7], True)
        else:
            checkgradfd_med(waveobj, Mediuma[:1], PRINT, 1e-6, [1e-5], True)
        if PRINT:    print 'check b-gradient with FD'
        if 'b' in PARAM:
            checkgradfd_med(waveobj, Mediumb, PRINT, 1e-6, [1e-5, 1e-6, 1e-7], True)
        else:
            checkgradfd_med(waveobj, Mediumb[:1], PRINT, 1e-6, [1e-5], True)

        if PRINT:    
            print '\n'
            print 'check a-Hessian with FD'
        if 'a' in PARAM:
            checkhessabfd_med(waveobj, Mediuma, PRINT, 1e-6, [1e-5, 1e-6, 1e-7], True, 'a')
        else:
            checkhessabfd_med(waveobj, Mediuma[:1], PRINT, 1e-6, [1e-5, 1e-6, 1e-7], True, 'a')
        if PRINT:    print 'check b-Hessian with FD'
        if 'b' in PARAM:
            checkhessabfd_med(waveobj, Mediumb, PRINT, 1e-6, [1e-5, 1e-6, 1e-7], True, 'b')
        else:
            checkhessabfd_med(waveobj, Mediumb[:1], PRINT, 1e-6, [1e-5, 1e-6, 1e-7], True, 'b')
##################################################
# Solve inverse problem
else:
    VlVl = createMixedFS(Vl, Vl)
    mt = dl.Function(VlVl)
    dl.assign(mt.sub(0), at)
    dl.assign(mt.sub(1), bt)

    m0 = dl.Function(VlVl)
    m0.vector().zero()
    m0.vector().axpy(1.0, mt.vector())
    if 'a' in PARAM:
        dl.assign(m0.sub(0), a0)
    if 'b' in PARAM:
        dl.assign(m0.sub(1), b0)

    regt = waveobj.regularization.costab(at,bt)
    reg0 = waveobj.regularization.costab(a0,b0)
    if PRINT:
        print 'Regularization at target={:.2e}, at initial state={:.2e}'.format(\
        regt, reg0)

    myplotf = None

    if PRINT:
        print '\n\nStart solution of inverse problem for parameter(s) {}'.format(PARAM)

    parameters = {}
    parameters['solverNS'] = 'Newton'
    #parameters['solverNS'] = 'BFGS'
    parameters['isprint'] = PRINT
    parameters['nbGNsteps'] = 20
    parameters['checkab'] = 5
    parameters['maxiterNewt'] = 5000
    parameters['maxtolcg'] = 0.5
    parameters['avgPC'] = False
    parameters['PC'] = 'prior'

    if parameters['solverNS'] == 'BFGS':
        parameters['H0inv'] = 'default'
        parameters['memory_limit'] = np.inf
    else:
        parameters['H0inv'] = 'Rinv'

    MPI.barrier(mpicommbarrier)
    tstart = time.time()

    waveobj.inversion(m0, mt, parameters,
    boundsLS=[[1e-4, 1.0], [1e-3, 1.0]], myplot=myplotf)

    tend = time.time()
    Dt = tend - tstart
    if PRINT:
        print 'Time to solve inverse problem {}'.format(Dt)

    minat = at.vector().min()
    maxat = at.vector().max()
    minbt = bt.vector().min()
    maxbt = bt.vector().max()
    minct = ct.vector().min()
    maxct = ct.vector().max()
    mina0 = a0.vector().min()
    maxa0 = a0.vector().max()
    minb0 = b0.vector().min()
    maxb0 = b0.vector().max()
    a = waveobj.PDE.a
    b = waveobj.PDE.b
    c = computecfromab(a.vector(), b.vector())
    mina = a.vector().min()
    maxa = a.vector().max()
    minb = b.vector().min()
    maxb = b.vector().max()
    minc = c.min()
    maxc = c.max()
    cf = dl.Function(Vl)
    cf.vector().zero()
    cf.vector().axpy(1.0, c)
    # medium misfits:
    test, trial = dl.TestFunction(Vl), dl.TrialFunction(Vl)
    MM = dl.assemble(dl.inner(test, trial)*dl.dx)
    normat = np.sqrt(at.vector().inner(MM*at.vector()))
    mmfa = at.vector() - a.vector()
    norm_mmfa = np.sqrt(mmfa.inner(MM*mmfa))
    erra = 100.0*norm_mmfa/normat
    normbt = np.sqrt(bt.vector().inner(MM*bt.vector()))
    mmfb = bt.vector() - b.vector()
    norm_mmfb = np.sqrt(mmfb.inner(MM*mmfb))
    errb = 100.0*norm_mmfb/normbt
    normct = np.sqrt(ct.vector().inner(MM*ct.vector()))
    mmfc = ct.vector() - c
    norm_mmfc = np.sqrt(mmfc.inner(MM*mmfc))
    errc = 100.0*norm_mmfc/normct
    if PRINT:
        print '\ntarget: min(a)={}, max(a)={}'.format(minat, maxat)
        print 'init: min(a)={}, max(a)={}'.format(mina0, maxa0)
        print 'MAP: min(a)={}, max(a)={}'.format(mina, maxa)
        print 'med_misfit={:.4e}, err={:.1f}%'.format(norm_mmfa, erra)

        print '\ntarget: min(b)={}, max(b)={}'.format(minbt, maxbt)
        print 'init: min(b)={}, max(b)={}'.format(minb0, maxb0)
        print 'MAP: min(b)={}, max(b)={}'.format(minb, maxb)
        print 'med_misfit={:.4e}, err={:.1f}%'.format(norm_mmfb, errb)

        print '\ntarget: min(c)={}, max(c)={}'.format(minct, maxct)
        print 'MAP: min(c)={}, max(c)={}'.format(minc, maxc)
        print 'med_misfit={:.4e}, err={:.1f}%'.format(norm_mmfc, errc)

        # WARNING: only makes sense if mpicomm_local == mpi_comm_self
        # otherwise, can't be restricted to PRINT processor only
        plotfolder = PARAM + '_k' + str(k) + '_e' + str(eps)
        myplot = PlotFenics(Outputfolder='output/plots/' + plotfolder, \
        comm = mesh.mpi_comm())
        waveobj._plotab(myplot, '-map_' + PARAM + '_k' + str(k) + '_e' + str(eps))

        myplot.set_varname('c-map_' + PARAM + '_k' + str(k) + '_e' + str(eps))
        myplot.plot_vtk(cf)


    """
    # Test gradient and Hessian after several steps of Newton method
    waveobj.GN = False
    waveobj.regularization = ZeroRegularization(Vl)
    if ALL and (PARAM == 'a' or PARAM == 'b') and PRINT:
        print '*** Warning: Single inversion but changing both parameters'
    MPa = [
    dl.Constant('1.0'), 
    dl. Expression('sin(pi*x[0])*sin(pi*x[1])', degree=10),
    dl.Expression('x[0]', degree=10), dl.Expression('x[1]', degree=10), 
    dl.Expression('sin(3*pi*x[0])*sin(3*pi*x[1])', degree=10)]
    MPb = [
    dl.Constant('1.0'), 
    dl. Expression('sin(pi*x[0])*sin(pi*x[1])', degree=10),
    dl.Expression('x[1]', degree=10), dl.Expression('x[0]', degree=10), 
    dl.Expression('sin(3*pi*x[0])*sin(3*pi*x[1])', degree=10)]

    if ALL:
        Medium = []
        tmp = dl.Function(Vl*Vl)
        for ii in range(nbtest):
            tmp.vector().zero()
            dl.assign(tmp.sub(0), dl.interpolate(MPa[ii], Vl))
            dl.assign(tmp.sub(1), dl.interpolate(MPb[ii], Vl))
            Medium.append(tmp.vector().copy())
        if PRINT:    print 'check gradient with FD'
        checkgradfd_med(waveobj, Medium, PRINT, 1e-6, [1e-5, 1e-6, 1e-7], True)
        if PRINT:    print '\ncheck Hessian with FD'
        checkhessabfd_med(waveobj, Medium, PRINT, 1e-6, [1e-5, 1e-6, 1e-7], True, 'all')
    else:
        Mediuma, Mediumb = [], []
        tmp = dl.Function(Vl*Vl)
        for ii in range(nbtest):
            tmp.vector().zero()
            dl.assign(tmp.sub(0), dl.interpolate(MPa[ii], Vl))
            Mediuma.append(tmp.vector().copy())
            tmp.vector().zero()
            dl.assign(tmp.sub(1), dl.interpolate(MPb[ii], Vl))
            Mediumb.append(tmp.vector().copy())
        if PRINT:    print 'check a-gradient with FD'
        if 'a' in PARAM:
            checkgradfd_med(waveobj, Mediuma, PRINT, 1e-6, [1e-5, 1e-6, 1e-7], True)
        else:
            checkgradfd_med(waveobj, Mediuma[:1], PRINT, 1e-6, [1e-5], True)
        if PRINT:    print 'check b-gradient with FD'
        if 'b' in PARAM:
            checkgradfd_med(waveobj, Mediumb, PRINT, 1e-6, [1e-5, 1e-6, 1e-7], True)
        else:
            checkgradfd_med(waveobj, Mediumb[:1], PRINT, 1e-6, [1e-5], True)

        if PRINT:    
            print '\n'
            print 'check a-Hessian with FD'
        if 'a' in PARAM:
            checkhessabfd_med(waveobj, Mediuma, PRINT, 1e-6, [1e-5, 1e-6, 1e-7], True, 'a')
            #checkhessabfd_med(waveobj, Mediuma, PRINT, 1e-6,\
            #[1e-1, 1e-2, 1e-3, 1e-4], False, 'a')
        else:
            checkhessabfd_med(waveobj, Mediuma[:1], PRINT, 1e-6, [1e-5, 1e-6, 1e-7], True, 'a')
        if PRINT:    print 'check b-Hessian with FD'
        if 'b' in PARAM:
            checkhessabfd_med(waveobj, Mediumb, PRINT, 1e-6, [1e-5, 1e-6, 1e-7], True, 'b')
        else:
            checkhessabfd_med(waveobj, Mediumb[:1], PRINT, 1e-6, [1e-5, 1e-6, 1e-7], True, 'b')
    """
