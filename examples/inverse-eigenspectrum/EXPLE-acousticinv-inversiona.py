"""
Acoustic inverse problem for parameter a,
with parameter b known exaclty
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
dl.set_log_active(False)
from hippylib.randomizedEigensolver import doublePassG

from fenicstools.plotfenics import PlotFenics
from fenicstools.objectiveacoustic import ObjectiveAcoustic
from fenicstools.optimsolver import compute_searchdirection, bcktrcklinesearch
from fenicstools.prior import LaplacianPrior
from fenicstools.miscfenics import checkdt, setfct
from fenicstools.acousticwave import AcousticWave
from fenicstools.sourceterms import PointSources, RickerWavelet
from fenicstools.observationoperator import TimeObsPtwise

NNxy = [20, 40, 60, 80]

for Nxy in NNxy:
    mesh = dl.UnitSquareMesh(Nxy, Nxy)
    Vm = dl.FunctionSpace(mesh, 'Lagrange', 1)
    r = 2
    V = dl.FunctionSpace(mesh, 'Lagrange', r)
    Dt = 5.0e-4
    t0, t1, t2, tf = 0.0, 1.0, 5.0, 6.0

    # source:
    Ricker = RickerWavelet(0.5, 1e-10)
    Pt = PointSources(V, [[0.5,1.0]])
    mydelta = Pt[0]
    src = dl.Function(V)
    srcv = src.vector()
    def mysrc(tt):
        srcv.zero()
        srcv.axpy(Ricker(tt), mydelta)
        return srcv

    # target medium:
    b_target = dl.Expression(\
    '1.0 + 1.0*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
    b_target_fn = dl.interpolate(b_target, Vm)
    a_target = dl.Expression(\
    '1.0 + 0.4*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
    a_target_fn = dl.interpolate(a_target, Vm)

    checkdt(Dt, 1./Nxy, r, np.sqrt(2.0), False)

    # observation operator:
    obspts = [[0.0, ii/10.] for ii in range(1,10)] + \
    [[1.0, ii/10.] for ii in range(1,10)] + \
    [[ii/10., 0.0] for ii in range(1,10)] + \
    [[ii/10., 1.0] for ii in range(1,10)]
    obsop = TimeObsPtwise({'V':V, 'Points':obspts}, [t0, t1, t2, tf])

    # define pde operator:
    wavepde = AcousticWave({'V':V, 'Vm':Vm})
    wavepde.timestepper = 'backward'
    wavepde.lump = True
    wavepde.update({'a':a_target_fn, 'b':b_target_fn, \
    't0':t0, 'tf':tf, 'Dt':Dt, 'u0init':dl.Function(V), 'utinit':dl.Function(V)})
    wavepde.ftime = mysrc

    # parameters
    Vm = wavepde.Vm
    V = wavepde.V
    lenobspts = obsop.PtwiseObs.nbPts

    # set up plots:
    filename, ext = splitext(sys.argv[0])
    if isdir(filename + '/'):   rmtree(filename + '/')
    myplot = PlotFenics(filename+str(Nxy))
    myplot.set_varname('a_target')
    myplot.plot_vtk(a_target_fn)

    # define objective function:
    regul = LaplacianPrior({'Vm':Vm,'gamma':5e-4,'beta':5e-4, 'm0':a_target_fn})
    waveobj = ObjectiveAcoustic(wavepde, 'a', regul)
    waveobj.obsop = obsop

    # noisy data
    print 'generate noisy data'
    waveobj.solvefwd()
    skip = 20
    myplot.plot_timeseries(waveobj.solfwd, 'pd', 0, skip, dl.Function(V))
    dd = waveobj.Bp.copy()
    nbobspt, dimsol = dd.shape
    noiselevel = 0.1   # = 10%
    sigmas = np.sqrt((dd**2).sum(axis=1)/dimsol)*noiselevel
    np.random.seed(11)
    rndnoise = np.random.randn(nbobspt*dimsol).reshape((nbobspt, dimsol))
    waveobj.dd = dd + sigmas.reshape((len(sigmas),1))*rndnoise
    waveobj.solvefwd_cost()
    print 'noise misfit={}, regul cost={}, ratio={}'.format(waveobj.cost_misfit, \
    waveobj.cost_reg, waveobj.cost_misfit/waveobj.cost_reg)

    ######### Media for gradient and Hessian checks
    #Medium = np.zeros((5, Vm.dim()))
    #for ii in range(5):
    #    smoothperturb = dl.Expression('sin(n*pi*x[0])*sin(n*pi*x[1])', n=ii+1)
    #    smoothperturb_fn = dl.interpolate(smoothperturb, Vm)
    #    Medium[ii,:] = smoothperturb_fn.vector().array()

    print '\t{:12s} {:10s} {:12s} {:12s} {:12s} {:10s} \t{:10s} {:12s} {:12s}'.format(\
    'iter', 'cost', 'misfit', 'reg', '|G|', 'medmisf', 'a_ls', 'tol_cg', 'n_cg')
    dtruenorm = a_target_fn.vector().inner(waveobj.Mass*a_target_fn.vector())
    ######### Inverse problem
    #waveobj.update_PDE({'a':a_initial_fn})
    waveobj.solvefwd_cost()
    myplot.set_varname('a0')
    myplot.plot_vtk(waveobj.PDE.a)
    tolgrad = 1e-10
    tolcost = 1e-14
    check = False
    for iter in xrange(50):
        # gradient
        waveobj.solveadj_constructgrad()
        gradnorm = waveobj.MGv.inner(waveobj.Grad.vector())
        if iter == 0:   gradnorm0 = gradnorm
        diff = waveobj.PDE.a.vector() - a_target_fn.vector()
        medmisfit = diff.inner(waveobj.Mass*diff)
        if check and iter % 5 == 1:
            checkgradfd_med(waveobj, Medium, 1e-6, [1e-4, 1e-5, 1e-6])
            checkhessfd_med(waveobj, Medium, 1e-6, [1e-4, 1e-5, 1e-6])
        print '{:12d} {:12.4e} {:12.2e} {:12.2e} {:11.4e} {:10.2e} ({:4.2f})'.\
        format(iter, waveobj.cost, waveobj.cost_misfit, waveobj.cost_reg, \
        gradnorm, medmisfit, medmisfit/dtruenorm),
        # plots
        #myplot.plot_timeseries(waveobj.solfwd, 'p'+str(iter), 0, skip, dl.Function(V))
        myplot.set_varname('a'+str(iter))
        myplot.plot_vtk(waveobj.PDE.a)
        myplot.set_varname('grad'+str(iter))
        myplot.plot_vtk(waveobj.Grad)
    #    fig = plt.figure()
    #    fig.set_size_inches(20., 15.)
    #    for ii in range(lenobspts):
    #        ax = fig.add_subplot(6,6,ii+1)
    #        ax.plot(waveobj.PDE.times, waveobj.dd[ii,:], 'k--')
    #        ax.plot(waveobj.PDE.times, waveobj.Bp[ii,:], 'b')
    #        ax.set_title('obs'+str(ii))
    #    fig.savefig(filename + '/observations' + str(iter) + '.eps')
    #    plt.close(fig)
        # stopping criterion (gradient)
        if gradnorm < gradnorm0 * tolgrad or gradnorm < 1e-12:
            print '\nGradient sufficiently reduced -- optimization stopped'
            break
        # search direction
        tolcg = min(0.5, np.sqrt(gradnorm/gradnorm0))
        cgiter, cgres, cgid, tolcg = compute_searchdirection(waveobj, 'Newt', tolcg)
        myplot.set_varname('srchdir'+str(iter))
        myplot.plot_vtk(waveobj.srchdir)
        # line search
        cost_old = waveobj.cost
        statusLS, LScount, alpha = bcktrcklinesearch(waveobj, 12)
        cost = waveobj.cost
        print '{:11.3f} {:12.2e} {:10d}'.\
        format(alpha, tolcg, cgiter)
        # stopping criterion (cost)
        if np.abs(cost-cost_old)/np.abs(cost_old) < tolcost:
            print 'Cost function stagnates -- optimization stopped'
            break


    # Compute eigenspectrum
    waveobj.alpha_reg = 0.0 # no regularization
    Omega = np.random.randn(Vm.dim()*60).reshape((Vm.dim(),60))
    #TODO: need to left-multiply Omega by R^{-1/2}.M (to be checked)
    d, U = doublePassG(waveobj, regul.precond, regul.get_precond(), Omega, 40)
    np.savetext(filename+str(Nxy)+'/eigenvalues.txt', d)
    myplot.set_varname('eigenvectors')
    for ii in range(U.shape[1]):
        setfct(waveobj.PDE.a, U[:,ii])
        myplot.plot_vtk(waveobj.PDE.a, ii)
