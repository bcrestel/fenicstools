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
from dolfin import MPI, mpi_comm_world
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

mpicomm = mpi_comm_world()
mpirank = MPI.rank(mpicomm)

NNxy = [20]

for Nxy in NNxy:
    if mpirank == 0:    print 'Meshing'
    mesh = dl.UnitSquareMesh(Nxy, Nxy)
    Vm = dl.FunctionSpace(mesh, 'Lagrange', 1)
    r = 2
    V = dl.FunctionSpace(mesh, 'Lagrange', r)
    Dt = 5.0e-4
    t0, t1, t2, tf = 0.0, 1.0, 5.0, 6.0

    # source:
    if mpirank == 0:    print 'Define sources'
    Ricker = RickerWavelet(0.5, 1e-10)
    srcloc = [[0.5,1.0]]
    Pt = PointSources(V, srcloc)
    src = dl.Function(V)
    srcv = src.vector()
    mysrc = [Ricker, Pt, srcv]

    # target medium:
    if mpirank == 0:    print 'Define media'
    b_target = dl.Expression(\
    '1.0 + 1.0*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
    b_target_fn = dl.interpolate(b_target, Vm)
    a_target = dl.Expression('1.0')
    a_target_fn = dl.interpolate(a_target, Vm)

    checkdt(Dt, 1./Nxy, r, np.sqrt(2.0), True)

    # observation operator:
    if mpirank == 0:    print 'Define observation locations'
    obspts = [[0.0, ii/10.] for ii in range(1,10)] + \
    [[1.0, ii/10.] for ii in range(1,10)] + \
    [[ii/10., 0.0] for ii in range(1,10)] + \
    [[ii/10., 1.0] for ii in range(1,10)]
    obsop = TimeObsPtwise({'V':V, 'Points':obspts}, [t0, t1, t2, tf])

    # define pde operator:
    if mpirank == 0:    print 'Define wave PDE'
    wavepde = AcousticWave({'V':V, 'Vm':Vm})
    wavepde.timestepper = 'backward'
    wavepde.lump = True
    wavepde.update({'a':a_target_fn, 'b':b_target_fn, \
    't0':t0, 'tf':tf, 'Dt':Dt, 'u0init':dl.Function(V), 'utinit':dl.Function(V)})

    # parameters
    Vm = wavepde.Vm
    V = wavepde.V
    lenobspts = obsop.PtwiseObs.nbPts

    # define objective function:
    if mpirank == 0:    print 'Define objective function'
    regul = LaplacianPrior({'Vm':Vm,'gamma':5e-4,'beta':5e-4, 'm0':a_target_fn})
    waveobj = ObjectiveAcoustic(wavepde, mysrc, 'b', regul)
    waveobj.obsop = obsop

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
    waveobj.solvefwd_cost()
    if mpirank == 0:
        print 'noise misfit={}, regul cost={}, ratio={}'.format(waveobj.cost_misfit, \
        waveobj.cost_reg, waveobj.cost_misfit/waveobj.cost_reg)

    if mpirank == 0:
        print '\t{:12s} {:10s} {:12s} {:12s} {:12s} {:10s} \t{:10s} {:12s} {:12s}'.format(\
        'iter', 'cost', 'misfit', 'reg', '|G|', 'medmisf', 'a_ls', 'tol_cg', 'n_cg')
    dtruenorm = a_target_fn.vector().inner(waveobj.Mass*a_target_fn.vector())
    ######### Inverse problem
    waveobj.solvefwd_cost()
    tolgrad = 1e-10
    tolcost = 1e-14
    for iter in xrange(50):
        # gradient
        waveobj.solveadj_constructgrad()
        gradnorm = waveobj.MGv.inner(waveobj.Grad.vector())
        if iter == 0:   gradnorm0 = gradnorm
        diff = waveobj.PDE.a.vector() - a_target_fn.vector()
        medmisfit = diff.inner(waveobj.Mass*diff)
        if mpirank == 0:
            print '{:12d} {:12.4e} {:12.2e} {:12.2e} {:11.4e} {:10.2e} ({:4.2f})'.\
            format(iter, waveobj.cost, waveobj.cost_misfit, waveobj.cost_reg, \
            gradnorm, medmisfit, medmisfit/dtruenorm),
        # stopping criterion (gradient)
        if gradnorm < gradnorm0 * tolgrad or gradnorm < 1e-12:
            if mpirank == 0:
                print '\nGradient sufficiently reduced -- optimization stopped'
            break
        # search direction
        tolcg = min(0.5, np.sqrt(gradnorm/gradnorm0))
        cgiter, cgres, cgid, tolcg = compute_searchdirection(waveobj, 'Newt', tolcg)
        # line search
        cost_old = waveobj.cost
        statusLS, LScount, alpha = bcktrcklinesearch(waveobj, 12)
        cost = waveobj.cost
        if mpirank == 0:
            print '{:11.3f} {:12.2e} {:10d}'.format(alpha, tolcg, cgiter)
        # stopping criterion (cost)
        if np.abs(cost-cost_old)/np.abs(cost_old) < tolcost:
            print 'Cost function stagnates -- optimization stopped'
            break
