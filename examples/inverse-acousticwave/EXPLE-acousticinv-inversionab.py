"""
Acoustic inverse problem for parameter a and b
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

from fenicstools.plotfenics import PlotFenics
from fenicstools.objectiveacoustic import ObjectiveAcoustic
from fenicstools.optimsolver import compute_searchdirection, bcktrcklinesearch
from fenicstools.jointregularization import Tikhonovab
from parametersinversion import parametersinversion

from dolfin import MPI, mpi_comm_world
mycomm = mpi_comm_world()
mpirank = MPI.rank(mycomm)


if __name__ == "__main__":
    # parameters
    a_target_fn, a_initial_fn, b_target_fn, b_initial_fn, \
    wavepde, mysrc, obsop = parametersinversion()
    Vm = wavepde.Vm
    V = wavepde.V
    lenobspts = obsop.PtwiseObs.nbPts

    # set up plots:
    filename, ext = splitext(sys.argv[0])
    if mpirank == 0 and isdir(filename + '/'):   rmtree(filename + '/')
    myplot = PlotFenics(filename)
    myplot.set_varname('b_target')
    myplot.plot_vtk(b_target_fn)
    myplot.set_varname('a_target')
    myplot.plot_vtk(a_target_fn)

    # define objective function:
    #regul = Tikhonovab({'Vm':Vm,'gamma':5e-4,'beta':1e-8, 'm0':[1.0,1.0], 'cg':1.0})
    regul = Tikhonovab({'Vm':Vm,'gamma':5e-4,'beta':1e-8, 'm0':[1.0,1.0]})
    waveobj = ObjectiveAcoustic(wavepde, mysrc, 'ab', regul)
    waveobj.obsop = obsop

    # noisy data
    if mpirank == 0: print 'generate noisy data'
    waveobj.solvefwd()
    skip = 20
    myplot.plot_timeseries(waveobj.solfwd, 'pd', 0, skip, dl.Function(V))
    DD = waveobj.Bp[:]
    noiselevel = 0.1   # = 10%
    np.random.seed(11)
    for ii, dd in enumerate(DD):
        nbobspt, dimsol = dd.shape
        sigmas = np.sqrt((dd**2).sum(axis=1)/dimsol)*noiselevel
        rndnoise = np.random.randn(nbobspt*dimsol).reshape((nbobspt, dimsol))
        DD[ii] = dd + sigmas.reshape((len(sigmas),1))*rndnoise
    waveobj.dd = DD
    waveobj.solvefwd_cost()
    if mpirank == 0:
        print 'noise misfit={}, regul cost={}, ratio={}'.format(\
        waveobj.cost_misfit, waveobj.cost_reg, waveobj.cost_misfit/waveobj.cost_reg)

    ######### Media for gradient and Hessian checks
    #Medium = np.zeros((5, Vm.dim()))
    #for ii in range(5):
    #    smoothperturb = dl.Expression('sin(n*pi*x[0])*sin(n*pi*x[1])', n=ii+1)
    #    smoothperturb_fn = dl.interpolate(smoothperturb, Vm)
    #    Medium[ii,:] = smoothperturb_fn.vector().array()

    ab_target_fn = dl.Function(Vm*Vm)
    dl.assign(ab_target_fn.sub(0), a_target_fn)
    dl.assign(ab_target_fn.sub(1), b_target_fn)
    ab_init_fn = dl.Function(Vm*Vm)
    dl.assign(ab_init_fn.sub(0), a_initial_fn)
    dl.assign(ab_init_fn.sub(1), b_initial_fn)
    waveobj.inversion(ab_init_fn, ab_target_fn, mycomm)
    """
    if mpirank == 0:
        print '\t{:12s} {:10s} {:12s} {:12s} {:12s} {:10s} \t{:10s} {:12s} {:12s}'.format(\
        'iter', 'cost', 'misfit', 'reg', '|G|', 'medmisf', 'a_ls', 'tol_cg', 'n_cg')
    ab_target_fn = dl.Function(Vm*Vm)
    dl.assign(ab_target_fn.sub(0), a_target_fn)
    dl.assign(ab_target_fn.sub(1), b_target_fn)
    dtruenorm = ab_target_fn.vector().inner(waveobj.Mass*ab_target_fn.vector())
    waveobj.update_PDE({'a':a_initial_fn, 'b':b_initial_fn})
    runjob(waveobj, myplot, ab_target_fn, dtruenorm, mpirank)
    """


"""
######### Inverse problem
#@profile
def runjob(waveobj, myplot, ab_target_fn, dtruenorm, mpirank):
    waveobj.solvefwd_cost()
    myplot.set_varname('ainit')
    myplot.plot_vtk(waveobj.PDE.a)
    myplot.set_varname('binit')
    myplot.plot_vtk(waveobj.PDE.b)
    tolgrad = 1e-10
    tolcost = 1e-14
#    check = False
    for it in xrange(50):
        # gradient
        waveobj.solveadj_constructgrad()    # expensive step (~20%)
        gradnorm = waveobj.MGv.inner(waveobj.Grad.vector())
        if it == 0:   gradnorm0 = gradnorm
        dl.assign(waveobj.ab.sub(0), waveobj.PDE.a)
        dl.assign(waveobj.ab.sub(1), waveobj.PDE.b)
        diff = waveobj.ab.vector() - ab_target_fn.vector()
        medmisfit = diff.inner(waveobj.Mass*diff)
#        if check and it % 5 == 1:
#            checkgradfd_med(waveobj, Medium, 1e-6, [1e-4, 1e-5, 1e-6])
#            checkhessfd_med(waveobj, Medium, 1e-6, [1e-4, 1e-5, 1e-6])
        if mpirank == 0:
            print '{:12d} {:12.4e} {:12.2e} {:12.2e} {:11.4e} {:10.2e} ({:4.2f})'.\
            format(it, waveobj.cost, waveobj.cost_misfit, waveobj.cost_reg, \
            gradnorm, medmisfit, medmisfit/dtruenorm),
        # plots
        #myplot.plot_timeseries(waveobj.solfwd, 'p'+str(it), 0, skip, dl.Function(V))
        myplot.set_varname('a'+str(it))
        myplot.plot_vtk(waveobj.PDE.a)
        myplot.set_varname('b'+str(it))
        myplot.plot_vtk(waveobj.PDE.b)
        Ga, Gb = waveobj.Grad.split(deepcopy=True)
        myplot.set_varname('grada'+str(it))
        myplot.plot_vtk(Ga)
        myplot.set_varname('gradb'+str(it))
        myplot.plot_vtk(Gb)
    #    fig = plt.figure()
    #    fig.set_size_inches(20., 15.)
    #    for ii in range(lenobspts):
    #        ax = fig.add_subplot(6,6,ii+1)
    #        ax.plot(waveobj.PDE.times, waveobj.dd[ii,:], 'k--')
    #        ax.plot(waveobj.PDE.times, waveobj.Bp[ii,:], 'b')
    #        ax.set_title('obs'+str(ii))
    #    fig.savefig(filename + '/observations' + str(it) + '.eps')
    #    plt.close(fig)
        # stopping criterion (gradient)
        if gradnorm < gradnorm0 * tolgrad:
            if mpirank == 0:
                print '\nGradient sufficiently reduced -- optimization stopped'
            break
        # search direction
        tolcg = min(0.5, np.sqrt(gradnorm/gradnorm0))
        cgiter, cgres, cgid, tolcg = compute_searchdirection(waveobj, 'Newt', tolcg)    # most cpu intensive step (~75%)
        myplot.set_varname('srchdir'+str(it))
        myplot.plot_vtk(waveobj.srchdir)
        # line search
        cost_old = waveobj.cost
        statusLS, LScount, alpha = bcktrcklinesearch(waveobj, 12)
        cost = waveobj.cost
        if mpirank == 0: print '{:11.3f} {:12.2e} {:10d}'.format(alpha, tolcg, cgiter)
        # stopping criterion (cost)
        if np.abs(cost-cost_old)/np.abs(cost_old) < tolcost:
            if mpirank == 0: print 'Cost function stagnates -- optimization stopped'
            break
"""

