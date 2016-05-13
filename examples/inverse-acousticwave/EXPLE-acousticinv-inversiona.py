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

from fenicstools.plotfenics import PlotFenics
from fenicstools.objectiveacoustic import ObjectiveAcoustic
from fenicstools.optimsolver import compute_searchdirection, bcktrcklinesearch
from fenicstools.prior import LaplacianPrior
from parametersinversion import parametersinversion

# parameters
a_target_fn, a_initial_fn, b_target_fn, b_initial_fn, wavepde, obsop = parametersinversion()
Vm = wavepde.Vm
V = wavepde.V
lenobspts = obsop.PtwiseObs.nbPts

# set up plots:
filename, ext = splitext(sys.argv[0])
if isdir(filename + '/'):   rmtree(filename + '/')
myplot = PlotFenics(filename)
myplot.set_varname('b_target')
myplot.plot_vtk(b_target_fn)
myplot.set_varname('a_target')
myplot.plot_vtk(a_target_fn)

# define objective function:
regul = LaplacianPrior({'Vm':Vm,'gamma':5e-4,'beta':1e-8, 'm0':1.0})
waveobj = ObjectiveAcoustic(wavepde, 'a', regul)
waveobj.obsop = obsop
sys.exit(1)

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
Medium = np.zeros((5, Vm.dim()))
for ii in range(5):
    smoothperturb = dl.Expression('sin(n*pi*x[0])*sin(n*pi*x[1])', n=ii+1)
    smoothperturb_fn = dl.interpolate(smoothperturb, Vm)
    Medium[ii,:] = smoothperturb_fn.vector().array()

print '\t{:12s} {:10s} {:12s} {:12s} {:12s} {:10s} \t{:10s} {:12s} {:12s}'.format(\
'iter', 'cost', 'misfit', 'reg', '|G|', 'medmisf', 'a_ls', 'tol_cg', 'n_cg')
dtruenorm = a_target_fn.vector().inner(waveobj.Mass*a_target_fn.vector())
######### Inverse problem
waveobj.update_PDE({'a':a_initial_fn})
waveobj.solvefwd_cost()
myplot.set_varname('a0')
myplot.plot_vtk(waveobj.PDE.a)
myplot.set_varname('b0')
myplot.plot_vtk(waveobj.PDE.b)
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
    myplot.set_varname('b'+str(iter))
    myplot.plot_vtk(waveobj.PDE.b)
    myplot.set_varname('grad'+str(iter))
    myplot.plot_vtk(waveobj.Grad)
    fig = plt.figure()
    fig.set_size_inches(20., 15.)
    for ii in range(lenobspts):
        ax = fig.add_subplot(6,6,ii+1)
        ax.plot(waveobj.PDE.times, waveobj.dd[ii,:], 'k--')
        ax.plot(waveobj.PDE.times, waveobj.Bp[ii,:], 'b')
        ax.set_title('obs'+str(ii))
    fig.savefig(filename + '/observations' + str(iter) + '.eps')
    plt.close(fig)
    # stopping criterion (gradient)
    if gradnorm < gradnorm0 * tolgrad:
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
