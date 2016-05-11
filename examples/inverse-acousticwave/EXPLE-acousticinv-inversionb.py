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
from fenicstools.acousticwave import AcousticWave
from fenicstools.objectiveacoustic import ObjectiveAcoustic
from fenicstools.sourceterms import PointSources, RickerWavelet
from fenicstools.observationoperator import TimeObsPtwise
from fenicstools.miscfenics import checkdt, setfct
from fenicstools.optimsolver import compute_searchdirection, bcktrcklinesearch
from fenicstools.prior import LaplacianPrior

######### Noisy data
Nxy = 20
mesh = dl.UnitSquareMesh(Nxy, Nxy)
Vm = dl.FunctionSpace(mesh, 'Lagrange', 1)
r = 2
V = dl.FunctionSpace(mesh, 'Lagrange', r)
Dt = 2.5e-3
checkdt(Dt, 1./Nxy, r, np.sqrt(2.0), False)
t0, t1, t2, tf = 0.0, 0.5, 4.5, 5.0
# set up plots:
filename, ext = splitext(sys.argv[0])
if isdir(filename + '/'):   rmtree(filename + '/')
myplot = PlotFenics(filename)
# source:
Ricker = RickerWavelet(0.5, 1e-10)
Pt = PointSources(V, [[0.5,1.0]])
mydelta = Pt[0].array()
def mysrc(tt):
    return Ricker(tt)*mydelta
# target medium:
b_target = dl.Expression(\
'1.0 + 2.0*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
b_target_fn = dl.interpolate(b_target, Vm)
myplot.set_varname('b_target')
myplot.plot_vtk(b_target_fn)
a_target = dl.Expression(\
'1.0 + 0.5*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
a_target_fn = dl.interpolate(a_target, Vm)
myplot.set_varname('a_target')
myplot.plot_vtk(a_target_fn)
# observation operator:
obspts = [[0.2, ii/10.] for ii in range(2,9)] + \
[[0.8, ii/10.] for ii in range(2,9)] + \
[[ii/10., 0.2] for ii in range(3,8)] + \
[[ii/10., 0.8] for ii in range(3,8)]
obsop = TimeObsPtwise({'V':V, 'Points':obspts}, [t0, t1, t2, tf])
# define pde operator:
wavepde = AcousticWave({'V':V, 'Vm':Vm})
wavepde.timestepper = 'backward'
#wavepde.lump = True    # not checked
#wavepde.set_abc(mesh, LeftRight(), True)   # not implemented
wavepde.update({'a':a_target_fn, 'b':b_target_fn, \
't0':t0, 'tf':tf, 'Dt':Dt, 'u0init':dl.Function(V), 'utinit':dl.Function(V)})
wavepde.ftime = mysrc
# define objective function:
regul = LaplacianPrior({'Vm':Vm,'gamma':1e-3,'beta':1e-14})
waveobj = ObjectiveAcoustic(wavepde, 'b', regul)
waveobj.obsop = obsop
# noisy data
print 'generate noisy data'
waveobj.solvefwd()
myplot.plot_timeseries(waveobj.solfwd, 'pd', 0, 20, dl.Function(V))
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

######### Inverse problem
waveobj.update_PDE({'b':1.0})
waveobj.solvefwd_cost()
print '\t{:12s} {:10s} {:12s} {:12s} {:12s} {:10s} \t{:10s} {:12s} {:12s}'.format(\
'iter', 'cost', 'misfit', 'reg', '|G|', 'medmisf', 'a_ls', 'tol_cg', 'n_cg')
dtruenorm = b_target_fn.vector().inner(waveobj.Mass*b_target_fn.vector())
diff = waveobj.PDE.b.vector() - b_target_fn.vector()
medmisfit = diff.inner(waveobj.Mass*diff)
cost = waveobj.cost
print '{:12d} {:12.4e} {:12.2e} {:12.2e} {:11s} {:10.2e} ({:4.2f})'.\
format(0, cost, waveobj.cost_misfit, waveobj.cost_reg, ' ', \
medmisfit, medmisfit/dtruenorm)
myplot.set_varname('medium0')
myplot.plot_vtk(waveobj.PDE.b)
tolgrad = 1e-10
tolcost = 1e-14
check = False
for iter in xrange(1, 9):
    waveobj.solveadj_constructgrad()
    if check and iter % 5 == 1:
        checkgradfd_med(waveobj, Medium, 1e-6, [1e-4, 1e-5, 1e-6])
        checkhessfd_med(waveobj, Medium, 1e-6, [1e-4, 1e-5, 1e-6])
    gradnorm = waveobj.MGv.inner(waveobj.Grad.vector())
    if iter == 1:   gradnorm0 = gradnorm
    tolcg = min(0.5, np.sqrt(gradnorm/gradnorm0))
    cgiter, cgres, cgid, tolcg = compute_searchdirection(waveobj, 'Newt', tolcg)
    statusLS, LScount, alpha = bcktrcklinesearch(waveobj, 12)
    diff = waveobj.PDE.b.vector() - b_target_fn.vector()
    medmisfit = diff.inner(waveobj.Mass*diff)
    cost_old = cost
    cost = waveobj.cost
    print '{:12d} {:12.4e} {:12.2e} {:12.2e} {:11.4e} {:10.2e} ({:4.2f}) {:11.3f} {:12.2e} {:10d}'.\
    format(iter, cost, waveobj.cost_misfit, waveobj.cost_reg, \
    gradnorm, medmisfit, medmisfit/dtruenorm, alpha, tolcg, cgiter)
    # plots
    myplot.plot_timeseries(waveobj.solfwd, 'p'+str(iter), 0, 20, dl.Function(V))
    myplot.set_varname('a'+str(iter))
    myplot.plot_vtk(waveobj.PDE.a)
    myplot.set_varname('b'+str(iter))
    myplot.plot_vtk(waveobj.PDE.b)
    myplot.set_varname('grad'+str(iter))
    myplot.plot_vtk(waveobj.Grad)
    myplot.set_varname('srchdir'+str(iter))
    myplot.plot_vtk(waveobj.srchdir)
    fig = plt.figure()
    fig.set_size_inches(20., 15.)
    for ii in range(len(obspts)):
        ax = fig.add_subplot(4,6,ii+1)
        ax.plot(waveobj.PDE.times, waveobj.dd[ii,:], 'k--')
        ax.plot(waveobj.PDE.times, waveobj.Bp[ii,:], 'b')
        ax.set_title('obs'+str(ii))
    fig.savefig(filename + '/observations' + str(iter) + '.eps')
    plt.close(fig)
    # stopping criteria
    if gradnorm < gradnorm0 * tolgrad:
        print 'Gradient sufficiently reduced -- optimization stopped'
        break
    if np.abs(cost-cost_old)/np.abs(cost_old) < tolcost:
        print 'Cost function stagnates -- optimization stopped'
        break
