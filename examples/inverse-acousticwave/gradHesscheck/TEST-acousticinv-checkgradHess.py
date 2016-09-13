"""
Acoustic wave inverse problem with a single low-frequency,
and homogeneous Neumann boundary conditions everywhere.
Check gradient and Hessian for joint inverse problem a and b
"""

import sys
from os.path import splitext, isdir
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt

import dolfin as dl
import ffc
from fenicstools.plotfenics import PlotFenics
from fenicstools.acousticwave import AcousticWave
from fenicstools.sourceterms import PointSources, RickerWavelet
from fenicstools.observationoperator import TimeObsPtwise
from fenicstools.miscfenics import checkdt, setfct
from fenicstools.objectiveacoustic import ObjectiveAcoustic
from fenicstools.optimsolver import checkgradfd_med, checkhessabfd_med, checkhessfd_med

from dolfin import MPI, mpi_comm_world
mpicomm = mpi_comm_world()
mpirank = MPI.rank(mpicomm)
PLOT = True
skip = 50

# ABC:
#class LeftRight(dl.SubDomain):
#    def inside(self, x, on_boundary):
#        return (x[0] < 1e-16 or x[0] > 1.0 - 1e-16) \
#        and on_boundary

#@profile
def run_testab(wavepde, mysrc, obsop, init_med, myplot=None):
    V, Vl = wavepde.V, wavepde.Vm
    fctV = dl.Function(V)
    # define objective function:
    waveobj = ObjectiveAcoustic(wavepde, mysrc, 'ab')
    waveobj.obsop = obsop
    # data
    if mpirank == 0:    print 'generate noisy data'
    waveobj.solvefwd()
    if PLOT:    myplot.plot_timeseries(waveobj.solfwd[0], 'pd', 0, skip, fctV)
    DD = waveobj.Bp[:]
    noiselevel = 0.1   # = 10%
    for ii, dd in enumerate(DD):
        np.random.seed(11)
        nbobspt, dimsol = dd.shape
        sigmas = np.sqrt((dd**2).sum(axis=1)/dimsol)*noiselevel
        rndnoise = np.random.randn(nbobspt*dimsol).reshape((nbobspt, dimsol))
        DD[ii] = dd + sigmas.reshape((len(sigmas),1))*rndnoise
    waveobj.dd = DD
    # gradient
    if mpirank == 0:    print 'generate observations'
    rho_init_fn, lambda_init_fn = init_med
    waveobj.update_PDE({'a':rho_init_fn, 'b':lambda_init_fn})
    waveobj.solvefwd_cost()
    if mpirank == 0:    print 'misfit = {}'.format(waveobj.cost_misfit)
    if PLOT:    myplot.plot_timeseries(waveobj.solfwd[0], 'p', 0, skip, fctV)
#    # Plot data and observations
#    fig = plt.figure()
#    fig.set_size_inches(20., 15.)
#    for ii in range(len(obspts)):
#        ax = fig.add_subplot(6,6,ii+1)
#        ax.plot(waveobj.PDE.times, waveobj.dd[ii,:], 'k--')
#        ax.plot(waveobj.PDE.times, waveobj.Bp[ii,:], 'b')
#        ax.set_title('Plot'+str(ii))
#    fig.savefig(filename + '/observations.eps')
    if mpirank == 0:    print 'compute gradient'
    waveobj.solveadj_constructgrad()
    Grada,Gradb = waveobj.Grad.split(deepcopy=True)
    if PLOT:    
        myplot.plot_timeseries(waveobj.soladj[0], 'v', 0, skip, fctV)
        myplot.set_varname('grada')
        myplot.plot_vtk(Grada)
        myplot.set_varname('gradb')
        myplot.plot_vtk(Gradb)
    # Prepare random medium directions to test gradient and Hessian for param a
    Medium = np.zeros((5, wavepde.a.vector().local_size() + wavepde.b.vector().local_size()))
    tmp = dl.Function(Vl*Vl)
    for ii in range(5):
        smoothperturb = dl.Expression('sin(n*pi*x[0])*sin(n*pi*x[1])', n=ii+1)
        smoothperturb_fn = dl.interpolate(smoothperturb, Vl)
        dl.assign(tmp.sub(0), smoothperturb_fn)
        Medium[ii,:] = tmp.vector().array()
    if mpirank == 0:    print 'check a-gradient with FD'
    checkgradfd_med(waveobj, Medium, 1e-6, [1e-4, 1e-5, 1e-6], True, mpicomm)
    if mpirank == 0:    print 'check a-Hessian with FD'
    checkhessabfd_med(waveobj, Medium, 1e-6, [1e-3, 1e-4, 1e-5, 1e-6], False, 'a')
    # Prepare random medium directions to test gradient and Hessian for param b
    Medium = np.zeros((5, wavepde.a.vector().local_size() + wavepde.b.vector().local_size()))
    tmp = dl.Function(Vl*Vl)
    for ii in range(5):
        smoothperturb = dl.Expression('sin(n*pi*x[0])*sin(n*pi*x[1])', n=ii+1)
        smoothperturb_fn = dl.interpolate(smoothperturb, Vl)
        dl.assign(tmp.sub(1), smoothperturb_fn)
        Medium[ii,:] = tmp.vector().array()
    if mpirank == 0:    print 'check b-gradient with FD'
    checkgradfd_med(waveobj, Medium, 1e-6, [1e-4, 1e-5, 1e-6], True, mpicomm)
    if mpirank == 0:    print 'check b-Hessian with FD'
    checkhessabfd_med(waveobj, Medium, 1e-6, [1e-3, 1e-4, 1e-5, 1e-6], False, 'b', mpicomm)
    """
    #print 'check gradient with FD'
    Medium = np.zeros((5, 2*Vl.dim()))
    tmp = dl.Function(Vl*Vl)
    for ii in range(Medium.shape[0]):
        smoothperturb = dl.Expression('sin(n*pi*x[0])*sin(n*pi*x[1])', n=ii+1)
        smoothperturb_fn = dl.interpolate(smoothperturb, Vl)
        dl.assign(tmp.sub(0), smoothperturb_fn)
        dl.assign(tmp.sub(1), smoothperturb_fn)
        Medium[ii,:] = tmp.vector().array()
    checkgradfd_med(waveobj, Medium, 1e-6, [1e-4, 1e-5, 1e-6])
    print 'check Hessian with FD'
    checkhessabfd_med(waveobj, Medium, 1e-6, [1e-3, 1e-4, 1e-5, 1e-6], False, 'all')
    """

def run_testa(fpeak, lambdaa, rho, Nxy, tfilterpts, r, Dt, skip):
    run_test(fpeak, lambdaa, rho, Nxy, tfilterpts, r, Dt, skip, 'a')

def run_testb(fpeak, lambdaa, rho, Nxy, tfilterpts, r, Dt, skip):
    run_test(fpeak, lambdaa, rho, Nxy, tfilterpts, r, Dt, skip, 'b')

def run_test(fpeak, lambdaa, rho, Nxy, tfilterpts, r, Dt, skip, param):
    """ param = a or b """
    lambdamin, lambdamax = lambdaa
    rhomin, rhomax = rho
    h = 1./Nxy
    checkdt(Dt, h, r, np.sqrt(lambdamax), True)
    mesh = dl.UnitSquareMesh(Nxy, Nxy)
    Vl = dl.FunctionSpace(mesh, 'Lagrange', 1)
    V = dl.FunctionSpace(mesh, 'Lagrange', r)
    fctV = dl.Function(V)
    # set up plots:
    filename, ext = splitext(sys.argv[0])
    if isdir(filename + '/'):   rmtree(filename + '/')
    myplot = PlotFenics(filename)
    # source:
    Ricker = RickerWavelet(fpeak, 1e-10)
    Pt = PointSources(V, [[0.5,1.0], [0.5,0.0]])
    src = dl.Function(V)
    srcv = src.vector()
    mysrc = [Ricker, Pt, srcv]
#%    mydelta = Pt[0].array()
#%    def mysrc(tt):
#%        return Ricker(tt)*mydelta
    # target medium:
    lambda_target = dl.Expression(\
    'lmin + (lmax-lmin)*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)', \
    lmin=lambdamin, lmax=lambdamax)
    lambda_target_fn = dl.interpolate(lambda_target, Vl)
    myplot.set_varname('lambda_target')
    myplot.plot_vtk(lambda_target_fn)
    rho_target = dl.Expression(\
    'lmin + (lmax-lmin)*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)', \
    lmin=rhomin, lmax=rhomax)
    rho_target_fn = dl.interpolate(rho_target, Vl)
    myplot.set_varname('rho_target')
    myplot.plot_vtk(rho_target_fn)
    # initial medium:
    init_expr = dl.Constant(lambdamin)
    init_fn = dl.interpolate(init_expr, Vl)
    myplot.set_varname('lambdarho_init')
    myplot.plot_vtk(init_fn)
    # observation operator:
    obspts = [[0.0, ii/10.] for ii in range(1,10)] + \
    [[1.0, ii/10.] for ii in range(1,10)] + \
    [[ii/10., 0.0] for ii in range(1,10)] + \
    [[ii/10., 1.0] for ii in range(1,10)]
    obsop = TimeObsPtwise({'V':V, 'Points':obspts}, tfilterpts)
    # define pde operator:
    wavepde = AcousticWave({'V':V, 'Vm':Vl})
    wavepde.timestepper = 'backward'
    wavepde.lump = True
    #wavepde.set_abc(mesh, LeftRight(), True)   # not implemented
    wavepde.update({'b':lambda_target_fn, 'a':rho_target_fn, \
    't0':tfilterpts[0], 'tf':tfilterpts[-1], 'Dt':Dt, 'u0init':dl.Function(V), 'utinit':dl.Function(V)})
    # define objective function:
    waveobj = ObjectiveAcoustic(wavepde, mysrc, param)
    waveobj.obsop = obsop
    # data
    print 'generate noisy data'
    waveobj.solvefwd()
    myplot.plot_timeseries(waveobj.solfwd, 'pd', 0, skip, fctV)
    DD = waveobj.Bp[:]
    np.random.seed(11)
    noiselevel = 0.1   # = 10%
    for ii, dd in enumerate(DD):
        nbobspt, dimsol = dd.shape
        sigmas = np.sqrt((dd**2).sum(axis=1)/dimsol)*noiselevel
        rndnoise = np.random.randn(nbobspt*dimsol).reshape((nbobspt, dimsol))
        DD[ii] = dd + sigmas.reshape((len(sigmas),1))*rndnoise
    waveobj.dd = DD
    # gradient
    print 'generate observations'
    waveobj.update_PDE({param:init_fn})
    waveobj.solvefwd_cost()
    cost1 = waveobj.cost_misfit
    print 'misfit = {}'.format(waveobj.cost_misfit)
    myplot.plot_timeseries(waveobj.solfwd, 'p', 0, skip, fctV)
#    # Plot data and observations
#    fig = plt.figure()
#    fig.set_size_inches(20., 15.)
#    for ii in range(len(obspts)):
#        ax = fig.add_subplot(6,6,ii+1)
#        ax.plot(waveobj.PDE.times, waveobj.dd[ii,:], 'k--')
#        ax.plot(waveobj.PDE.times, waveobj.Bp[ii,:], 'b')
#        ax.set_title('Plot'+str(ii))
#    fig.savefig(filename + '/observations.eps')
    print 'compute gradient'
    waveobj.solveadj_constructgrad()
    myplot.plot_timeseries(waveobj.soladj, 'v', 0, skip, fctV)
    myplot.set_varname('grad')
    myplot.plot_vtk(waveobj.Grad)
    print 'check gradient with FD'
    Medium = np.zeros((5, Vl.dim()))
    for ii in range(Medium.shape[0]):
        smoothperturb = dl.Expression('sin(n*pi*x[0])*sin(n*pi*x[1])', n=ii+1)
        smoothperturb_fn = dl.interpolate(smoothperturb, Vl)
        Medium[ii,:] = smoothperturb_fn.vector().array()
    checkgradfd_med(waveobj, Medium, 1e-6, [1e-4, 1e-5, 1e-6])
    print 'check Hessian with FD'
    checkhessfd_med(waveobj, Medium, 1e-6, [1e-3, 1e-4, 1e-5, 1e-6], False)


if __name__ == "__main__":
    # Mesh
    Nxy = 25
    mesh = dl.UnitSquareMesh(Nxy, Nxy)
    Vl = dl.FunctionSpace(mesh, 'Lagrange', 1)
    V = dl.FunctionSpace(mesh, 'Lagrange', 2)
    # target medium:
    lambdamin, lambdamax = 1.0, 3.0
    rhomin, rhomax = 1.0, 1.5
    lambda_target = dl.Expression(\
    'lmin + (lmax-lmin)*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)', \
    lmin=lambdamin, lmax=lambdamax)
    lambda_target_fn = dl.interpolate(lambda_target, Vl)
    rho_target = dl.Expression(\
    'lmin + (lmax-lmin)*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)', \
    lmin=rhomin, lmax=rhomax)
    rho_target_fn = dl.interpolate(rho_target, Vl)
    # initial medium:
    lambda_init = dl.Constant(lambdamin)
    lambda_init_fn = dl.interpolate(lambda_init, Vl)
    rho_init_fn = dl.interpolate(lambda_init, Vl)
    # Source:
    fpeak = 0.5  #Hz
    Ricker = RickerWavelet(fpeak, 1e-10)
    Pt = PointSources(V, [[0.5,1.0], [0.5,0.0], [0.5,0.5]])
    src = dl.Function(V)
    srcv = src.vector()
    mysrc = [Ricker, Pt, srcv]
    # observation operator:
    obspts = [[0.0, ii/10.] for ii in range(1,10)] + \
    [[1.0, ii/10.] for ii in range(1,10)] + \
    [[ii/10., 0.0] for ii in range(1,10)] + \
    [[ii/10., 1.0] for ii in range(1,10)]
    Dt = 2.0e-3
    t0, t1, t2, tf = 0.0, 0.5, 4.5, 5.0
    tfilterpts = [t0, t1, t2, tf]
    obsop = TimeObsPtwise({'V':V, 'Points':obspts}, tfilterpts)
    # PDE operator:
    wavepde = AcousticWave({'V':V, 'Vm':Vl})
    wavepde.timestepper = 'backward'
    wavepde.lump = True
    #wavepde.set_abc(mesh, LeftRight(), True)   # not implemented
    wavepde.update({'b':lambda_target_fn, 'a':rho_target_fn, \
    't0':tfilterpts[0], 'tf':tfilterpts[-1], 'Dt':Dt, \
    'u0init':dl.Function(V), 'utinit':dl.Function(V)})
    checkdt(Dt, 1./Nxy, 2, np.sqrt(lambdamax), True)
    # set up plots:
    if PLOT:
        filename, ext = splitext(sys.argv[0])
        if mpirank == 0 and isdir(filename + '/'):   rmtree(filename + '/')
        MPI.barrier(mpicomm)
        myplot = PlotFenics(filename)
        myplot.set_varname('lambda_target')
        myplot.plot_vtk(lambda_target_fn)
        myplot.set_varname('rho_target')
        myplot.plot_vtk(rho_target_fn)
        myplot.set_varname('lambdarho_init')
        myplot.plot_vtk(lambda_init_fn)
    else:   myplot = None

    # run
    run_testab(wavepde, mysrc, obsop, [rho_init_fn, lambda_init_fn], myplot)
    #run_testa(fpeak, [lambdamin, lambdamax], [rhomin, rhomax], Nxy, tfilterpts, r, Dt, skip)
    #run_testb(fpeak, [lambdamin, lambdamax], [rhomin, rhomax], Nxy, tfilterpts, r, Dt, skip)



#    dl.parameters['form_compiler']['cpp_optimize'] = True
#    dl.parameters['form_compiler']['cpp_optimize_flags'] = '-O2'
#    dl.parameters['form_compiler']['optimize'] = True
#    ffc.parameters.FFC_PARAMETERS['optimize'] = True
#    ffc.parameters.FFC_PARAMETERS['cpp_optimize'] = True
#    ffc.parameters.FFC_PARAMETERS['cpp_optimize_flags'] = '-O2'
