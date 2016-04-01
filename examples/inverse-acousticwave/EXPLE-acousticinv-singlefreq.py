"""
Acoustic wave inverse problem with a single low-frequency,
and homogeneous Neumann boundary conditions everywhere.
"""

import sys
from os.path import splitext, isdir
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt

import dolfin as dl
from fenicstools.plotfenics import PlotFenics
from fenicstools.acousticwave import AcousticWave
from fenicstools.sourceterms import PointSources, RickerWavelet
from fenicstools.observationoperator import TimeObsPtwise
from fenicstools.miscfenics import checkdt, setfct
from fenicstools.objectiveacoustic import ObjectiveAcoustic
from fenicstools.optimsolver import checkgradfd_med, checkhessfd_med, checkhessfd



# ABC:
class LeftRight(dl.SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] < 1e-16 or x[0] > 1.0 - 1e-16) \
        and on_boundary

@profile
def run_test(fpeak, lambdamin, lambdamax, Nxy, tfilterpts, r, Dt, skip):
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
    Pt = PointSources(V, [[0.5,0.5]])
    mydelta = Pt[0].array()
    def mysrc(tt):
        return Ricker(tt)*mydelta
    # target medium:
    lambda_target = dl.Expression('lmin + x[0]*(lmax-lmin)', \
    lmin=lambdamin, lmax=lambdamax)
    lambda_target_fn = dl.interpolate(lambda_target, Vl)
    myplot.set_varname('lambda_target')
    myplot.plot_vtk(lambda_target_fn)
    # initial medium:
    lambda_init = dl.Constant(lambdamin)
    lambda_init_fn = dl.interpolate(lambda_init, Vl)
    myplot.set_varname('lambda_init')
    myplot.plot_vtk(lambda_init_fn)
    # observation operator:
    #obspts = [[0.2, 0.5], [0.5, 0.2], [0.5, 0.8], [0.8, 0.5]]
    obspts = [[0.2, ii/10.] for ii in range(2,9)] + \
    [[0.8, ii/10.] for ii in range(2,9)] + \
    [[ii/10., 0.2] for ii in range(3,8)] + \
    [[ii/10., 0.8] for ii in range(3,8)]
    obsop = TimeObsPtwise({'V':V, 'Points':obspts}, tfilterpts)
    # define pde operator:
    wavepde = AcousticWave({'V':V, 'Vl':Vl, 'Vr':Vl})
    wavepde.timestepper = 'backward'
    wavepde.lump = True
    wavepde.set_abc(mesh, LeftRight(), True)
    wavepde.update({'lambda':lambda_target_fn, 'rho':1.0, \
    't0':t0, 'tf':tf, 'Dt':Dt, 'u0init':dl.Function(V), 'utinit':dl.Function(V)})
    wavepde.ftime = mysrc
    # define objective function:
    waveobj = ObjectiveAcoustic(wavepde)
    waveobj.obsop = obsop
    # data
    print 'generate data'
    waveobj.solvefwd()
    myplot.plot_timeseries(waveobj.solfwd, 'pd', 0, skip, fctV)
    dd = waveobj.Bp.copy()
    # gradient
    print 'generate observations'
    waveobj.dd = dd
    waveobj.update_m(lambda_init_fn)
    waveobj.solvefwd_cost()
    cost1 = waveobj.misfit
    print 'misfit = {}'.format(waveobj.misfit)
    myplot.plot_timeseries(waveobj.solfwd, 'p', 0, skip, fctV)
    # Plot data and observations
    fig = plt.figure()
    if len(obspts) > 9: fig.set_size_inches(20., 15.)
    for ii in range(len(obspts)):
        if len(obspts) == 4:    ax = fig.add_subplot(2,2,ii+1)
        else:   ax = fig.add_subplot(4,6,ii+1)
        ax.plot(waveobj.PDE.times, waveobj.dd[ii,:], 'k--')
        ax.plot(waveobj.PDE.times, waveobj.Bp[ii,:], 'b')
        ax.set_title('Plot'+str(ii))
    fig.savefig(filename + '/observations.eps')
    print 'compute gradient'
    waveobj.solveadj_constructgrad()
    myplot.plot_timeseries(waveobj.soladj, 'v', 0, skip, fctV)
    MG = waveobj.MGv.array().copy()
    myplot.set_varname('grad')
    myplot.plot_vtk(waveobj.Grad)
    """
    print 'check gradient with FD'
    Medium = np.zeros((3, Vl.dim()))
    for ii in range(3):
        smoothperturb = dl.Expression('sin(n*pi*x[0])*sin(n*pi*x[1])', n=ii+1)
        smoothperturb_fn = dl.interpolate(smoothperturb, Vl)
        Medium[ii,:] = smoothperturb_fn.vector().array()
    checkgradfd_med(waveobj, Medium, 1e-6, [1e-5, 1e-4])
    print 'check Hessian with FD'
    checkhessfd_med(waveobj, Medium, 1e-6, [1e-1, 1e-2, 1e-3, 1e-4], False)
    """


if __name__ == "__main__":
    # Inputs:
    fpeak = 1.0  #Hz
    lambdamin = 1.0
    lambdamax = 2.0
    Nxy = 25
    t0, t1, t2, tf = 0.0, 0.5, 2.5, 3.0
    tfilterpts = [t0, t1, t2, tf]
    r = 2   # order polynomial approx
    Dt = 2.5e-3
    skip = 20
    # run
    run_test(fpeak, lambdamin, lambdamax, Nxy, tfilterpts, r, Dt, skip)
