"""
Compute objective function for different situations of wave speeds
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




def run_test(fpeak, Nxy, tfilterpts, r, Dt, skip):
    h = 1./Nxy
    checkdt(Dt, h, r, 1.0, False)
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
    # observation operator:
    #obspts = [[0.2, 0.5], [0.5, 0.2], [0.5, 0.8], [0.8, 0.5]]
    obspts = [[0.2, ii/10.] for ii in range(2,9)] + \
    [[0.8, ii/10.] for ii in range(2,9)] + \
    [[ii/10., 0.2] for ii in range(3,8)] + \
    [[ii/10., 0.8] for ii in range(3,8)]
    obsop = TimeObsPtwise({'V':V, 'Points':obspts}, tfilterpts)
    # define pde operator:
    wavepde = AcousticWave({'V':V, 'Vm':Vl})
    wavepde.timestepper = 'backward'
    wavepde.update({'b':1.0, 'a':1.0, \
    't0':tfilterpts[0], 'tf':tfilterpts[-1], 'Dt':Dt, 'u0init':dl.Function(V), 'utinit':dl.Function(V)})
    wavepde.ftime = mysrc
    # define objective function:
    waveobj = ObjectiveAcoustic(wavepde)
    waveobj.obsop = obsop
    # data
    print 'generate data'
    waveobj.solvefwd()
    myplot.plot_timeseries(waveobj.solfwd, 'pd', 0, skip, fctV)
    dd = waveobj.Bp.copy()
    waveobj.dd = dd
    # medium 1:
    med_expr = dl.Expression(\
    '1.0 + 2.0*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
    med_fn = dl.interpolate(med_expr, Vl)
    myplot.set_varname('medium1')
    myplot.plot_vtk(med_fn)
    # observations
    print 'generate observations'
    waveobj.update_PDE({'a':med_fn, 'b':med_fn})
    waveobj.solvefwd_cost()
    cost1 = waveobj.cost_misfit
    print 'misfit = {}'.format(waveobj.cost_misfit)
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

if __name__ == "__main__":
#    dl.parameters['form_compiler']['cpp_optimize'] = True
#    dl.parameters['form_compiler']['cpp_optimize_flags'] = '-O2'
#    dl.parameters['form_compiler']['optimize'] = True
#    ffc.parameters.FFC_PARAMETERS['optimize'] = True
#    ffc.parameters.FFC_PARAMETERS['cpp_optimize'] = True
#    ffc.parameters.FFC_PARAMETERS['cpp_optimize_flags'] = '-O2'
    # Inputs:
    fpeak = 8.  #Hz
    Nxy = 200
    t0, t1, t2, tf = 0.0, 0.5, 2.0, 2.5
    tfilterpts = [t0, t1, t2, tf]
    r = 2   # order polynomial approx
    Dt = 5e-4
    skip = 200
    # run
    run_test(fpeak, Nxy, tfilterpts, r, Dt, skip)
