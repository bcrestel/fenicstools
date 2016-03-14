"""
Acoustic wave inverse problem with a single frequency
Plot how data misfit evolves along smooth perturbations of the medium
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


class ABC(dl.SubDomain):
    def inside(self, x, on_boundary):
        return (x[1] < 1.0 - 1e-16) and on_boundary

# low frequency:
#fpeak = 0.4 # Hz
#Nxy = 10
#Dt = 2.5e-3
#t0, tf = 0.0, 7.0
# high frequency: 
fpeak = 4. # Hz
Nxy = 100
Dt = 2.5e-4
t0, tf = 0.0, 2.0
# parameters
cmin = 2.0
cmax = 3.0
epsmin = -1.0   # medium perturbation
epsmax = 1.0    # medium perturbation
h = 1./Nxy
r = 2
checkdt(Dt, h, r, cmax+epsmax, True)
# mesh
mesh = dl.UnitSquareMesh(Nxy, Nxy)
Vl, V = dl.FunctionSpace(mesh, 'Lagrange', 1), dl.FunctionSpace(mesh, 'Lagrange', r)
fctV = dl.Function(V)
fctVl = dl.Function(Vl)
# set up plots:
filename, ext = splitext(sys.argv[0])
if isdir(filename + '/'):   rmtree(filename + '/')
myplot = PlotFenics(filename)
# source:
Ricker = RickerWavelet(fpeak, 1e-10)
Pt = PointSources(V, [[.5,1.]])
mydelta = Pt[0].array()
def mysrc(tt):
    return Ricker(tt)*mydelta
# target medium
medformula = 'A*A*(x[1]>=0.5) + 2.0*2.0*(x[1]<0.5)'
TargetMedExpr = dl.Expression(medformula, A=1.5)
TargetMed = dl.interpolate(TargetMedExpr, Vl)
myplot.set_varname('target_medium')
myplot.plot_vtk(TargetMed)
# perturbation
#PerturbationMedExpr = dl.Expression(medformula, A=1.0)
#PerturbMed = dl.interpolate(PerturbationMedExpr, Vl)
#myplot.set_varname('perturb_medium')
#myplot.plot_vtk(PerturbMed)
# observation operator:
obspts = [[x/10.,1.0] for x in range(1,10)]
obsop = TimeObsPtwise({'V':V, 'Points':obspts})
# define pde operator:
wavepde = AcousticWave({'V':V, 'Vl':Vl, 'Vr':Vl})
wavepde.timestepper = 'backward'
wavepde.lump = True
wavepde.set_abc(mesh, ABC(), True)
wavepde.update({'lambda':TargetMed, 'rho':1.0, \
't0':t0, 'tf':tf, 'Dt':Dt, 'u0init':dl.Function(V), 'utinit':dl.Function(V)})
wavepde.ftime = mysrc
# define objective function:
waveobj = ObjectiveAcoustic(wavepde)
waveobj.obsop = obsop
# data
print 'generate data'
waveobj.solvefwd()
myplot.plot_timeseries(waveobj.solfwd, 'pd', 0, 40, fctV)
dd = waveobj.Bp.copy()
waveobj.dd = dd

# Plot observations
#fig = plt.figure()
#for ii in range(len(obspts)):
#    ax = fig.add_subplot(3,3,ii+1)
#    ax.plot(waveobj.times, waveobj.dd[ii,:], 'k--')
#    ax.plot(waveobj.times, waveobj.Bp[ii,:], 'r--')
#    ax.set_title('recv '+str(ii+1))
#fig.savefig(filename + '/observations.eps')
# perturbate medium
V = np.linspace(1.0, 3.0, 20)
MISFIT = []
for ii, eps in enumerate(V):
    print 'run case ', ii
    PerturbationMedExpr = dl.Expression(medformula, A=eps)
    PerturbMed = dl.interpolate(PerturbationMedExpr, Vl)
    #setfct(fctVl, TargetMed)
    #fctVl.vector().axpy(eps, PerturbMed.vector())
    waveobj.update_m(PerturbMed)
    waveobj.solvefwd_cost()
    MISFIT.append(waveobj.misfit)
# plot result:
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(V, MISFIT)
plt.show()

