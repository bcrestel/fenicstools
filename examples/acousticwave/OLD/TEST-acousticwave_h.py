""" Test convergence of solution wrt grid spacing parameter """

import sys
from os.path import splitext, isdir
from shutil import rmtree

import dolfin as dl
from fenicstools.sourceterms import PointSources, RickerWavelet
from fenicstools.acousticwave import AcousticWave
from fenicstools.plotfenics import PlotFenics
from fenicstools.miscfenics import setfct

r = 2   # polynomial order
c = 1.0 # wave velocity
freq = 4.0  # Hz
maxfreq = 10.0
cmin = c/maxfreq
Ricker = RickerWavelet(freq, 1e-10)
Dt = 1e-4
tf = 0.5

filename, ext = splitext(sys.argv[0])
if isdir(filename + '/'):   rmtree(filename + '/')
myplot = PlotFenics(filename)
boolplot = 100

print 'Compute most accurate solution as reference'
qq = 20
N = int(qq/cmin)
h = 1./N
mesh = dl.UnitSquareMesh(N,N)
Vl = dl.FunctionSpace(mesh, 'Lagrange', 1)
Vex = dl.FunctionSpace(mesh, 'Lagrange', r)
Pt = PointSources(Vex, [[.5,.5]])
mydelta = Pt[0].array()
def mysrc(tt):
    return Ricker(tt)*mydelta
Waveex = AcousticWave({'V':Vex, 'Vm':Vl})
Waveex.timestepper = 'backward'
Waveex.lump = True
Waveex.update({'a':1.0, 'b':1.0, 't0':0.0, 'tf':tf, 'Dt':Dt,\
'u0init':dl.Function(Vex), 'utinit':dl.Function(Vex)})
Waveex.ftime = mysrc
sol,_ = Waveex.solve()
Waveex.exact = dl.Function(Vex)
normex = Waveex.computeabserror()
# plot
myplot.set_varname('u-q'+str(qq))
plotu = dl.Function(Vex)
for index, uu in enumerate(sol):
    if index%boolplot == 0:
        setfct(plotu, uu[0])
        myplot.plot_vtk(plotu, index)
myplot.gather_vtkplots()

print 'Check different spatial sampling'
QQ = [4, 5, 6, 10]
for qq in QQ:
    N = int(qq/cmin)
    h = 1./N
    mesh = dl.UnitSquareMesh(N,N)
    Vl = dl.FunctionSpace(mesh, 'Lagrange', 1)
    V = dl.FunctionSpace(mesh, 'Lagrange', r)
    Pt = PointSources(V, [[.5,.5]])
    mydelta = Pt[0].array()
    def mysrc(tt):
        return Ricker(tt)*mydelta
    Wave = AcousticWave({'V':V, 'Vm':Vl})
    Wave.timestepper = 'backward'
    Wave.lump = True
    Wave.update({'a':1.0, 'b':1.0, 't0':0.0, 'tf':tf, 'Dt':Dt,\
    'u0init':dl.Function(V), 'utinit':dl.Function(V)})
    Wave.ftime = mysrc
    sol,_ = Wave.solve()
    # error:
    Waveex.exact = dl.project(Wave.u2, Vex)
    print 'q={}, error={}'.format(qq, Waveex.computeabserror()/normex)
    # plot
    myplot.set_varname('u-q'+str(qq))
    plotu = dl.Function(V)
    for index, uu in enumerate(sol):
        if index%boolplot == 0:
            setfct(plotu, uu[0])
            myplot.plot_vtk(plotu, index)
    myplot.gather_vtkplots()
