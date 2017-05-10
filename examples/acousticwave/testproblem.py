"""
Ricker wavelet at the center of a unit square with dashpot absorbing boundary
conditions on all 4 boundaries
"""

import sys
from os.path import splitext, isdir
from shutil import rmtree

import dolfin as dl
from hippylib import vector2Function
from dolfin import MPI

from fenicstools.plotfenics import PlotFenics
from fenicstools.acousticwave import AcousticWave
from fenicstools.sourceterms import PointSources, RickerWavelet

def petsc4py2Fenics(v, V):
    u = dl.Function(V)
    dl.as_backend_type(u.vector()).vec().axpy(1.0, v)
    return u

def createparam(CC, Vl, X, H1, H2, H3, TT):
    c = dl.interpolate(dl.Expression(' \
    (x[0]>=LL)*(x[0]<=RR)*(x[1]>=HC-TT)*(x[1]<=HC+TT)*vva \
    + (1.0-(x[0]>=LL)*(x[0]<=RR)*(x[1]>=HC-TT)*(x[1]<=HC+TT))*( \
    vvb*(x[1]>HA) +  \
    vvc*(x[1]<=HA)*(x[1]>HB) + \
    vvd*(x[1]<=HB))', 
    vva=CC[0], vvb=CC[1], vvc=CC[2], vvd=CC[3],
    LL=X/4.0, RR=3.0*X/4.0, HA=H1, HB=H2, HC=H3, TT=TT), Vl)
    return c

# Inputs:
Nxy = 100
h = 1./Nxy
# dist is in [km]
X, Y = 1, 1
mesh = dl.RectangleMesh(dl.Point(0.0,0.0),dl.Point(X,Y),X*Nxy,Y*Nxy)
mpicomm = mesh.mpi_comm()
mpirank = MPI.rank(mpicomm)
Vl = dl.FunctionSpace(mesh, 'Lagrange', 1)
Dt = 1.0e-4   #Dt = h/(r*alpha)
tf = 1.0

# Plots:
filename, ext = splitext(sys.argv[0])
if mpirank == 0: 
    if isdir(filename + '/'):   rmtree(filename + '/')
MPI.barrier(mpicomm)
myplot = PlotFenics(filename)

# Source term:
fpeak = 6.0
Ricker = RickerWavelet(fpeak, 1e-6)

# Boundary conditions:
class ABCdom(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[1] < Y)

r = 2
V = dl.FunctionSpace(mesh, 'Lagrange', r)
Pt = PointSources(V, [[0.5*X,Y]])
mydelta = Pt[0]
def mysrc(tt):
    return mydelta * Ricker(tt)
# Computation:
if mpirank == 0: print '\n\th = {}, Dt = {}'.format(h, Dt)
Wave = AcousticWave({'V':V, 'Vm':Vl}, 
{'print':(not mpirank), 'lumpM':True, 'timestepper':'centered'})
Wave.set_abc(mesh, ABCdom(), lumpD=True)
Wave.exact = dl.Function(V)
Wave.ftime = mysrc
# medium parameters:
H1, H2, H3, TT = 0.8, 0.2, 0.6, 0.1
CC = [5.0, 2.0, 3.0, 4.0]
RR = [2.0, 2.1, 2.2, 2.5]
LL, AA, BB = [], [], []
for cc, rr in zip(CC, RR):
    ll = rr*cc*cc
    LL.append(ll)
    AA.append(1./ll)
    BB.append(1./rr)
# velocity is in [km/s]
c = createparam(CC, Vl, X, H1, H2, H3, TT)
myplot.set_varname('c')
myplot.plot_vtk(c)
# density is in [10^12 kg/km^3]=[g/cm^3]
# assume rocks shale-sand-shale + salt inside small rectangle
# see Marmousi2 print-out
rho = createparam(RR, Vl, X, H1, H2, H3, TT)
myplot.set_varname('rho')
myplot.plot_vtk(rho)
# bulk modulus is in [10^12 kg/km.s^2]=[GPa]
lam = createparam(LL, Vl, X, H1, H2, H3, TT)
myplot.set_varname('lambda')
myplot.plot_vtk(lam)
#
af = createparam(AA, Vl, X, H1, H2, H3, TT)
myplot.set_varname('alpha')
myplot.plot_vtk(af)
bf = createparam(BB, Vl, X, H1, H2, H3, TT)
myplot.set_varname('beta')
myplot.plot_vtk(bf)
# Check:
ones = dl.interpolate(dl.Expression('1.0'), Vl)
check1 = af.vector() * lam.vector()
erra = dl.norm(check1 - ones.vector())
assert erra < 1e-16
check2 = bf.vector() * rho.vector()
errb = dl.norm(check2 - ones.vector())
assert errb < 1e-16
#
Wave.update({'b':bf, 'a':af, 't0':0.0, 'tf':tf, 'Dt':Dt,\
'u0init':dl.Function(V), 'utinit':dl.Function(V)})

sol, error = Wave.solve()
if mpirank == 0: print 'relative error = {:.5e}'.format(error)
MPI.barrier(mesh.mpi_comm())

# Plots:
try:
    boolplot = int(sys.argv[1])
except:
    boolplot = 0
if boolplot > 0:
    myplot.set_varname('p')
    plotp = dl.Function(V)
    for index, pp in enumerate(sol):
        if index%boolplot == 0:
            plotp.vector()[:] = pp[0]
            myplot.plot_vtk(plotp, index)
    myplot.gather_vtkplots()
