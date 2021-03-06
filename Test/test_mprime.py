import dolfin as dl
import numpy as np
import sys

from fenicstools.linalg.lumpedmatrixsolver import LumpedMatrixSolverS, LumpedMassMatrixPrime
from fenicstools.miscfenics import setfct

from dolfin import MPI, mpi_comm_world
mycomm = mpi_comm_world()
mpisize = MPI.size(mycomm)
mpirank = MPI.rank(mycomm)


#@profile
def run():
    mesh = dl.UnitSquareMesh(50,50)
    Vr = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vphi = dl.FunctionSpace(mesh, 'Lagrange', 2)
    Vphidofmap = Vphi.dofmap().dofs()
    test, trial = dl.TestFunction(Vphi), dl.TrialFunction(Vphi)
    u, v = dl.Function(Vphi), dl.Function(Vphi)
    rho = dl.Function(Vr)
    Mweak = dl.inner(rho*test, trial)*dl.dx
    Mprime = LumpedMassMatrixPrime(Vr, Vphi, None)
    h = 1e-5
    fact = [1.0, -1.0]

    RHO = \
    [dl.interpolate(dl.Expression('2.0 + sin(n*pi*x[0])*sin(n*pi*x[1])', n=1.0, degree=10), Vr), \
    dl.interpolate(dl.Expression('2.0 + sin(n*pi*x[0])*sin(n*pi*x[1])', n=8.0, degree=10), Vr), \
    dl.interpolate(dl.Expression('2.0 + sin(2*pi*x[0])*sin(1*pi*x[1])*(x[0]<0.5)', degree=10), Vr), \
    dl.interpolate(dl.Expression('2.0 + 2.0*(x[0]<0.5)*(x[1]<0.5) - 1.8*(x[0]>=0.5)*(x[1]<0.5)', degree=10), Vr)]

    np.random.seed(11)
    locsize = len(u.vector().array())
    for jj, rho1 in enumerate(RHO):
        if mpirank == 0:    print '\nmedium {}'.format(jj)
        setfct(rho, rho1)
        M = dl.assemble(Mweak)
        Ml = LumpedMatrixSolverS(Vphi)
        Ml.set_operator(M)
        Mprime.updater(Ml.ratio)
        for ii in range(5):
            rndvecu = np.random.randn(Vphi.dim())
            rndvecv = np.random.randn(Vphi.dim())
            if mpirank == 0:    print 'test {}'.format(ii)
            rnddir = dl.interpolate(dl.Expression('2.0+sin(n*pi*x[0])*sin(n*pi*x[1])',\
            n=ii+1, degree=10), Vr)
            setfct(u, rndvecu[Vphidofmap])
            setfct(v, rndvecv[Vphidofmap])
            analytical = rnddir.vector().inner(Mprime.get_gradient(u.vector(), v.vector()))
            uMv = []
            for ff in fact:
                setfct(rho, rho1)
                rho.vector().axpy(ff*h, rnddir.vector())
                M = dl.assemble(Mweak)
                Ml = LumpedMatrixSolverS(Vphi)
                Ml.set_operator(M)
                uMv.append(u.vector().inner(Ml*v.vector()))
            fd = (uMv[0]-uMv[1])/(2*h)
            err = np.abs((analytical-fd)/analytical)
            if mpirank == 0:    
                print 'analytical={}, fd={}, err={}'.format(analytical, fd, err),
                if err < 1e-6:  print '\t =>> OK!!'
                else:   print ''
            MPI.barrier(mycomm)


if __name__ == "__main__":
    run()
