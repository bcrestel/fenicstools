""" Script to investigate behaviour of PETScMatrix in parallel (MPI) """
import dolfin as dl
import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

from dolfin import MPI, mpi_comm_world
mycomm = mpi_comm_world()
mpisize = MPI.size(mycomm)
mpirank = MPI.rank(mycomm)

mesh = dl.UnitSquareMesh(2,2)
Vr = dl.FunctionSpace(mesh, 'Lagrange', 1)
Vc = dl.FunctionSpace(mesh, 'Lagrange', 1)
MPETSc = PETSc.Mat()
MPETSc.create(PETSc.COMM_WORLD)
MPETSc.setSizes([Vr.dim(), Vc.dim()])
MPETSc.setType('aij') # sparse
#MPETSc.setPreallocationNNZ(5)
MPETSc.setUp()
Istart, Iend = MPETSc.getOwnershipRange()
for I in xrange(Istart, Iend) :
    MPETSc[I,I] = I
    if I-1 >= 0: MPETSc[I,I-1] = 1.0
    if I-2 >= 0: MPETSc[I,I-2] = -1.0
    if I+1 < Vc.dim():  MPETSc[I,I+1] = 1.0
    if I+2 < Vc.dim():  MPETSc[I,I+2] = -1.0
MPETSc.assemblyBegin()
MPETSc.assemblyEnd()

Mdolfin = dl.PETScMatrix(MPETSc)
print 'rank={}'.format(mpirank), Mdolfin.array().shape

MPI.barrier(mycomm)
x, y = MPETSc.getVecs()
x.set(1.0)
y.set(0.0)
y = MPETSc * x
print 'rank={}'.format(mpirank), y[...]

MPI.barrier(mycomm)
x2 = dl.interpolate(dl.Constant(1.0), Vc)
print 'rank={}'.format(mpirank), len(x2.vector().array())
#y2 = Mdolfin * x2.vector() # fails -- not sure why

# workaround?
x2P = dl.as_backend_type(x2.vector()).vec()
# next line fails as petsc4py and dolfin do not have same partition
#TODO: define partitioning of PETSc matrix to match the fem one
# see Umbe's email 2016/08/29
y2P = MPETSc * x2P  
y2 = dl.PETScVector(y2P)

print 'rank={}'.format(mpirank), y2.array()
