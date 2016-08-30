""" Script to investigate behaviour of PETScMatrix in parallel (MPI) """
import dolfin as dl
import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

from dolfin import MPI, mpi_comm_world
#mycomm = mpi_comm_world()
mycomm = PETSc.COMM_WORLD
mpisize = MPI.size(mycomm)
mpirank = MPI.rank(mycomm)

mesh = dl.UnitSquareMesh(2,2)
Vr = dl.FunctionSpace(mesh, 'Lagrange', 1)
Vc = dl.FunctionSpace(mesh, 'Lagrange', 2)
#
print 'rank={}, dofmap:'.format(mpirank), Vr.dofmap().dofs()
Vrdofmap, Vcdofmap = Vr.dofmap(), Vc.dofmap()
rmap = PETSc.LGMap().create(Vrdofmap.dofs(), mycomm)
cmap = PETSc.LGMap().create(Vcdofmap.dofs(), mycomm)
#
MPETSc = PETSc.Mat()
MPETSc.create(PETSc.COMM_WORLD)
MPETSc.setSizes([ [Vrdofmap.local_dimension("owned"), Vr.dim()], \
[Vcdofmap.local_dimension("owned"), Vc.dim()] ])
MPETSc.setType('aij') # sparse
#MPETSc.setPreallocationNNZ(5)
MPETSc.setUp()
MPETSc.setLGMap(rmap, cmap)
#
Istart, Iend = MPETSc.getOwnershipRange()
for I in xrange(Istart, Iend) :
    MPETSc[I,I] = I
    if I-1 >= 0: MPETSc[I,I-1] = 1.0
    if I-2 >= 0: MPETSc[I,I-2] = -1.0
    if I+1 < Vc.dim():  MPETSc[I,I+1] = 1.0
    if I+2 < Vc.dim():  MPETSc[I,I+2] = -1.0
MPETSc.assemblyBegin()
MPETSc.assemblyEnd()

print 'rank={}, MPETSc shape:'.format(mpirank), MPETSc.getLocalSize()
Mdolfin = dl.PETScMatrix(MPETSc)
print 'rank={}, Mdolfin:'.format(mpirank), Mdolfin.array()

MPI.barrier(mycomm)
if mpirank == 0:    print 'Do matvec in PETSc'
x, y = MPETSc.getVecs()
x.set(1.0)
y.set(0.0)
y = MPETSc * x
print 'rank={}, y:'.format(mpirank), y[...]

MPI.barrier(mycomm)
if mpirank == 0:    print 'Do the matvec with fenics'
x2 = dl.interpolate(dl.Constant(1.0), Vc)
print 'rank={}, len(x2):'.format(mpirank), len(x2.vector().array())
y2 = Mdolfin * x2.vector()

MPI.barrier(mycomm)
if mpirank == 0:    print 'Do matvec with PETSc wrapper in fenics'
x2P = dl.as_backend_type(x2.vector()).vec()
y2P = MPETSc * x2P  
y2 = dl.PETScVector(y2P)
print 'rank={}'.format(mpirank), y2.array()


"""
std::shared_ptr<const GenericDofMap> dofmap = Vh.dofmap();
PetscInt global_dof_dimension = dofmap->global_dimension();
PetscInt local_dof_dimension = dofmap->local_dimension("owned");
std::vector<dolfin::la_index> LGdofs = dofmap->dofs();

MatCreate(comm,&mat);
MatSetSizes(mat,local_nrows,local_dof_dimension,global_nrows,global_dof_dimension);
MatSetType(mat,MATAIJ);
MatSetUp(mat);
ISLocalToGlobalMapping rmapping, cmapping;
PetscCopyMode mode = PETSC_COPY_VALUES;
#if PETSC_VERSION_LT(3,5,0)
ISLocalToGlobalMappingCreate(comm, LGrows.size(), &LGrows[0], mode, &rmapping);
ISLocalToGlobalMappingCreate(comm, LGdofs.size(),&LGdofs[0],mode,&cmapping);
#else
PetscInt bs = 1;
ISLocalToGlobalMappingCreate(comm, bs, LGrows.size(), &LGrows[0], mode, &rmapping);
ISLocalToGlobalMappingCreate(comm, bs, LGdofs.size(),&LGdofs[0],mode,&cmapping);
#endif
MatSetLocalToGlobalMapping(mat,rmapping,cmapping);
"""
