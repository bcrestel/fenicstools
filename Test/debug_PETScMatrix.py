""" 
Script to investigate behaviour of PETScMatrix in parallel (MPI) 
Partitioning of PETSc and Fenics (mesh) are both monotonic, but they
don't split in the same chunk sizes.
"""
import dolfin as dl
import numpy as np
import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

from dolfin import MPI, mpi_comm_world
#mycomm = mpi_comm_world()
mycomm = PETSc.COMM_WORLD
mpisize = MPI.size(mycomm)
mpirank = MPI.rank(mycomm)

mesh = dl.UnitSquareMesh(100,100)
Vr = dl.FunctionSpace(mesh, 'Lagrange', 1)
Vc = dl.FunctionSpace(mesh, 'Lagrange', 1)
#
Vrdofmap, Vcdofmap = Vr.dofmap(), Vc.dofmap()
#print 'rank={}, Vr dofmap:'.format(mpirank), Vr.dofmap().dofs()
#print 'rank={}, Vc dofmap:'.format(mpirank), Vc.dofmap().dofs()
rmap = PETSc.LGMap().create(Vrdofmap.dofs(), mycomm)
cmap = PETSc.LGMap().create(Vcdofmap.dofs(), mycomm)
"""
if mpirank == 0:
    gindices = PETSc.IS().createGeneral([6,7,8])
    #gindices = [6,7,8]
else:
    gindices = PETSc.IS().createGeneral([0,1,2,3,4,5])
    #gindices = [0,1,2,3,4,5]
rmap = PETSc.LGMap().create(gindices, mycomm)
cmap = PETSc.LGMap().create(gindices, mycomm)
print 'rank={}, rmap indices:'.format(mpirank), rmap.getIndices()
print 'rank={}, cmap indices:'.format(mpirank), cmap.getIndices()
"""
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
#MPETSc.getDiagonal().view()

#print 'rank={}, MPETSc shape:'.format(mpirank), MPETSc.getLocalSize()
Mdolfin = dl.PETScMatrix(MPETSc)
#print 'rank={}, Mdolfin:'.format(mpirank), Mdolfin.array()


# Compare petsc and fenics dofs
if not list(Vrdofmap.dofs()) == range(Istart, Iend):
    print 'rank={}'.format(mpirank), list(Vrdofmap.dofs), range(Istart, Iend)



"""
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
y2 = Mdolfin * x2.vector()

MPI.barrier(mycomm)
if mpirank == 0:    print 'Do matvec with PETSc wrapper in fenics'
x2P = dl.as_backend_type(x2.vector()).vec()
y2P = MPETSc * x2P  
y2 = dl.PETScVector(y2P)
print 'rank={}'.format(mpirank), y2.array()
yy = dl.Vector()
y2.gather(yy, np.array(range(Vr.dim()), "intc"))
print yy.array()

"" "
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
