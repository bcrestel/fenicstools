""" Put smaller pieces of the Hessian matrix together,
save it, then plot it
"""
import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc

from fenicstools.linalg.miscroutines import gathermatrixrows, plotPETScmatrix, \
setupPETScmatrix, loadPETScmatrixfromfile


#prefix = 'Hessian4.0-1src-1rcv/Hessian4.0_'
prefix = 'Hessian4.0-5src-27rcv/Hessian4.0_'
#prefix = 'Hessian8.0-5src-27rcv/Hessian8.0_'
Nxy = 50
mysize = 16

# Set up PETSc matrix to store entire Hessian
mesh = dl.UnitSquareMesh(Nxy, Nxy)
Vm = dl.FunctionSpace(mesh, 'Lagrange', 1)
listdofmap = Vm.dofmap().tabulate_all_coordinates(mesh).reshape((-1,2))
np.savetxt(prefix + 'dof.txt', listdofmap)
mpicomm = dl.mpi_comm_world()
Hessian,_,_ = setupPETScmatrix(Vm, Vm, 'dense', mpicomm)

# Get filenames and range of rows for each sub-piece
filenames = []
rows = []
for myrank in range(mysize):
    a = myrank*(Vm.dim()/mysize)
    if myrank+1 < mysize:
        b = (myrank+1)*(Vm.dim()/mysize)+5
    else:
        b = Vm.dim()
    filenames.append(prefix + str(a) + '-' + str(b) + '.dat')
    rows.append(range(Vm.dim())[a:b])

# Fill in the Hessian
H0 = loadPETScmatrixfromfile(filenames[0], mpicomm, 'dense')
gathermatrixrows(Hessian, filenames, rows, mpicomm, 'dense')
# Save the Hessian
Hessfilename = prefix + 'full.dat'
myviewer = PETSc.Viewer().createBinary(Hessfilename, \
mode='w', format=PETSc.Viewer.Format.NATIVE, comm=mpicomm)
myviewer(Hessian)
# Plot the Hessian
fig,_ = plotPETScmatrix(Hessian, 0)
fig.savefig(prefix + 'full.eps')
# Plot its diagonal
x, y = H0.getVecs()
H0.getDiagonal(y)
fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.semilogy(y[:], 'o')
fig2.savefig(prefix + 'full-diagonal.eps')
print np.where(y[:] < 0.0)
