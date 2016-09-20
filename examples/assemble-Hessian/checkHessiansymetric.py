""" Check that Hessian is symmetric """
import sys
import numpy as np
from dolfin import PETScMatrix, mpi_comm_world
from fenicstools.linalg.miscroutines import loadPETScmatrixfromfile

mpicomm = mpi_comm_world()

try:
    filename = sys.argv[1]
except:
    print '*** Error: provide name of a file to load'
    sys.exit(1)

H = PETScMatrix(loadPETScmatrixfromfile(filename, mpicomm, 'dense')).array()
relerr = np.abs((H - H.T)/H)
print 'max rel err in symmetry of Hessian: {}'.format(np.max(relerr))
