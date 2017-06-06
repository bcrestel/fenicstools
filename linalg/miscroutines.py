""" Implement a few general command for linear algebra in Fenics with PETSc as
the linear algebra backend """

import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    pass
from dolfin import as_backend_type, PETScVector, PETScMatrix, SLEPcEigenSolver
from dolfin import MPI

import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc


def get_diagonal(M):
    """ Extract diagonal of a square Matrix M and return a PETScVector """

    return PETScVector(as_backend_type(M).mat().getDiagonal())



def setglobalvalue(fct, globalindex, value):
    """ set the globalindex of a Fenics Function fct to value, i.e., 
    fct.vector()[globalindex] = value """

    vec = fct.vector()
    V = fct.function_space()
    dof2globindex = V.dofmap().dofs()
    if vec.owns_index(globalindex):
        localindex = np.array(np.where(dof2globindex == globalindex)[0], \
        dtype=np.intc)
        vec.set_local(value*np.ones(localindex.shape), localindex)
    vec.apply('insert')



def setupPETScmatrix(Vr, Vc, mattype, mpicomm):
    """ 
    Set up a PETSc matrix partitioned consistently with Fenics mesh
    Vr, Vc = function spaces for the rows and columns
    """
    
    # extract local to global map for each fct space
    VrDM, VcDM = Vr.dofmap(), Vc.dofmap()
    r_map = PETSc.LGMap().create(VrDM.dofs(), comm=mpicomm)
    c_map = PETSc.LGMap().create(VcDM.dofs(), comm=mpicomm)
    # set up matrix
    petscmatrix = PETSc.Mat()
    petscmatrix.create(mpicomm)
    petscmatrix.setSizes([ [VrDM.local_dimension("owned"), Vr.dim()], \
    [VcDM.local_dimension("owned"), Vc.dim()] ])
    petscmatrix.setType(mattype) # 'aij', 'dense'
    petscmatrix.setUp()
    petscmatrix.setLGMap(r_map, c_map)
    # compare PETSc and Fenics local partitions:
    Istart, Iend = petscmatrix.getOwnershipRange()
    assert list(VrDM.dofs()) == range(Istart, Iend)
    return petscmatrix, VrDM, VcDM



def loadPETScmatrixfromfile(filename, mpicomm, mattype='aij'):
    """ load PETSc matrix that was saved to file *.dat
    Assumed to be done in serial """

    viewer = PETSc.Viewer().createBinary(filename, mode='r')
    petscmatrix = PETSc.Mat()
    petscmatrix.create(mpicomm)
    petscmatrix.setType(mattype)
    petscmatrix.load(viewer)
    return petscmatrix



def plotPETScmatrix(Matrix, log=0):
    """ Plot PETSc matrix as 2d data array """

    # Convert to numpy array
    Array = PETScMatrix(Matrix).array()
    if log != 0:
        # take log of absolute value
        Arraylog = np.log(np.abs(Array))
        Arrayplot = Arraylog
        #mycmap = plt.cm.Greys
        mycmap = plt.cm.seismic
        #myvmin, myvmax = min(0.0, np.min(Arraylog)), max(0.0, np.max(Arraylog))
    else:   
        Arrayplot = Array
        mycmap = plt.cm.seismic
        absmax = np.max(np.abs(Array))
        #myvmin, myvmax = -absmax, absmax
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    hh = ax.imshow(Arrayplot, cmap=mycmap, aspect='equal', \
    interpolation='nearest', norm=None)#, vmin=myvmin, vmax=myvmax)
    plt.colorbar(hh)
    return fig, Array



def gathermatrixrows(Matrix, filenames, rows, mpicomm, mattype):
    """ Assemble complete matrix from partition of the rows 
    Inputs:
        Matrix = unassembled PETSc matrix to store whole matrix
        filenames = list of filenames for each *.dat file
        rows = list of indices of the rows corresponding to each *.dat files
        mattype = string defining mattype of matrix
    """
    for ff, rr in zip(filenames, rows):
        mm = loadPETScmatrixfromfile(ff, mpicomm, mattype)
        for r in rr:
            Matrix[r,:] = mm[r,:]
    Matrix.assemblyBegin()
    Matrix.assemblyEnd()


def compute_eig(M, filename):
    """ Compute eigenvalues of a PETScMatrix M,
    and print to filename """
    mpirank = MPI.rank(M.mpi_comm())

    if mpirank == 0:    print '\t\tCompute eigenvalues'
    eigsolver = SLEPcEigenSolver(M)
    eigsolver.solve()

    if mpirank == 0:    print '\t\tSort eigenvalues'
    eig = []
    for ii in range(eigsolver.get_number_converged()):
        eig.append(eigsolver.get_eigenvalue(ii)[0])
    eig.sort()

    if mpirank == 0:    print '\t\tPrint results to file'
    np.savetxt(filename, np.array(eig))

def compute_eigfenics(M, filename):
    """ Compute eigenvalues of a Fenics matrix M,
    and print to filename """
    compute_eig(as_backend_type(M), filename)

