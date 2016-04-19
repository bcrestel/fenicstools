""" Solve denoising problem using FD discretization """

import numpy as np
import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.misc import imresize

def read_data(filename='image.dat'):
    """ Read data file
    outputs:
        data = np.array of data points
        Lx, Ly = length in x and y direction
    """
    data = np.loadtxt(filename, delimiter=',')
    data = data.T
    data = imresize(data, (200,100), 'nearest')
    data = np.round(data/np.max(data))
    Lx, Ly = 1., float(data.shape[1])/float(data.shape[0])
    return data, Lx, Ly

def plot_data(data, Lx, Ly):
    m, n = data.shape
    X, Y = np.mgrid[0:Lx:1j*m, 0:Ly:1j*n]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(X, Y, data, int(np.sqrt(m*n)/2.), cmap=cm.Greys)
    ax.axis('equal')
    #ax.colorbar()
    plt.show()

def assemble_Dx(m, n, hx):
    """ Assemble PETSc matrix for first order centered-difference
    pb of size m x n, and hx = grid size along x-direction """
    Dx = PETSc.Mat()
    Dx.create(PETSc.COMM_WORLD)
    Dx.setSizes([(m-2)*(n-2), m*n])
    Dx.setType('aij') # sparse
    Dx.setPreallocationNNZ(2)
    Dx.setUp()
    # pre-compute entry
    val = 1.0/(2.0*hx)
    # fill matrix
    rows = np.concatenate([[ii, ii] for ii in xrange(m-2)])
    cols = np.concatenate([[ii, ii+2] for ii in xrange(m-2)])
    vals = np.concatenate([[-val, val] for ii in xrange(m-2)])
    for jj in xrange(n-2):
        ROWS, COLS = rows+jj*(m-2), cols+(jj+1)*m
        for rr, cc, vv in zip(ROWS, COLS, vals):
            Dx[rr,cc] = vv
    # Assemble
    Dx.assemblyBegin()
    Dx.assemblyEnd()
    return Dx

def test_Dx():
    # parameters
    m, n = 101, 101
    hx = 1.0/(m-1)
    # assemble Dx
    Dx = assemble_Dx(m, n, hx)
    # Create rhs
    X, Y = np.mgrid[0.0:1.0:1j*m,0.0:1.0:1j*n]
    Z = X**2
    plot_data(Z, 1.0, 1.0)
    f = Z.T.reshape((m*n,))
    x, y = Dx.getVecs()
    x.setArray(f)
    # Compute y = Dx * x
    Dx.mult(x, y)
    print y[:m-2]
    yarr =y[:].reshape((m-2, n-2)).T
    plot_data(yarr, 1.0, 1.0)
    

if __name__ == "__main__":
    data, Lx, Ly = read_data()
    m, n = data.shape
    hx, hy = Lx/m, Ly/n
    #plot_data(data, Lx, Ly)
    #Dx = assemble_Dx(m, n, hx)
    #Dx = assemble_Dx(5, 4, 0.1)
    test_Dx()
