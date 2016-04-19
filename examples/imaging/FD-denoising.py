""" Solve denoising problem using FD discretization """

import numpy as np
import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.misc import imresize


def compute_costTV(Dx, Dy, H, f, data, psi):
    # Dx*f and Dy*f
    Dxf, Dyf = Dx.getVecLeft(), Dy.getVecLeft()
    Dx.mult(f, Dxf)
    Dy.mult(f, Dyf)
    # Psi'*Dxf and Psi'*Dyf
    Istart, Iend = Dxf.getOwnershipRange()
    PP = Dx.getVecLeft()
    for ii in xrange(Istart, Iend):
        Dxfij, Dyfij = Dxf.getValue(ii), Dyf.getValue(ii)
        Df2 = Dxfij**2 + Dyfij**2
        PP[ii] = psi(Df2)
    return 0.5*H*PP.sum()
    

def assemble_gradientTV(Dx, Dy, H, f, psip):
    # Dx*f and Dy*f
    Dxf, Dyf = Dx.getVecLeft(), Dy.getVecLeft()
    Dx.mult(f, Dxf)
    Dy.mult(f, Dyf)
    # Psi'*Dxf and Psi'*Dyf
    Istart, Iend = Dxf.getOwnershipRange()
    PP = Dx.getVecLeft()
    for ii in xrange(Istart, Iend):
        Dxfij, Dyfij = Dxf.getValue(ii), Dyf.getValue(ii)
        Df2 = Dxfij**2 + Dyfij**2
        PP[ii] = psip(Df2)
    PDxf, PDyf = PP*Dxf, PP*Dyf
    # gradient
    DxPDxf, DyPDyf = Dx.getVecRight(), Dy.getVecRight()
    Dx.multTranspose(PDxf, DxPDxf)
    Dy.multTranspose(PDyf, DyPDyf)
    return H*(DxPDxf + DyPDyf)


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

def assemble_Dy(m, n, hy):
    """ Assemble PETSc matrix for first order centered-difference
    pb of size m x n, and hx = grid size along x-direction """
    Dy = PETSc.Mat()
    Dy.create(PETSc.COMM_WORLD)
    Dy.setSizes([(m-2)*(n-2), m*n])
    Dy.setType('aij') # sparse
    Dy.setPreallocationNNZ(2)
    Dy.setUp()
    # pre-compute entry
    val = 1.0/(2.0*hy)
    # fill matrix
    rows = np.concatenate([[ii, ii] for ii in xrange(m-2)])
    cols = np.concatenate([[ii+1, ii+1+2*m] for ii in xrange(m-2)])
    vals = np.concatenate([[-val, val] for ii in xrange(m-2)])
    for jj in xrange(n-2):
        ROWS, COLS = rows+jj*(m-2), cols+jj*m
        for rr, cc, vv in zip(ROWS, COLS, vals):
            Dy[rr,cc] = vv
    # Assemble
    Dy.assemblyBegin()
    Dy.assemblyEnd()
    return Dy



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

def test_Dxy():
    # parameters
    m, n = 101, 101
    hx, hy = 1.0/(m-1), 1.0/(n-1)
    # assemble Dx
    Dx = assemble_Dx(m, n, hx)
    Dy = assemble_Dy(m, n, hy)
    # Create rhs
    X, Y = np.mgrid[0.0:1.0:1j*m,0.0:1.0:1j*n]
    Zx = np.cos(2*np.pi*X)
    Zy = np.cos(2*np.pi*Y)
    plot_data(Zx, 1.0, 1.0)
    plot_data(Zy, 1.0, 1.0)
    fx = Zx.T.reshape((m*n,))
    fy = Zy.T.reshape((m*n,))
    x, y = Dx.getVecs()
    print 'Test Dx'
    x.setArray(fx)
    # Compute y = Dx * x
    Dx.mult(x, y)
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y[:m-2])
    yarr =y[:].reshape((m-2, n-2)).T
    plot_data(yarr, 1.0, 1.0)
    print 'Test Dy'
    x.setArray(fy)
    # Compute y = Dx * x
    Dy.mult(x, y)
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    yarr =y[:].reshape((m-2, n-2)).T
    ax.plot(yarr[0,:m-2])
    plot_data(yarr, 1.0, 1.0)

if __name__ == "__main__":
    #test_Dxy()
    # original data
    truedata, Lx, Ly = read_data()
    m, n = truedata.shape
    hx, hy = Lx/m, Ly/n
    H = hx*hy
    #plot_data(truedata, Lx, Ly)
    # noisy data
    data = truedata + 0.3*np.random.randn(m*n).reshape((m,n))
    #plot_data(data, Lx, Ly)
    # Assemble Dx, Dy
    Dx = assemble_Dx(m, n, hx)
    Dy = assemble_Dy(m, n, hy)
    eps = 1e-2
    def psi(x): return np.sqrt(x + eps)
    def diffpsi(x): return 1.0/np.sqrt(x + eps)
    # cost and gradient
    image = Dx.getVecRight()    # initial state
    image.set(0.0)
    cost = compute_costTV(Dx, Dy, H, image, data, psi)
    grad = assemble_gradientTV(Dx, Dy, H, image, diffpsi)
    #TODO: Test gradient with FD
