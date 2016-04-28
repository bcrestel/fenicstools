import sys
import os.path
import shutil
import math
import numpy as np

import dolfin as dl
from fenicstools.imaging import ObjectiveImageDenoising
from fenicstools.plotfenics import PlotFenics
from fenicstools.miscfenics import setfct

dl.set_log_active(False)

def run_exple(PLOT=True, TEST=False):
    # testcase == 0
    print 'Run basic example -- PLOT={}. TEST={}'.format(PLOT, TEST)
    # Target data:
    data = np.loadtxt('image.dat', delimiter=',')
    Lx, Ly = float(data.shape[1])/float(data.shape[0]), 1.
    class Image(dl.Expression):
        def __init__(self, Lx, Ly, data):
            self.data = data
            self.hx = Lx/float(self.data.shape[1]-1)
            self.hy = Ly/float(self.data.shape[0]-1)

        def eval(self, values, x):
            j = math.floor(x[0]/self.hx)
            i = math.floor(x[1]/self.hy)
            values[0] = self.data[i,j]
    trueImage = Image(Lx, Ly, data)
    if dl.__version__.split('.')[1] == '5':
        mesh = dl.RectangleMesh(0,0, Lx,Ly, 200,100)
    else:
        mesh = dl.RectangleMesh(dl.Point(0.,0.), dl.Point(Lx,Ly), 200, 100)
    # Generate data 
    #denoise = ObjectiveImageDenoising(mesh, trueImage, \
    #{'regularization':'tikhonov', 'gamma':1.0, 'beta':0.0})
    denoise = ObjectiveImageDenoising(mesh, trueImage, \
    {'regularization':'TV', 'eps':1e-4, 'k':1.0, 'mode':'primaldual'})
    denoise.generatedata(0.6)
    if PLOT:
        denoise.plot(0)
        denoise.plot(1)
    # Test gradient and Hessian
    if TEST:
        print 'Test gradient and Hessian'
        denoise.test_gradient()
        denoise.test_hessian()
        sys.exit(0)
    # Solve
    #ALPHAS = 10**(-np.linspace(0.,4.,5))
    ALPHAS = [1e-2]
    for aa in ALPHAS:
        denoise.g = dl.Function(denoise.V)
        setfct(denoise.g, denoise.dn)   # start from noisy image
        denoise.regparam = aa
        denoise.solve()
        if PLOT:    denoise.plot(2,'-'+str(aa))



def run_continuation(PLOT=True, TEST=False):
    # testcase == 1
    print 'Run continuation scheme on eps -- PLOT={}. TEST={}'.format(PLOT, TEST)
    # Target data:
    data = np.loadtxt('image.dat', delimiter=',')
    Lx, Ly = float(data.shape[1])/float(data.shape[0]), 1.
    class Image(dl.Expression):
        def __init__(self, Lx, Ly, data):
            self.data = data
            self.hx = Lx/float(self.data.shape[1]-1)
            self.hy = Ly/float(self.data.shape[0]-1)

        def eval(self, values, x):
            j = math.floor(x[0]/self.hx)
            i = math.floor(x[1]/self.hy)
            values[0] = self.data[i,j]
    trueImage = Image(Lx, Ly, data)
    if dl.__version__.split('.')[1] == '5':
        mesh = dl.RectangleMesh(0,0, Lx,Ly, 200,100)
    else:
        mesh = dl.RectangleMesh(dl.Point(0.,0.), dl.Point(Lx,Ly), 200, 100)
    # Generate data 
    #denoise = ObjectiveImageDenoising(mesh, trueImage, \
    #{'regularization':'tikhonov', 'gamma':1.0, 'beta':0.0})
    denoise = ObjectiveImageDenoising(mesh, trueImage)
    denoise.generatedata(0.6)
    if PLOT:
        denoise.plot(0)
        denoise.plot(1)
    # Test gradient and Hessian
    if TEST:
        print 'Test gradient and Hessian'
        denoise.test_gradient()
        denoise.test_hessian()
        sys.exit(0)
    # Solve
    denoise.regparam = 1e-2
    EPS = 10**(-np.linspace(0.,4.,5))
    #EPS = [1.0]
    denoise.g = dl.Function(denoise.V)
    for eps in EPS:
        print 'eps={}'.format(eps)
        paramregul = {'regularization':'TV', 'eps':eps, \
        'k':1.0, 'mode':'full'}
        denoise.define_regularization(paramregul)
        denoise.solve()
        if PLOT:    denoise.plot(2,'-'+str(eps))


if __name__ == "__main__":
    # choose test case
    # 0 = basic exple
    # 1 = continuation on eps
    try:
        testcase = int(sys.argv[1])
    except:
        testcase = 0
    # choose options
    try:
        PLOT = bool(sys.argv[2])
        TEST = bool(sys.argv[3])
    except:
        PLOT = True
        TEST = False
    # run
    if testcase == 0:   run_exple(PLOT, TEST)
    else:   run_continuation(PLOT, TEST)

