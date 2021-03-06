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




def run_exple(denoise, PLOT=True, TEST=False):
    # testcase == 0
    print 'Run basic example -- PLOT={}. TEST={}'.format(PLOT, TEST)
    # Solve
    ALPHAS = 10**(np.linspace(2.,-2.,5))
    #ALPHAS = [1e-2]
    for aa in ALPHAS:
        print '\n\tRegularization parameter={:f}'.format(aa)
        denoise.alpha = aa
        denoise.solve()
        if PLOT:    denoise.plot(2,'-'+str(aa))


def run_continuation(denoise, PLOT=True, TEST=False):
    # testcase == 1
    print 'Run continuation scheme on eps -- PLOT={}. TEST={}'.format(PLOT, TEST)
    # Solve
    denoise.regparam = 1e-2
    EPS = 10**(-np.linspace(0.,4.,5))
    #EPS = [1.0]
    denoise.g = dl.Function(denoise.V)
    setfct(denoise.g, denoise.dn)   # start from noisy image
    for eps in EPS:
        print 'eps={}'.format(eps)
        paramregul = {'regularization':'TV', 'eps':eps, 'k':1.0, 'GN':False}
        denoise.define_regularization(paramregul)
        denoise.solve()
        if PLOT:    denoise.plot(2,'-'+str(eps))



########################################################################
########################################################################
########################################################################

#denoise = ObjectiveImageDenoising(1, 'tikhonov', image='image2.dat')

#denoise = ObjectiveImageDenoising(1, 'TV', parameters={'eps':1e-2, 'k':1.0, 'GNhessian':False})
#denoise = ObjectiveImageDenoising(1, 'TVPD', parameters={'eps':1e-6, 'k':1e-2})

#denoise = ObjectiveImageDenoising(1, 'TV', h=1.00, parameters={'eps':1e-10, 'k':1.0, 'GNhessian':False})
denoise = ObjectiveImageDenoising(1, 'TVPD', h=1.00, parameters={'eps':1e-6, 'k':1.0})

denoise.alpha = 1.0

print 'a={}, k={}, eps={}'.format(\
denoise.alpha, denoise.Regul.parameters['k'], denoise.Regul.parameters['eps'])

denoise.solve()

"""
# choose test case
# 0 = basic exple
# 1 = continuation on eps
testcase = 0

# choose options
PLOT = 0
FDCHECK = 0

# plot
if PLOT:
    denoise.plot(0)
    denoise.plot(1)

# test gradient and Hessian
if FDCHECK:
    print 'Test gradient'
    denoise.test_gradient()
    print '\nTest Hessian'
    denoise.test_hessian()
    sys.exit(0)

# run
if testcase == 0:   
    run_exple(denoise, PLOT, FDCHECK)
else:   
    run_continuation(denoise, PLOT, FDCHECK)
"""

