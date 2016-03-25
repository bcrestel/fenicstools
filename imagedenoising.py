
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import dolfin as dl

from miscfenics import setfct
from prior import LaplacianPrior

class ObjectiveImageDenoising1D():
    """
    Class for linear 1D image denoising problem, built on an integral (blurring) kernel
    """

    def __init__(self, mesh, k, regularization='tikhonov'):
        """
        Inputs:
            mesh = Fenics mesh
            k = Fenics Expression of the blurring kernel; must have parameter t
            f = target image
        """
        self.mesh = mesh
        self.V = dl.FunctionSpace(self.mesh, 'Lagrange', 1)
        self.dimV = self.V.dim()
        self.xx = self.V.dofmap().tabulate_all_coordinates(self.mesh)
        # Target data:
        self.f = 0.75*(self.xx>=.1)*(self.xx<=.25)
        self.f += (self.xx>=0.28)*(self.xx<=0.3)*(15*self.xx-15*0.28)
        self.f += (self.xx>0.3)*(self.xx<0.33)*0.3
        self.f += (self.xx>=0.33)*(self.xx<=0.35)*(-15*self.xx+15*0.35)
        self.f += (self.xx>=.5)*(self.xx-.5)**2*(self.xx-1.0)**2/.25**4
        self.g = None   # current iterate
        # kernel operator
        self.k = k
        self.test, self.trial = dl.TestFunction(self.V), dl.TrialFunction(self.V)
        self.Kweak = dl.inner(self.k, self.test)*dl.dx
        self.assembleK()
        # mass matrix
        self.Mweak = dl.inner(self.test, self.trial)*dl.dx
        self.M = dl.assemble(self.Mweak).array()
        # regularization
        self.regularization = regularization
        if self.regularization == 'tikhonov':
            self.RegTikh = LaplacianPrior({'gamma':1.0, 'beta':0.0, 'Vm':self.V})
            self.R = self.RegTikh.Minvprior.array()

    def assembleK(self):
        self.K = np.zeros((self.dimV, self.dimV))
        for ii, tt in enumerate(self.xx):
            self.k.t = tt
            self.K[ii,:] = dl.assemble(self.Kweak).array()
        
    def generatedata(self, noisepercent):
        """ compute data and add noisepercent (%) of noise """
        self.d = self.K.dot(self.f)
        sigma = noisepercent*np.linalg.norm(self.d)/np.sqrt(self.dimV)
        eta = sigma*np.random.randn(self.dimV)
        print 'noise residual={}'.format(np.linalg.norm(eta)**2)
        self.dn = self.d + eta

    def update_reg(self, parameters):
        if self.regularization == 'tikhonov':
            self.gamma = parameters
            self.R = self.RegTikh.Minvprior.array()*self.gamma


    ### COST and DERIVATIVES
    def computecost(self, f=None):
        """ Compute cost functional at f """
        if f == None:   f = self.g
        self.misfit = .5*np.linalg.norm(self.K.dot(f) - self.dn)**2
        self.reg = .5*(self.R.dot(f)).dot(f)
        self.cost = self.misfit + self.reg
        return self.cost

    def gradient(self, f):
        """ Compute M.g (discrete gradient) at a given point f """
        self.MG = self.K.T.dot(self.K.dot(f) - self.dn) + self.R.dot(f)

    def Hessian(self, f):
        """ Assemble Hessian at f """
        self.Hess = self.K.T.dot(self.K) + self.R
        
    ### TESTS
    def test_gradient(self, f=None, n=5):
        """ test gradient with FD approx around point f """
        if f == None:   f = self.f.copy()
        pm = [1.0, -1.0]
        eps = 1e-5
        self.gradient(f)
        for nn in xrange(1, n+1):
            df = np.sin(np.pi*nn*self.xx)
            cost = []
            for sign in pm:
                self.g = f + sign*eps*df
                cost.append(self.computecost(self.g))
            MGFD = (cost[0] - cost[1])/(2*eps)
            MGdf = self.MG.dot(df)
            print 'n={}:\tMGFD={:.5e}, MGdf={:.5e}, error={:.2e}'.format(\
            nn, MGFD, MGdf, np.abs(MGdf-MGFD)/np.abs(MGdf))

    def test_hessian(self, f=None, n=5):
        """ test Hessian with FD approx around point f """
        if f == None:   f = self.f.copy()
        pm = [1.0, -1.0]
        eps = 1e-5
        self.Hessian(f)
        for nn in xrange(1, n+1):
            df = np.sin(np.pi*nn*self.xx)
            MG = []
            for sign in pm:
                self.g = f + sign*eps*df
                self.gradient(self.g)
                MG.append(self.MG)
            HFD = (MG[0] - MG[1])/(2*eps)
            Hdf = self.Hess.dot(df)
            print 'n={}:\tHFD={:.5e}, Hdf={:.5e}, error={:.2e}'.format(\
            nn, np.linalg.norm(HFD), np.linalg.norm(Hdf), \
            np.linalg.norm(Hdf-HFD)/np.linalg.norm(Hdf))
        
    ### SOLVER
    def searchdirection(self):
        """ Compute search direction """
        self.gradient(self.g)
        self.Hessian(self.g)
        self.df = np.linalg.solve(self.Hess, -self.MG)
        assert self.df.dot(self.MG) < 0.0, "not a descent direction"
        print self.df.dot(self.MG)

    def linesearch(self):
        """ Perform exact line search """
        def costLS(alpha):  return self.computecost(self.g + alpha*self.df)
        self.alpha = optimize.fmin(costLS, 1.0)
#        costLS = np.vectorize(costLS)
#        alpha = np.linspace(0.0, 5.0, 1000)
#        fig2 = plt.figure()
#        ax = fig2.add_subplot(111)
#        ax.plot(alpha, costLS(alpha))
        self.g = self.g + self.alpha*self.df

    def solve(self):
        """ Solve image denoising pb """
        if self.regularization == 'tikhonov':
            self.Hessian(None)
            self.g = np.linalg.solve(self.Hess, self.K.T.dot(self.dn))
            self.computecost()
            self.alpha = 1.0


    ### OUTPUT
    def printout(self):
        """ Print results """
        self.medmisfit = np.linalg.norm(self.g-self.f)
        self.relmedmisfit = self.medmisfit/np.linalg.norm(self.f)
        print 'cost={:.2e}, misfit={:.2e}, reg={:.2e}, alpha={:.2e}, medmisfit={:.2e} ({:.3f})'.format(\
        self.cost, self.misfit, self.reg, self.alpha, self.medmisfit, self.relmedmisfit)

    def plot(self, u=None):
        """ Plot data and target """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax.plot(self.xx, self.dn, label='noisy data')
        ax.plot(self.xx, self.f, label='target')
        if not u == None:   
            ax.plot(self.xx, u, label='u')
        elif not self.g == None:   
            ax.plot(self.xx, self.g, label='sol')
        ax.legend(loc='best')
        return fig
