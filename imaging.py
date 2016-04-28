import sys
import os.path
import shutil
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import dolfin as dl

from miscfenics import setfct
from prior import LaplacianPrior
from regularization import TV
from plotfenics import PlotFenics


class ObjectiveImageDenoising():
    """
    Class to do image denoising
    """
    def __init__(self, mesh, trueImage, parameters=None):
        """
        Inputs:
            mesh = Fenics mesh
            trueImage = object from class Image
            parameters = dict
        """
        # Mesh
        self.mesh = mesh
        self.V = dl.FunctionSpace(self.mesh, 'Lagrange', 1)
        self.xx = self.V.dofmap().tabulate_all_coordinates(self.mesh)
        self.dimV = self.V.dim()
        self.test, self.trial = dl.TestFunction(self.V), dl.TrialFunction(self.V)
        self.f_true = dl.interpolate(trueImage, self.V)  
        self.g, self.dg, self.gtmp = dl.Function(self.V), dl.Function(self.V), dl.Function(self.V)
        self.Grad = dl.Function(self.V)
        self.Gradnorm0 = None
        # mass matrix
        self.Mweak = dl.inner(self.test, self.trial)*dl.dx
        self.M = dl.assemble(self.Mweak)
        self.solverM = dl.LUSolver('petsc')
        self.solverM.parameters['symmetric'] = True
        self.solverM.parameters['reuse_factorization'] = True
        self.solverM.set_operator(self.M)
        self.targetnorm = np.sqrt((self.M*self.f_true.vector()).inner(self.f_true.vector()))
        # line search parameters
        self.parameters = {'alpha0':1.0, 'rho':0.5, 'c':5e-5, 'max_backtrack':12}
        # regularization
        if not parameters == None: 
            self.parameters.update(parameters)
            self.define_regularization()
        self.regparam = 1.0
        # plots:
        filename, ext = os.path.splitext(sys.argv[0])
        if os.path.isdir(filename + '/'):   shutil.rmtree(filename + '/')
        self.myplot = PlotFenics(filename)

        
    def generatedata(self, noisepercent):
        """ compute data and add noisepercent (%) of noise """
        sigma = noisepercent*np.linalg.norm(self.f_true.vector().array())/np.sqrt(self.dimV)
        print 'sigma_noise = ', sigma
        np.random.seed(0)  #TODO: tmp
        eta = sigma*np.random.randn(self.dimV)
        self.dn = dl.Function(self.V)
        setfct(self.dn, eta)
        self.dn.vector().axpy(1.0, self.f_true.vector())
        print 'min(true)={}, max(true)={}'.format(\
        np.amin(self.f_true.vector().array()), \
        np.amax(self.f_true.vector().array()))
        print 'min(noisy)={}, max(noisy)={}'.format(\
        np.amin(self.dn.vector().array()), \
        np.amax(self.dn.vector().array()))

    def define_regularization(self, parameters=None):
        if not parameters == None:  self.parameters.update(parameters)
        regularization = self.parameters['regularization']
        if regularization == 'tikhonov':
            gamma = self.parameters['gamma']
            beta = self.parameters['beta']
            self.Reg = LaplacianPrior({'gamma':gamma, 'beta':beta, 'Vm':self.V})
            self.inexact = False
        elif regularization == 'TV':
            eps = self.parameters['eps']
            k = self.parameters['k']
            mode = self.parameters['mode']
            self.Reg = TV({'eps':eps, 'k':k, 'Vm':self.V, 'mode':mode})
            self.inexact = True


    ### COST and DERIVATIVES
    def computecost(self, f=None):
        """ Compute cost functional at f """
        if f == None:   f = self.g
        df = f.vector() - self.dn.vector()
        self.misfit = 0.5 * (self.M*df).inner(df)
        self.reg = self.Reg.cost(f)
        self.cost = self.misfit + self.regparam*self.reg
        return self.cost

    def gradient(self, f=None):
        """ Compute M.g (discrete gradient) at a given point f """
        if f == None:   f = self.g
        df = f.vector() - self.dn.vector()
        self.MGk = self.M*df
        self.MGr = self.Reg.grad(f)
        self.MG = self.MGk + self.MGr*self.regparam
        self.solverM.solve(self.Grad.vector(), self.MG)
        self.Gradnorm = np.sqrt((self.MG).inner(self.Grad.vector()))
        if self.Gradnorm0 == None:  self.Gradnorm0 = self.Gradnorm

    def Hessian(self, f=None):
        """ Assemble Hessian at f """
        if f == None:   f = self.g
        regularization = self.parameters['regularization']
        if regularization == 'TV':  
            self.Reg.assemble_hessian(f)
            self.Hess = self.M + self.Reg.H*self.regparam
        elif regularization == 'tikhonov':
            self.Hess = self.M + self.Reg.Minvprior*self.regparam
        
        
    ### SOLVER
    def searchdirection(self):
        """ Compute search direction """
        self.gradient()
        self.Hessian()
        solver = dl.PETScKrylovSolver("cg", "petsc_amg")
        solver.parameters['nonzero_initial_guess'] = False
        # Inexact CG:
        if self.inexact:
            self.cgtol = min(0.5, np.sqrt(self.Gradnorm/self.Gradnorm0))
        else:   self.cgtol = 1e-6
        solver.parameters['relative_tolerance'] = self.cgtol
        solver.set_operator(self.Hess)
        self.cgiter = solver.solve(self.dg.vector(), -1.0*self.MG)
        if (self.MG).inner(self.dg.vector()) > 0.0:    
            print "*** WARNING: NOT a descent direction"

    def linesearch(self):
        """ Perform inexact backtracking line search """
        regularization = self.parameters['regularization']
        mode = self.Reg.parameters['mode']
        # update direction + line search for dual variable (CGM version)
        if regularization == 'TV' and mode == 'primaldual': 
            self.Reg.update_wCGM(self.dg)
        # line search for primal variable
        self.alpha = self.parameters['alpha0']
        rho = self.parameters['rho']
        c = self.parameters['c']
        self.computecost()
        costref = self.cost
        cdJdf = ( (self.MG).inner(self.dg.vector()) )*c
        self.LS = False
        for ii in xrange(self.parameters['max_backtrack']):
            setfct(self.gtmp, self.g.vector() + self.dg.vector()*self.alpha) 
            if self.computecost(self.gtmp) < costref + self.alpha*cdJdf:
                self.g.vector().axpy(self.alpha, self.dg.vector())
                self.LS = True
                break
            else:   self.alpha *= rho
        # update direction + line search for dual variable (alt version)
        #if regularization == 'TV' and mode == 'primaldual': 
        #    self.Reg.update_walt(self.dg, self.alpha)

    def solve(self, plot=False):
        """ Solve image denoising pb """
        regularization = self.parameters['regularization']
        print '\t{:12s} {:12s} {:12s} {:12s} {:12s} {:12s} {:12s}\t{:12s} {:12s}'.format(\
        'a_reg', 'cost', 'misfit', 'reg', '||G||', 'a_LS', 'medmisfit', \
        'tol_cg', 'n_cg')
        #
        if regularization == 'tikhonov':
            self.searchdirection()
            self.g.vector().axpy(1.0, self.dg.vector())
            self.computecost()
            self.alpha = 1.0
            self.printout()
        elif regularization == 'TV':
            self.computecost()
            cost = self.cost
            # initial printout
            df = self.f_true.vector() - self.g.vector()
            self.medmisfit = np.sqrt((self.M*df).inner(df))
            self.relmedmisfit = self.medmisfit/self.targetnorm
            print ('{:12.1e} {:12.4e} {:12.4e} {:12.4e} {:12s} {:12s} {:12.2e}'\
            +' ({:.3f})').format(\
            self.regparam, self.cost, self.misfit, self.reg, '', '', \
            self.medmisfit**2, self.relmedmisfit)
            for ii in xrange(1000):
                self.searchdirection()
                self.linesearch()
                print ii+1,
                self.printout()
                # Check termination conditions:
                if not self.LS:
                    print 'Line search failed'
                    break
                if self.Gradnorm < min(1e-12, 1e-10*self.Gradnorm0) or \
                np.abs(cost-self.cost)/cost < 1e-12:
                    print 'optimization converged'
                    break
                cost = self.cost


    ### OUTPUT
    def printout(self):
        """ Print results """
        df = self.f_true.vector() - self.g.vector()
        self.medmisfit = np.sqrt((self.M*df).inner(df))
        self.relmedmisfit = self.medmisfit/self.targetnorm
        print ('{:12.1e} {:12.4e} {:12.4e} {:12.4e} {:12.4e} {:12.2e} {:12.2e}'\
        +' ({:.3f}) {:12.2e} {:6d}').format(\
        self.regparam, self.cost, self.misfit, self.reg, self.Gradnorm, self.alpha, \
        self.medmisfit**2, self.relmedmisfit, self.cgtol, self.cgiter)

    def plot(self, index=0, add=''):
        """ Plot target (w/ noise 0, or w/o noise 1) or current iterate (2) """
        if index == 0:
            self.myplot.set_varname('target' + add)
            self.myplot.plot_vtk(self.f_true)
        elif index == 1:
            self.myplot.set_varname('data' + add)
            self.myplot.plot_vtk(self.dn)
        elif index == 2:
            self.myplot.set_varname('solution' + add)
            self.myplot.plot_vtk(self.g)


    ### TESTS
    def test_gradient(self, f=None, n=5):
        """ test gradient with FD approx around point f """
        if f == None:   f = self.f_true
        pm = [1.0, -1.0]
        eps = 1e-5
        self.gradient(f)
        for nn in xrange(1, n+1):
            expr = dl.Expression('sin(n*pi*x[0])*sin(n*pi*x[1])', n=nn)
            df = dl.interpolate(expr, self.V)
            MGdf = self.MG.inner(df.vector())
            cost = []
            for sign in pm:
                setfct(self.g, f)
                self.g.vector().axpy(sign*eps, df.vector())
                cost.append(self.computecost(self.g))
            MGFD = (cost[0] - cost[1])/(2*eps)
            print 'n={}:\tMGFD={:.5e}, MGdf={:.5e}, error={:.2e}'.format(\
            nn, MGFD, MGdf, np.abs(MGdf-MGFD)/np.abs(MGdf))

    def test_hessian(self, f=None, n=5):
        """ test Hessian with FD approx around point f """
        if f == None:   f = self.f_true
        pm = [1.0, -1.0]
        eps = 1e-5
        self.Hessian(f)
        for nn in xrange(1, n+1):
            expr = dl.Expression('sin(n*pi*x[0])*sin(n*pi*x[1])', n=nn)
            df = dl.interpolate(expr, self.V)
            Hdf = (self.Hess*df.vector()).array()
            MG = []
            for sign in pm:
                setfct(self.g, f)
                self.g.vector().axpy(sign*eps, df.vector())
                self.gradient(self.g)
                MG.append(self.MG.array())
            HFD = (MG[0] - MG[1])/(2*eps)
            print 'n={}:\tHFD={:.5e}, Hdf={:.5e}, error={:.2e}'.format(\
            nn, np.linalg.norm(HFD), np.linalg.norm(Hdf), \
            np.linalg.norm(Hdf-HFD)/np.linalg.norm(Hdf))



##----------------------------------------------------------------------------------------
##----------------------------------------------------------------------------------------
##----------------------------------------------------------------------------------------
class ObjectiveImageDeblurring1D():
    """
    Class for linear 1D image denoising problem, built on an integral (blurring) kernel
    """

    def __init__(self, mesh, k, regularization='tikhonov'):
        """
        Inputs:
            pbtype = 'denoising' or 'deblurring'
            mesh = Fenics mesh
            k = Fenics Expression of the blurring kernel; must have parameter t
            f = target image
        """
        self.mesh = mesh
        self.V = dl.FunctionSpace(self.mesh, 'Lagrange', 1)
        self.dimV = self.V.dim()
        self.xx = self.V.dofmap().tabulate_all_coordinates(self.mesh)
        self.test, self.trial = dl.TestFunction(self.V), dl.TrialFunction(self.V)
        # Target data:
        self.f_true = 0.75*(self.xx>=.1)*(self.xx<=.25)
        self.f_true += (self.xx>=0.28)*(self.xx<=0.3)*(15*self.xx-15*0.28)
        self.f_true += (self.xx>0.3)*(self.xx<0.33)*0.3
        self.f_true += (self.xx>=0.33)*(self.xx<=0.35)*(-15*self.xx+15*0.35)
        self.f_true += (self.xx>=.4)*(self.xx<=.9)*(self.xx-.4)**2*(self.xx-0.9)**2/.25**4
        self.g = None   # current iterate
        # kernel operator
        self.k = k
        self.Kweak = dl.inner(self.k, self.test)*dl.dx
        self.assembleK()
        # mass matrix
        self.Mweak = dl.inner(self.test, self.trial)*dl.dx
        self.M = dl.assemble(self.Mweak)
        # regularization
        self.parameters['regularization'] = regularization
        if regularization == 'tikhonov':
            self.RegTikh = LaplacianPrior({'gamma':1.0, 'beta':0.0, 'Vm':self.V})
            self.R = self.RegTikh.Minvprior.array()
        elif regularization == 'TV':
            self.RegTV = TV({'eps':1e-2, 'Vm':self.V})
        # line search parameters
        self.parameters['alpha0'] = 1.0
        self.parameters['rho'] = 0.5
        self.parameters['c'] = 5e-5
        self.parameters['max_backtrack'] = 12

    def assembleK(self):
        self.K = np.zeros((self.dimV, self.dimV))
        for ii, tt in enumerate(self.xx):
            self.k.t = tt
            self.K[ii,:] = dl.assemble(self.Kweak).array()
        
    def generatedata(self, noisepercent):
        """ compute data and add noisepercent (%) of noise """
        pbtype = self.parameters['pbtype']
        self.d = self.K.dot(self.f_true)
        sigma = noisepercent*np.linalg.norm(self.d)/np.sqrt(self.dimV)
        eta = sigma*np.random.randn(self.dimV)
        print 'noise residual={}'.format(.5*np.linalg.norm(eta)**2)
        self.dn = self.d + eta

    def update_reg(self, gamma):
        regularization = self.parameters['regularization']
        if regularization == 'tikhonov':
            self.gamma = gamma
            self.R = self.RegTikh.Minvprior.array()*self.gamma
        elif regularization == 'TV':
            self.RegTV.update({'k':gamma})


    ### COST and DERIVATIVES
    def computecost(self, f=None):
        """ Compute cost functional at f """
        regularization = self.parameters['regularization']
        if f == None:   f = self.g
        #
        if pbtype == 'deblurring':  
            self.misfit = .5*np.linalg.norm(self.K.dot(f)-self.dn)**2
        if regularization == 'tikhonov':   self.reg = .5*(self.R.dot(f)).dot(f)
        elif regularization == 'TV':   self.reg = self.RegTV.cost(f)
        self.cost = self.misfit + self.reg
        return self.cost

    def gradient(self, f=None):
        """ Compute M.g (discrete gradient) at a given point f """
        regularization = self.parameters['regularization']
        if f == None:   f = self.g
        #
        self.MGk = self.K.T.dot(self.K.dot(f) - self.dn) 
        if regularization == 'tikhonov':   self.MGr = self.R.dot(f)
        elif regularization == 'TV':   self.MGr = self.RegTV.grad(f).array()
        self.MG = self.MGk + self.MGr

    def Hessian(self, f=None):
        """ Assemble Hessian at f """
        regularization = self.parameters['regularization']
        if f == None:   f = self.g
        #
        self.Hessk = self.K.T.dot(self.K) 
        if regularization == 'tikhonov':   self.Hessr = self.R
        elif regularization == 'TV':
            self.RegTV.assemble_hessian(f)
            self.Hessr = self.RegTV.H.array()
        self.Hess = self.Hessk + self.Hessr
        
        
    ### SOLVER
    def searchdirection(self):
        """ Compute search direction """
        self.gradient()
        self.Hessian()
        self.df = np.linalg.solve(self.Hess, -self.MG)
        assert self.df.dot(self.MG) < 0.0, "not a descent direction"

    def linesearch(self):
        """ Perform inexact backtracking line search """
        self.alpha = self.parameters['alpha0']
        rho = self.parameters['rho']
        c = self.parameters['c']
        costref = self.cost
        cdJdf = c*self.MG.dot(self.df)
        self.LS = False
        for ii in xrange(self.parameters['max_backtrack']):
            if self.computecost(self.g + self.alpha*self.df) < \
            costref + self.alpha*cdJdf:
                self.g = self.g + self.alpha*self.df
                self.LS = True
                break
            else:
                self.alpha *= rho

    def solve(self, plot=False):
        """ Solve image denoising pb """
        regularization = self.parameters['regularization']
        #
        if regularization == 'tikhonov':
            self.Hessian(None)
            self.g = np.linalg.solve(self.Hess, self.K.T.dot(self.dn))
            self.computecost()
            self.alpha = 1.0
        elif regularization == 'TV':
            self.computecost()
            self.alpha = 1.0
            self.printout()
            cost = self.cost
            self.COST = [cost]
            self.MEDMIS = [self.relmedmisfit]
            for ii in xrange(500):
                self.searchdirection()
                self.linesearch(1.0)
                print ii,
                self.printout()
                if plot and ii%50 == 0:
                    self.plot()
                    plt.show()
                if not self.LS:
                    print 'Line search failed'
                    break
                if np.abs(cost - self.cost)/cost < 1e-10:
                    print 'optimization converged'
                    break
                cost = self.cost
                self.COST.append(cost)
                self.MEDMIS.append(self.relmedmisfit)


    ### TESTS
    def test_gradient(self, f=None, n=5):
        """ test gradient with FD approx around point f """
        if f == None:   f = self.f_true.copy()
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
        if f == None:   f = self.f_true.copy()
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


    ### OUTPUT
    def printout(self):
        """ Print results """
        self.medmisfit = np.linalg.norm(self.g-self.f_true)
        self.relmedmisfit = self.medmisfit/np.linalg.norm(self.f_true)
        print 'cost={:.2e}, misfit={:.2e}, reg={:.2e}, alpha={:.2e}, medmisfit={:.2e} ({:.3f})'.format(\
        self.cost, self.misfit, self.reg, self.alpha, self.medmisfit, self.relmedmisfit)

    def plot(self, u=None):
        """ Plot data and target """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax.plot(self.xx, self.dn, label='noisy data')
        ax.plot(self.xx, self.f_true, label='target')
        if not u == None:   
            ax.plot(self.xx, u, label='u')
        elif not self.g == None:   
            ax.plot(self.xx, self.g, label='sol')
        ax.legend(loc='best')
        return fig





