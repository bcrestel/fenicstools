import sys
import numpy as np

from dolfin import sqrt, inner, nabla_grad, grad, dx, \
Function, TestFunction, TrialFunction, assemble, solve, \
Constant, plot, interactive, assign
from miscfenics import isFunction, isVector, setfct

class TV():
    """
    Define Total Variation regularization
    """

    def __init__(self, parameters):
        """
        parameters should be:
            - k(x) = factor inside TV
            - eps = regularization parameter
        ||f||_TV = int k(x) sqrt{|grad f|^2 + eps} dx
        """
        self.parameters = {'k':1.0, 'eps':1e-2, 'GNhessian':False} # default parameters
        if parameters.has_key('Vm'):
            self.parameters.update(parameters)
            self.update()
        else:   
            print "inputs parameters must contain field 'Vm'"
            sys.exit(1)

    def isPD(self): return False

    def update(self, parameters=None):
        """ Update the parameters.
        parameters should be:
            - k(x) = factor inside TV
            - eps = regularization parameter
            - Vm = FunctionSpace for parameter. 
        ||f||_TV = int k(x) sqrt{|grad f|^2 + eps} dx
        """
        # reset some variables
        self.H = None
        # udpate parameters
        if parameters == None:  
            parameters = self.parameters
        else:
            self.parameters.update(parameters)
        GN = self.parameters['GNhessian']
        self.Vm = self.parameters['Vm']
        eps = self.parameters['eps']
        self.k = self.parameters['k']
        # define functions
        self.m = Function(self.Vm)
        self.test, self.trial = TestFunction(self.Vm), TrialFunction(self.Vm)
        # frequently-used variable
        self.fTV = inner(nabla_grad(self.m), nabla_grad(self.m)) + Constant(eps)
        self.kovsq = self.k / sqrt(self.fTV)
        #
        # cost functional
        self.wkformcost = self.k*sqrt(self.fTV)*dx
        # gradient
        self.wkformgrad = self.kovsq*inner(nabla_grad(self.m), nabla_grad(self.test))*dx
        # Hessian
        self.wkformGNhess = self.kovsq*inner(nabla_grad(self.trial), nabla_grad(self.test))*dx
        self.wkformFhess = self.kovsq*( \
        inner(nabla_grad(self.trial), nabla_grad(self.test)) - \
        inner(nabla_grad(self.m), nabla_grad(self.test))*\
        inner(nabla_grad(self.trial), nabla_grad(self.m))/self.fTV )*dx
        if self.isPD(): 
            self.updatePD()
            self.wkformhess = self.wkformPDhess
        else:
            if GN: self.wkformhess = self.wkformGNhess
            else:   self.wkformhess = self.wkformFhess

    def cost(self, m_in):
        """ returns the cost functional for self.m=m_in """
        setfct(self.m, m_in)
        self.H = None
        return assemble(self.wkformcost)

    def grad(self, m_in):
        """ returns the gradient (in vector format) evaluated at self.m = m_in """
        setfct(self.m, m_in)
        self.H = None
        return assemble(self.wkformgrad)

    def assemble_hessian(self, m_in):
        """ Assemble the Hessian of TV at m_in """
        setfct(self.m, m_in)
        self.H = assemble(self.wkformhess)

    def assemble_GNhessian(self, m_in):
        """ Assemble the Gauss-Newton Hessian at m_in """
        setfct(self.m, m_in)
        self.H = assemble(self.wkformGNhess)

    def hessian(self, mhat):
        """ returns the Hessian applied along a direction mhat """
        isVector(mhat)
        return self.H * mhat


#----------------------------------------------------------------------
#----------------------------------------------------------------------
class TVPD(TV):
    """ Total variation using primal-dual Newton """

    def isPD(self): return True

    def updatePD(self):
        """ Update the parameters.
        parameters should be:
            - k(x) = factor inside TV
            - eps = regularization parameter
            - Vm = FunctionSpace for parameter. 
        ||f||_TV = int k(x) sqrt{|grad f|^2 + eps} dx
        """
        # primal dual variables
        #TODO: dual variable should be piecewise constant (DG0)
        self.wH = Function(self.Vm*self.Vm)  # dual variable used in Hessian (re-scaled)
        #self.wH = nabla_grad(self.m)/sqrt(self.fTV) # full Hessian
        self.w = Function(self.Vm*self.Vm)  # dual variable for primal-dual, initialized at 0
        self.dm = Function(self.Vm)
        self.dw = Function(self.Vm*self.Vm)  
        self.testw = TestFunction(self.Vm*self.Vm)
        self.trialw = TrialFunction(self.Vm*self.Vm)
        # investigate convergence of dual variable
        self.dualres = self.w - nabla_grad(self.m)/sqrt(self.fTV)
        self.dualresnorm = inner(self.dualres, self.dualres)*dx
        self.normgraddm = inner(nabla_grad(self.dm), nabla_grad(self.dm))*dx
        # Hessian
        self.wkformPDhess = self.kovsq * ( \
        inner(nabla_grad(self.trial), nabla_grad(self.test)) - \
        0.5*( inner(self.wH, nabla_grad(self.test))*\
        inner(nabla_grad(self.trial), nabla_grad(self.m)) + \
        inner(nabla_grad(self.m), nabla_grad(self.test))*\
        inner(nabla_grad(self.trial), self.wH) ) / sqrt(self.fTV) \
        )*dx
        # update dual variable
        self.Mw = assemble(inner(self.trialw, self.testw)*dx)
        self.rhswwk = inner(-self.w, self.testw)*dx + \
        inner(nabla_grad(self.m)+nabla_grad(self.dm), self.testw) \
        /sqrt(self.fTV)*dx + \
        inner(-inner(nabla_grad(self.m),nabla_grad(self.dm))* \
        self.wH/sqrt(self.fTV), self.testw)*dx

    def compute_dw(self, dm):
        """ Compute dw """
        setfct(self.dm, dm)
        b = assemble(self.rhswwk)
        solve(self.Mw, self.dw.vector(), b)


    def update_w(self, alpha):
        """ update w and re-scale wH """
        self.w.vector().axpy(alpha, self.dw.vector())
        # project each w (coord-wise) onto unit sphere to get wH
        (wx,wy) = self.w.split(deepcopy=True)
        wxa, wya = wx.vector().array(), wy.vector().array()
        normw = np.sqrt(wxa**2 + wya**2)
        factorw = [max(1.0, ii) for ii in normw]
        setfct(wx, wxa/factorw)
        setfct(wy, wya/factorw)
        assign(self.wH.sub(0), wx)
        assign(self.wH.sub(1), wy)
        # check
        (wx,wy) = self.wH.split(deepcopy=True)
        wxa, wya = wx.vector().array(), wy.vector().array()
        normw = np.sqrt(wxa**2 + wya**2)
        assert np.amax(normw) <= 1.0 + 1e-14
        # Print results
        dualresnorm = assemble(self.dualresnorm)
        normgraddm = assemble(self.normgraddm)
        print 'line search dual variable: max(fw)={}, ||(w-df/|df+e|)||={}, ||grad(dm)||={}'.\
        format(np.amax(factorw), np.sqrt(dualresnorm), np.sqrt(normgraddm))
