import sys
import numpy as np

from dolfin import sqrt, inner, nabla_grad, grad, dx, \
Function, TestFunction, TrialFunction, assemble, solve, \
Constant, plot, interactive
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
        self.parameters = {'k':1.0, 'eps':1e-2, 'mode':'primaldual'}
        if parameters.has_key('Vm'):
            self.parameters.update(parameters)
            self.update()
        else:   
            print "inputs parameters must contain field 'Vm'"
            sys.exit(1)

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
        self.updatew = True
        # udpate parameters
        if parameters == None:  
            parameters = self.parameters
        else:
            self.parameters.update(parameters)
        mode = self.parameters['mode']
        self.Vm = parameters['Vm']
        eps = self.parameters['eps']
        self.k = self.parameters['k']
        # define functions
        self.m = Function(self.Vm)
        self.dm = Function(self.Vm)
        self.test, self.trial = TestFunction(self.Vm), TrialFunction(self.Vm)
        # frequently-used variable
        self.fTV = inner(nabla_grad(self.m), nabla_grad(self.m)) + Constant(eps)
        self.kovsq = self.k / sqrt(self.fTV)
        # primal dual variables
        self.w = Function(self.Vm*self.Vm)  # dual variable for primal-dual, initialized at 0
        if mode == 'primaldual':
            self.dw = Function(self.Vm*self.Vm)  
            self.testw = TestFunction(self.Vm*self.Vm)
            self.trialw = TrialFunction(self.Vm*self.Vm)
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
        self.wkformPDhess = self.kovsq * ( \
        inner(nabla_grad(self.trial), nabla_grad(self.test)) - \
        0.5*( inner(self.w, nabla_grad(self.test))*\
        inner(nabla_grad(self.trial), nabla_grad(self.m)) + \
        inner(nabla_grad(self.m), nabla_grad(self.test))*\
        inner(nabla_grad(self.trial), self.w) ) / sqrt(self.fTV) \
        )*dx
        if mode == 'GNhessian': self.wkformhess = self.wkformGNhess
        elif mode == 'primaldual':  self.wkformhess = self.wkformPDhess
        else:   self.wkformhess = self.wkformFhess
        # update dual variable
        if mode == 'primaldual':
            self.LSrhow = 0.9
            self.wkformdwA = inner(sqrt(self.fTV)*self.trialw, self.testw)*dx
            self.wkformdwrhs = \
            - inner(self.w*sqrt(self.fTV)-nabla_grad(self.m),self.testw)*dx +\
            inner(nabla_grad(self.dm)-inner(nabla_grad(self.m),nabla_grad(self.dm))\
            *self.w/sqrt(self.fTV),self.testw)*dx

    def cost(self, m_in):
        """ returns the cost functional for self.m=m_in """
        setfct(self.m, m_in)
        self.H = None
        return assemble(self.wkformcost)

    def grad(self, m_in):
        """ returns the gradient (in vector format) evaluated at self.m=m_in """
        setfct(self.m, m_in)
        self.H = None
        return assemble(self.wkformgrad)

    def assemble_hessian(self, m_in):
        """ Assemble the Hessian of TV at m_in """
        mode = self.parameters['mode']
        if self.updatew == True:
            setfct(self.m, m_in)
            self.H = assemble(self.wkformhess)
            if mode == 'primaldual':    self.updatew = False
        else:
            print 'You need to update dual variable w'
            sys.exit(1)

    def assemble_GNhessian(self, m_in):
        """ Assemble the Gauss-Newton Hessian at m_in """
        setfct(self.m, m_in)
        self.H = assemble(self.wkformGNhess)

    def hessian(self, mhat):
        """ returns the Hessian applied along a direction mhat """
        isVector(mhat)
        return self.H * mhat

    def update_w(self, dm, alpha=None):
        """ Compute dw and run line search on w """
        # check
        mode = self.parameters['mode']
        if not mode == 'primaldual':    sys.exit(1)
        # compute dw
        setfct(self.dm, dm)
        A = assemble(self.wkformdwA)
        b = assemble(self.wkformdwrhs)
        solve(A, self.dw.vector(), b)
        # line search for dual variable:
        if alpha == None:
            # Compute max step length that can be taken
            (wx,wy) = self.w.split(deepcopy=True)
            (dwx,dwy) = self.dw.split(deepcopy=True)
            wxa, wya = wx.vector().array(), wy.vector().array()
            dwxa, dwya = dwx.vector().array(), dwy.vector().array()
            wTdw = wxa*dwxa + wya*dwya
            normw2 = wxa**2 + wya**2
            normdw2 = dwxa**2 + dwya**2
            # Check we don't have awkward situation
            Delta = wTdw**2 + normdw2*(1.0-normw2)
            assert len(np.where(Delta < 1e-14)[0]) == 0
            # then compute max alpha
            ALPHAS = (np.sqrt(Delta) - wTdw)/normdw2
            alpha = np.amin(ALPHAS)
            self.w.vector().axpy(self.LSrhow*alpha, self.dw.vector())
        else:
            self.w.vector().axpy(alpha, self.dw.vector())
        print 'line search dual variable: alpha={}, max(|w_i|)={}'.\
        format(alpha, np.amax(np.sqrt(normw2)))
        self.updatew = True
        # Tmp check
        #assert np.amax( np.abs(self.w.vector().array()) ) <= 1.0
