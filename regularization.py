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
        if parameters.has_key('eps') and parameters.has_key('Vm'):
            if not parameters.has_key('k'):
                parameters['k'] = 1.0
            # GN Hessian?
            if parameters.has_key('GNhessian'): self.GNhessian = parameters['GNhessian']
            else:   self.GNhessian = True
            # Primal-dual method when using full Hessian
            if self.GNhessian:  self.primaldual = False
            else:   self.primaldual = True
            self.update(parameters)
        else:   
            print "inputs parameters must contain field 'eps' and 'Vm'"
            sys.exit(1)

    def update(self, parameters):
        """ Update the parameters.
        parameters should be:
            - k(x) = factor inside TV
            - eps = regularization parameter
            - Vm = FunctionSpace for parameter. 
        ||f||_TV = int k(x) sqrt{|grad f|^2 + eps} dx
        """
        self.H = None
        self.updatew = True
        if parameters.has_key('k'): self.k = parameters['k']
        if parameters.has_key('eps'): self.eps = parameters['eps']
        if parameters.has_key('Vm'):
            self.Vm = parameters['Vm']
            self.m = Function(self.Vm)
            self.w = Function(self.Vm*self.Vm)  # dual variable for primal-dual, initialized at 0
            self.dw = Function(self.Vm*self.Vm)  
            self.dm = Function(self.Vm)
            self.test, self.trial = TestFunction(self.Vm), TrialFunction(self.Vm)
            if self.primaldual: 
                self.testw = TestFunction(self.Vm*self.Vm)
                self.trialw = TrialFunction(self.Vm*self.Vm)
        self.fTV = inner(nabla_grad(self.m), nabla_grad(self.m)) + self.eps
        self.kovsq = self.k / sqrt(self.fTV)
        if not self.primaldual: 
            self.w = nabla_grad(self.m)/sqrt(self.fTV)
        #
        # cost functional
        self.wkformcost = self.k*sqrt(self.fTV)*dx
        # gradient
        self.wkformgrad = self.kovsq*inner(nabla_grad(self.m), nabla_grad(self.test))*dx
        # Hessian
        self.wkformGNhess = self.kovsq*inner(nabla_grad(self.trial), nabla_grad(self.test))*dx
        if self.GNhessian:
            self.wkformhess = self.wkformGNhess 
        else:
            if self.primaldual:
                self.wkformhess = self.kovsq * ( \
                inner(nabla_grad(self.trial), nabla_grad(self.test)) - \
                0.5*( inner(self.w, nabla_grad(self.test))*\
                inner(nabla_grad(self.trial), nabla_grad(self.m)) + \
                inner(nabla_grad(self.m), nabla_grad(self.test))*\
                inner(nabla_grad(self.trial), self.w) ) / sqrt(self.fTV) \
                )*dx
            else:
                self.wkformhess = self.kovsq*( \
                inner(nabla_grad(self.trial), nabla_grad(self.test)) - \
                inner(nabla_grad(self.m), nabla_grad(self.test))*\
                inner(nabla_grad(self.trial), nabla_grad(self.m))/self.fTV )*dx
        # Update dual variable
        if self.primaldual:
            self.LSrhow = 0.95
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
        if self.updatew == True:
            setfct(self.m, m_in)
            self.H = assemble(self.wkformhess)
            if self.primaldual == True: self.updatew = False
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
