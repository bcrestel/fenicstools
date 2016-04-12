import sys

from dolfin import sqrt, inner, nabla_grad, grad, dx, \
Function, TestFunction, TrialFunction, assemble
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
            if parameters.has_key('GNhessian'): self.GNhessian = parameters['GNhessian']
            else:   self.GNhessian = True
            self.primaldual = False
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
        if parameters.has_key('k'): self.k = parameters['k']
        if parameters.has_key('eps'): self.eps = parameters['eps']
        if parameters.has_key('Vm'):
            self.Vm = parameters['Vm']
            self.m = Function(self.Vm)
            self.w = Function(self.Vm)  # dual variable for primal-dual
            self.dw = Function(self.Vm)  
            self.dm = Function(self.Vm)
            self.test = TestFunction(self.Vm)
            self.trial = TrialFunction(self.Vm)
        self.fTV = inner(nabla_grad(self.m), nabla_grad(self.m)) + self.eps
        self.kovsq = self.k / sqrt(self.fTV)
        if not self.primaldual: self.w = nabla_grad(self.m)/sqrt(self.fTV)
        #
        # cost functional
        self.wkformcost = self.k * sqrt(self.fTV)*dx
        # gradient
        self.wkformgrad = self.k*inner(self.w, nabla_grad(self.test))*dx
        #self.wkformgrad = self.kovsq*inner(nabla_grad(self.m), nabla_grad(self.test))*dx
        # Hessian
        if self.GNhessian:
            self.wkformhess = self.kovsq*inner(nabla_grad(self.trial), nabla_grad(self.test))*dx
        else:
            self.wkformhess = self.kovsq * ( \
            inner(nabla_grad(self.trial), nabla_grad(self.test)) - \
            0.5*( inner(self.w, nabla_grad(self.test))*\
            inner(nabla_grad(self.trial), nabla_grad(self.m)) + \
            inner(nabla_grad(self.m), nabla_grad(self.test))*\
            inner(nabla_grad(self.trial), self.w) ) / sqrt(self.fTV) \
            )*dx
#            self.wkformhess = self.kovsq * ( \
#            inner(nabla_grad(self.trial), nabla_grad(self.test)) - \
#            inner(nabla_grad(self.m), nabla_grad(self.test))* \
#            inner(nabla_grad(self.trial), nabla_grad(self.m))/self.fTV)*dx
        #TODO: Check here
        #if self.primaldual:
        #    self.wkformdw-A = inner(sqrt(self.fTV)*self.dw, self.test)*dx
        #    self.wkformdw-rhs = inner(self.w*sqrt(self.fTV)-nabla_grad(self.m), self.test)*dx + \
        #    inner(nabla_grad(self.dm)-inner(nabla_grad(self.m),nabla_grad(self.dm))*self.w \
        #    /sqrt(self.fTV), self.test)*dx

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
        setfct(self.m, m_in)
        self.H = assemble(self.wkformhess)

    def hessian(self, mhat):
        """ returns the Hessian applied along a direction m_in """
        isVector(mhat)
        return self.H * mhat

    def update_w(self, dm):
        """ Compute dw and run line search on w """
        setfct(self.dm, dm)
        solve(self.wkformdw-A == self.wkformdw-rhs, self.dw)
        #TODO: code line search for w + alpha*dw
