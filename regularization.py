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
            if not self.GNhessian:  self.w = nabla_grad(self.m)/sqrt(self.fTV)
        #
        # cost functional
        self.wkformcost = self.k * sqrt(self.fTV)*dx
        # gradient
        self.wkformgrad = self.k*inner(self.w, nabla_grad(self.test))*dx
        # Hessian
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
        if self.primaldual:
            self.LSrhow = 0.95
            self.wkformdwA = inner(sqrt(self.fTV)*self.trialw, self.testw)*dx
            self.wkformdwrhs = inner(self.w*sqrt(self.fTV)-nabla_grad(self.m), self.testw)*dx + \
            inner(nabla_grad(self.dm)-inner(nabla_grad(self.m),nabla_grad(self.dm))*self.w \
            /sqrt(self.fTV), self.testw)*dx

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

    def hessian(self, mhat):
        """ returns the Hessian applied along a direction mhat """
        isVector(mhat)
        return self.H * mhat

    def update_w(self, dm):
        """ Compute dw and run line search on w """
        setfct(self.dm, dm)
        solve(self.wkformdwA == self.wkformdwrhs, self.dw)
        # line search for dual variable:
        dw = self.dw.vector().array()
        aa = (np.sign(dw) - self.w.vector().array())/dw
        alpha = np.amin(aa)
        self.w.vector().axpy(self.LSrhow*alpha, self.dw.vector())
        print 'line search dual variable: alpha={}, max |w_i|={}'.\
        format(alpha, np.amax(np.abs(self.w.vector().array())))
        self.updatew = True
        # Tmp check
        assert np.amax( np.abs(self.w.vector().array()) ) <= 1.0
