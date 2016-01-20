
from dolfin import sqrt, inner, nabla_grad, dx, \
Function, TestFunction, TrialFunction, assemble
from miscfenics import isFunction, isVector, setfct

#TODO: to be checked
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
        self.k = parameters['k']
        self.eps = parameters['eps']
        self.Vm = parameters['Vm']
        self.m = Function(self.Vm)
        self.test = TestFunction(self.Vm)
        self.trial = TrialFunction(self.Vm)
        #
        self.wkformcost = self.k * \
        sqrt(inner(nabla_grad(self.m), nabla_grad(self.m)) + self.eps)*dx
        #
        self.wkformgrad = inner(nabla_grad(self.m), nabla_grad(self.test)) * \
        self.k/sqrt(inner(nabla_grad(self.m), nabla_grad(self.m)) + self.eps)*dx
        #
        self.wkformhess = (inner(nabla_grad(self.test),nabla_grad(self.trial)) -\
        inner(nabla_grad(self.test),nabla_grad(self.m))*\
        inner(nabla_grad(self.m),nabla_grad(self.trial))/\
        (inner(nabla_grad(self.m), nabla_grad(self.m)) + self.eps)) * \
        self.k/sqrt(inner(nabla_grad(self.m), nabla_grad(self.m)) + self.eps)*dx


    def cost(self, m_in):
        """ returns the cost functional for self.m=m_in """
        isFunction(m_in)
        setfct(self.m, m_in)
        return assemble(self.wkformcost)

    def grad(self, m_in):
        """ returns the gradient (in vector format) evaluated at self.m=m_in """
        isFunction(m_in)
        setfct(self.m, m_in)
        return assemble(self.wkformgrad)

    def assemble_hessian(self, m_in=None):
        """ Assemble the Hessian of TV at m_in """
        if not m_in == None:    setfct(self.m, m_in)
        self.H = assemble(self.wkformhess)

    def hessian(self, mhat):
        """ returns the Hessian applied along a direction m_in """
        isVector(mhat)
        return self.H * mhat
