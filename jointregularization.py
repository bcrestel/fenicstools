"""
Define joint regularization terms
"""

from dolfin import inner, nabla_grad, dx, \
Function, TestFunction, TrialFunction, assemble, \
PETScKrylovSolver, assign, sqrt, Constant
from miscfenics import setfct


class Tikhonovab():
    """ Define Tikhonov regularization for a and b parameters """

    def __init__(self, parameters):
        """ parameters must contain:
        Vm = FunctionSpace
        gamma = reg param for Laplacian
        beta = reg param for mass matrix
        optional:
        m0 = reference state
        cg = reg param for cross-gradient
        """
        V = parameters['Vm']
        gamma = parameters['gamma']
        beta = parameters['beta']
        m0a, m0b = Function(V), Function(V)
        if parameters.has_key('m0'):
            m0 = parameters['m0']
            setfct(m0a, m0[0])
            setfct(m0b, m0[1])
        self.m0av = m0a.vector()
        self.m0bv = m0b.vector()
        VV = V*V
        test, trial = TestFunction(VV), TrialFunction(VV)
        K = assemble(inner(nabla_grad(trial), nabla_grad(test))*dx)
        M = assemble(inner(trial, test)*dx)
        self.R = gamma*K + beta*M
        if beta < 1e-10:
            self.Rprecond = gamma*K + 1e-10*M
        else:
            self.Rprecond = self.R
        self.precond = self.Rprecond
        self.a, self.b = Function(V), Function(V)
        self.ab = Function(VV)
        self.abv = self.ab.vector()
        if parameters.has_key('cg'):
            self.cg = crossgradient({'Vm':V})
            self.cgparam = parameters['cg']
        else:   self.cgparam = -1.0

    # For compatibility with Total Variation regularization
    def isTV(self): return False
    def isPD(self): return False

    def costab(self, ma_in, mb_in):
        """ ma_in, mb_in = Function(V) """
        setfct(self.a, ma_in)
        self.a.vector().axpy(-1.0, self.m0av)
        setfct(self.b, mb_in)
        self.b.vector().axpy(-1.0, self.m0bv)
        assign(self.ab.sub(0), self.a)
        assign(self.ab.sub(1), self.b)
        cost = 0.5 * self.abv.inner(self.R * self.abv)
        if self.cgparam > 0.0:
            cost += self.cgparam*self.cg.costab(ma_in, mb_in)
        return cost

    def gradab(self, ma_in, mb_in):
        """ ma_in, mb_in = Function(V) """
        setfct(self.a, ma_in)
        self.a.vector().axpy(-1.0, self.m0av)
        setfct(self.b, mb_in)
        self.b.vector().axpy(-1.0, self.m0bv)
        assign(self.ab.sub(0), self.a)
        assign(self.ab.sub(1), self.b)
        grad = self.R * self.abv
        if self.cgparam > 0.0:
            grad.axpy(self.cgparam, self.cg.gradab(ma_in, mb_in))
        return grad # this is local

    def assemble_hessianab(self, a, b):
        if self.cgparam > 0.0:
            self.cg.assemble_hessianab(a, b)
            self.precond = self.Rprecond + self.cgparam*self.cg.Hdiag
        else:
            pass

    def hessianab(self, ahat, bhat):
        """ ahat, bhat = Vector(V) """
        setfct(self.a, ahat)
        setfct(self.b, bhat)
        assign(self.ab.sub(0), self.a)
        assign(self.ab.sub(1), self.b)
        Hx = self.R * self.abv
        if self.cgparam > 0.0:
            Hx.axpy(self.cgparam, self.cg.hessianab(ahat, bhat))
        return Hx

    def getprecond(self):
        solver = PETScKrylovSolver("cg", "amg")
        solver.parameters["maximum_iterations"] = 1000
        solver.parameters["relative_tolerance"] = 1e-12
        solver.parameters["error_on_nonconvergence"] = True 
        solver.parameters["nonzero_initial_guess"] = False 
        """
        solver = PETScKrylovSolver("richardson", "amg")
        solver.parameters["maximum_iterations"] = 1
        solver.parameters["error_on_nonconvergence"] = False
        solver.parameters["nonzero_initial_guess"] = False
        """
        solver.set_operator(self.precond)
        return solver
        

class crossgradient():
    """ Define cross-gradient joint regularization """
    def __init__(self, parameters):
        V = parameters['Vm']
        VV = V*V
        self.ab = Function(VV)
        self.abv = self.ab.vector()
        self.MG = Function(VV)
        # cost
        self.a, self.b = Function(V), Function(V)
        self.cost = 0.5*( inner(nabla_grad(self.a), nabla_grad(self.a))*\
        inner(nabla_grad(self.b), nabla_grad(self.b))*dx - \
        inner(nabla_grad(self.a), nabla_grad(self.b))*\
        inner(nabla_grad(self.a), nabla_grad(self.b))*dx )
        # gradient
        testa, testb = TestFunction(V*V)
        grada = inner( nabla_grad(testa), \
        inner(nabla_grad(self.b), nabla_grad(self.b))*nabla_grad(self.a) - \
        inner(nabla_grad(self.a), nabla_grad(self.b))*nabla_grad(self.b) )*dx
        gradb = inner( nabla_grad(testb), \
        inner(nabla_grad(self.a), nabla_grad(self.a))*nabla_grad(self.b) - \
        inner(nabla_grad(self.a), nabla_grad(self.b))*nabla_grad(self.a) )*dx
        self.grad = grada + gradb
        # Hessian
        self.ahat, self.bhat = Function(V), Function(V)
        self.abhat = Function(V*V)
        at, bt = TestFunction(V*V)
        ah, bh = TrialFunction(V*V)
        wkform11 = inner( nabla_grad(at), \
        inner(nabla_grad(self.b), nabla_grad(self.b))*nabla_grad(ah) - \
        inner(nabla_grad(ah), nabla_grad(self.b))*nabla_grad(self.b) )*dx
        #
        wkform21 = inner( nabla_grad(bt), \
        2*inner(nabla_grad(self.a), nabla_grad(ah))*nabla_grad(self.b) - \
        inner(nabla_grad(self.a), nabla_grad(self.b))*nabla_grad(ah) - \
        inner(nabla_grad(ah), nabla_grad(self.b))*nabla_grad(self.a) )*dx
        #
        wkform12 = inner( nabla_grad(at), \
        2*inner(nabla_grad(self.b), nabla_grad(bh))*nabla_grad(self.a) - \
        inner(nabla_grad(self.a), nabla_grad(self.b))*nabla_grad(bh) - \
        inner(nabla_grad(self.a), nabla_grad(bh))*nabla_grad(self.b) )*dx
        #
        wkform22 = inner( nabla_grad(bt), \
        inner(nabla_grad(self.a), nabla_grad(self.a))*nabla_grad(bh) - \
        inner(nabla_grad(self.a), nabla_grad(bh))*nabla_grad(self.a) )*dx
        #
        self.hessian = wkform11 + wkform21 + wkform12 + wkform22
        self.precond = wkform11 + wkform22

    def costab(self, ma_in, mb_in):
        """ ma_in, mb_in = Function(V) """
        setfct(self.a, ma_in)
        setfct(self.b, mb_in)
        self.H = None
        return assemble(self.cost)  # this is global (in parallel)

    def gradab(self, ma_in, mb_in):
        """ ma_in, mb_in = Function(V) """
        setfct(self.a, ma_in)
        setfct(self.b, mb_in)
        self.H = None
        return assemble(self.grad)

    def assemble_hessianab(self, a, b):
        setfct(self.a, a)
        setfct(self.b, b)
        self.H = assemble(self.hessian)
        self.Hdiag = assemble(self.precond)

    def hessianab(self, ahat, bhat):
        """ ahat, bhat = Vector(V) """
        setfct(self.ahat, ahat)
        setfct(self.bhat, bhat)
        assign(self.abhat.sub(0), self.ahat)
        assign(self.abhat.sub(1), self.bhat)
        return self.H * self.abhat.vector()


#----------------------------------------------------------------------
#----------------------------------------------------------------------
#TODO: needs to be checked against finite-difference
class VTV():
    """ Define Vectorial Total Variation regularization for 2 parameters """

    def __init__(self, Vm, parameters=[]):
        """ Vm = FunctionSpace for the parameters (m1, m2) """

        self.parameters = {'k':1.0, 'eps':1e-2}

        self.m1 = Function(Vm)
        self.m2 = Function(Vm)
        self.m1h = Function(Vm)
        self.m2h = Function(Vm)
        self.m12h = Function(Vm*Vm)
        testm1, testm2 = TestFunction(Vm*Vm)
        trialm1, trialm2 = TrialFunction(Vm*Vm)

        # cost function
        normm1 = inner(nabla_grad(self.m1), nabla_grad(self.m1))
        normm2 = inner(nabla_grad(self.m2), nabla_grad(self.m2))
        TVnormsq = normm1 + normm2 + Constant(self.eps)
        TVnorm = sqrt(TVnormsq)
        self.wkformcost = self.k * TVnorm * dx

        # gradient
        gradm1 = self.k/TVnorm*inner(nabla_grad(self.m1), nabla_grad(testm1))*dx
        gradm2 = self.k/TVnorm*inner(nabla_grad(self.m2), nabla_grad(testm2))*dx
        self.gradm = gradm1 + gradm2

        # Hessian
        H11 = self.k/TVnorm*(inner(nabla_grad(trialm1), nabla_grad(testm1)) - \
        inner(nabla_grad(testm1),nabla_grad(self.m1))* \
        inner(nabla_grad(self.m1), nabla_grad(trialm1))/TVnormsq)*dx
        H12 = -self.k/(TVnorm*TVnormsq)*(inner(nabla_grad(testm1),nabla_grad(self.m1))* \
        inner(nabla_grad(self.m2), nabla_grad(trialm2)))*dx
        H21 = -self.k/(TVnorm*TVnormsq)*(inner(nabla_grad(testm2),nabla_grad(self.m2))* \
        inner(nabla_grad(self.m1), nabla_grad(trialm1)))*dx
        H22 = self.k/TVnorm*(inner(nabla_grad(trialm2), nabla_grad(testm2)) - \
        inner(nabla_grad(testm2),nabla_grad(self.m2))* \
        inner(nabla_grad(self.m2), nabla_grad(trialm2))/TVnormsq)*dx
        self.hessian = H11 + H12 + H21 + H22
        #TODO: need a preconditioner for Hessian


    def update(self, parameters=[]):
        """ Update parameters of regularization """

        self.parameters.update(parameters)
        self.k = self.parameters['k']
        self.eps = self.parameters['eps']


    def costab(self, m1, m2):
        """ Compute value of cost function at (m1,m2) """

        setfct(self.m1, m1)
        setfct(self.m2, m2)
        self.H = None
        return assemble(self.wkformcost)


    def gradab(self, m1, m2):
        """ returns gradient at (m1,m2) as a vector """

        setfct(self.m1, m1)
        setfct(self.m2, m2)
        self.H = None
        return assemble(self.gradm)


    def assemble_hessianab(self, m1, m2):
        """ Assemble Hessian matrix for regularization """

        setfct(self.m1, m1)
        setfct(self.m2, m2)
        self.H = assemble(self.hessian)


    def hessianab(self, m1h, m2h):
        """ m1h, m2h = Vector(V) """

        setfct(self.m1h, m1h)
        setfct(self.m2h, m2h)
        assign(self.m12h.sub(0), self.m1h)
        assign(self.m12h.sub(1), self.m2h)
        return self.H * self.m12h.vector()
