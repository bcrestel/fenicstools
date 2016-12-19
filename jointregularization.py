"""
Define joint regularization terms
"""

from dolfin import inner, nabla_grad, dx, \
Function, TestFunction, TrialFunction, assemble, \
PETScKrylovSolver, assign, sqrt, Constant
from miscfenics import setfct
from linalg.splitandassign import BlockDiagonal



class SumRegularization():
    """ Sum of independent regularizations for each med. param, 
    potentially connected by cross-gradient """

    def __init__(self, regul1, regul2, mpicomm, coeff_cg=0.0):
        """ regul1, regul2 = regularization/prior objects for medium 1 and 2
        coeff_cg = regularization constant for cross-gradient term; if 0.0, no
        cross-gradient (independent reconstructions) """

        assert id(regul1) != id(regul2), "Need to define two distinct regul objects"

        self.regul1 = regul1
        self.regul2 = regul2
        self.coeff_cg = coeff_cg

        V1 = self.regul1.Vm
        V2 = self.regul2.Vm
        self.V1V2 = V1*V2
        self.a, self.b = Function(V1), Function(V2)
        self.bd = BlockDiagonal(V1, V2, mpicomm)

        if self.coeff_cg > 0.0:
            self.crossgrad = crossgradient(self.V1V2)

        try:
            solver = PETScKrylovSolver('cg', 'ml_amg')
            self.amgprecond = 'ml_amg'
        except:
            self.amgprecond = 'petsc_amg'


    def isTV(self):
        return self.regul1.isTV() * self.regul2.isTV()
    def isPD(self):
        return self.regul1.isPD() * self.regul2.isPD()


    def costab(self, m1, m2):
        cost = self.regul1.cost(m1) + self.regul2.cost(m2)
        if self.coeff_cg > 0.0:
            cost += self.coeff_cg*self.crossgrad.costab(m1, m2)
        return cost

    def costabvect(self, m1, m2):
        cost = self.regul1.costvect(m1) + self.regul2.costvect(m2)
        if self.coeff_cg > 0.0:
            cost += self.coeff_cg*self.crossgrad.costab(m1, m2)
        return cost


    def gradab(self, m1, m2):
        self.a.vector().zero()
        self.b.vector().zero()
        grad = Function(self.V1V2)

        setfct(self.a, self.regul1.grad(m1))
        setfct(self.b, self.regul2.grad(m2))
        assign(grad.sub(0), self.a)
        assign(grad.sub(1), self.b)
        if self.coeff_cg > 0.0:
            grad.vector().axpy(self.coeff_cg, self.crossgrad.gradab(m1, m2))

        return grad.vector()

    def gradabvect(self, m1, m2):
        """ relies on gradvect metod from regularization instead of grad
        gradvect takes a Vector() as input argument """
        self.a.vector().zero()
        self.b.vector().zero()
        grad = Function(self.V1V2)

        setfct(self.a, self.regul1.gradvect(m1))
        setfct(self.b, self.regul2.gradvect(m2))
        assign(grad.sub(0), self.a)
        assign(grad.sub(1), self.b)
        if self.coeff_cg > 0.0:
            grad.vector().axpy(self.coeff_cg, self.crossgrad.gradab(m1, m2))

        return grad.vector()


    def _blockdiagprecond(self):
        """ assemble a block-diagonal preconditioner
        with preconditioners for each regularization term """
        R1 = self.regul1.precond
        R2 = self.regul2.precond
        return self.bd.assemble(R1, R2)

    def assemble_hessianab(self, m1, m2):
        self.regul1.assemble_hessian(m1)
        self.regul2.assemble_hessian(m2)
        if self.coeff_cg > 0.0:
            self.crossgrad.assemble_hessianab(m1, m2)
            self.precond = self._blockdiagprecond() \
            + self.crossgrad.Hdiag*self.coeff_cg
        else:
            self.precond = self._blockdiagprecond()

    def hessianab(self, m1, m2):
        self.a.vector().zero()
        self.b.vector().zero()
        Hx = Function(self.V1V2)

        setfct(self.a, self.regul1.hessian(m1))
        setfct(self.b, self.regul2.hessian(m2))
        assign(Hx.sub(0), self.a)
        assign(Hx.sub(1), self.b)
        if self.coeff_cg > 0.0:
            Hx.vector().axpy(self.coeff_cg, self.crossgrad.hessianab(m1, m2))

        return Hx.vector()


    def getprecond(self):
        solver = PETScKrylovSolver("cg", self.amgprecond)
        solver.parameters["maximum_iterations"] = 1000
        solver.parameters["absolute_tolerance"] = 1e-24
        solver.parameters["relative_tolerance"] = 1e-24
        solver.parameters["error_on_nonconvergence"] = True 
        solver.parameters["nonzero_initial_guess"] = False 
        solver.set_operator(self.precond)
        return solver



#----------------------------------------------------------------------
#----------------------------------------------------------------------
# Tikhonov-type regularization
class Tikhonovab():
#TODO: DO NOT USE! Replaced by SumRegularization with LaplacianPrior
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
            self.cg = crossgradient(V)
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

    def costabvect(self, ma_in, mb_in):    return self.costab(ma_in, mb_in)


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

    def gradabvect(self, ma_in, mb_in): return self.gradab(ma_in, mb_in)


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
        solver.parameters["relative_tolerance"] = 1e-24
        solver.parameters["absolute_tolerance"] = 1e-24
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

    def __init__(self, VV):
        """ Input argument:
        VV = MixedFunctionSpace for both inversion parameters """
        self.ab = Function(VV)
        self.abv = self.ab.vector()
        self.MG = Function(VV)
        # cost
        self.a, self.b = self.ab.split(deepcopy=True)
        self.cost = 0.5*( inner(nabla_grad(self.a), nabla_grad(self.a))*\
        inner(nabla_grad(self.b), nabla_grad(self.b))*dx - \
        inner(nabla_grad(self.a), nabla_grad(self.b))*\
        inner(nabla_grad(self.a), nabla_grad(self.b))*dx )
        # gradient
        testa, testb = TestFunction(VV)
        grada = inner( nabla_grad(testa), \
        inner(nabla_grad(self.b), nabla_grad(self.b))*nabla_grad(self.a) - \
        inner(nabla_grad(self.a), nabla_grad(self.b))*nabla_grad(self.b) )*dx
        gradb = inner( nabla_grad(testb), \
        inner(nabla_grad(self.a), nabla_grad(self.a))*nabla_grad(self.b) - \
        inner(nabla_grad(self.a), nabla_grad(self.b))*nabla_grad(self.a) )*dx
        self.grad = grada + gradb
        # Hessian
        self.ahat, self.bhat = self.ab.split(deepcopy=True)
        self.abhat = Function(VV)
        at, bt = TestFunction(VV)
        ah, bh = TrialFunction(VV)
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
        #self.H = None
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
# TV-type regularization
class VTV():
    """ Define Vectorial Total Variation regularization for 2 parameters """

    def __init__(self, Vm, parameters=[]):
        """ Vm = FunctionSpace for the parameters (m1, m2) """

        self.parameters = {'k':1.0, 'eps':1e-2}
        self.parameters.update(parameters)
        k = self.parameters['k']
        eps = self.parameters['eps']

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
        TVnormsq = normm1 + normm2 + Constant(eps)
        TVnorm = sqrt(TVnormsq)
        self.wkformcost = Constant(k) * TVnorm * dx

        # gradient
        gradm1 = Constant(k)/TVnorm*inner(nabla_grad(self.m1), nabla_grad(testm1))*dx
        gradm2 = Constant(k)/TVnorm*inner(nabla_grad(self.m2), nabla_grad(testm2))*dx
        self.gradm = gradm1 + gradm2

        # Hessian
        H11 = Constant(k)/TVnorm*(inner(nabla_grad(trialm1), nabla_grad(testm1)) - \
        inner(nabla_grad(testm1),nabla_grad(self.m1))* \
        inner(nabla_grad(self.m1), nabla_grad(trialm1))/TVnormsq)*dx
        H12 = -Constant(k)/(TVnorm*TVnormsq)*(inner(nabla_grad(testm1),nabla_grad(self.m1))* \
        inner(nabla_grad(self.m2), nabla_grad(trialm2)))*dx
        H21 = -Constant(k)/(TVnorm*TVnormsq)*(inner(nabla_grad(testm2),nabla_grad(self.m2))* \
        inner(nabla_grad(self.m1), nabla_grad(trialm1)))*dx
        H22 = Constant(k)/TVnorm*(inner(nabla_grad(trialm2), nabla_grad(testm2)) - \
        inner(nabla_grad(testm2),nabla_grad(self.m2))* \
        inner(nabla_grad(self.m2), nabla_grad(trialm2))/TVnormsq)*dx
        self.hessian = H11 + H12 + H21 + H22
        #TODO: need a preconditioner for Hessian


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
