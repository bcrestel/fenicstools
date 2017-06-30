"""
Define joint regularization terms
"""
import sys
import numpy as np
from dolfin import inner, nabla_grad, dx, interpolate, cells, \
Function, TestFunction, TrialFunction, assemble, project, \
PETScKrylovSolver, assign, sqrt, Constant, as_backend_type, \
FunctionSpace, VectorFunctionSpace, norm, MPI, Vector, split, derivative,\
UnitSquareMesh, SpatialCoordinate
from miscfenics import setfct, ZeroRegularization, amg_solver, createMixedFS
from linalg.splitandassign import BlockDiagonal, PrecondPlusIdentity
from linalg.miscroutines import setglobalvalue
try:
    from regularization import TV, TVPD
    from hippylib.linalg import pointwiseMaxCount
except:
    pass



class SingleRegularization():
    """
    Implement regularization for a single parameter
    Used to solve single inverse problem with code for joint inverse problem
    Parameter fixed has zero cost, zero gradient, and zero Hessian, but identity
    preconditioner
    """
    def __init__(self, regul, param, isprint=False):
        """
        Arguments:
            regul = regularization for inversion parameters
            param = inversion parameters (either 'a' or 'b')
            isprint = boolean
        """
        self.param = param
        if self.param == 'a':
            self.regul1 = regul
            self.regul2 = ZeroRegularization(regul.Vm)
        elif self.param == 'b':
            self.regul1 = ZeroRegularization(regul.Vm)
            self.regul2 = regul
        else:
            if isprint:
                print "[SingleRegularization] *** Error: argument 'param' must be 'a' or 'b'"
                sys.exit(1)
        self.isprint = isprint

        Vm = regul.Vm
        self.VmVm = createMixedFS(Vm, Vm)
        self.ab = Function(self.VmVm)
        bd = BlockDiagonal(Vm, Vm, Vm.mesh().mpi_comm())
        self.saa = bd.saa

        if isprint:
            print '[SingleRegularization] inversion parameter {}'.format(self.param)
            if self.isPD():
                print '[SingleRegularization] Using primal-dual TV'


    def isTV(self):
        return self.regul1.isTV() or self.regul2.isTV()
    def isPD(self):
        return self.regul1.isPD() or self.regul2.isPD()


    def costab(self, m1, m2):
        return self.regul1.cost(m1) + self.regul2.cost(m2)

    def costabvect(self, m1, m2):
        return self.regul1.costvect(m1) + self.regul2.costvect(m2)


    def gradab(self, m1, m2):
        grad1 = self.regul1.grad(m1)
        grad2 = self.regul2.grad(m2)
        return self.saa.assign(grad1, grad2)

    def gradabvect(self, m1, m2):
        """ relies on gradvect method from regularization instead of grad
        gradvect takes a Vector() as input argument """
        grad1 = self.regul1.gradvect(m1)
        grad2 = self.regul2.gradvect(m2)
        return self.saa.assign(grad1, grad2)


    def assemble_hessianab(self, m1, m2):
        self.regul1.assemble_hessian(m1)
        self.regul2.assemble_hessian(m2)

    def hessianab(self, m1, m2):
        Hx1 = self.regul1.hessian(m1)
        Hx2 = self.regul2.hessian(m2)
        return self.saa.assign(Hx1, Hx2)


    def getprecond(self):
        if self.param == 'a':
            precondsolver = self.regul1.getprecond()
        elif self.param == 'b':
            precondsolver = self.regul2.getprecond()
        return PrecondPlusIdentity(precondsolver, self.param, self.VmVm)


    def update_w(self, mhat, alphaLS, compute_what=True):
        """ update dual variable in direction what 
        and update re-scaled version """
        mhat1, mhat2 = self.saa.split(mhat)
        self.regul1.update_w(mhat1, alphaLS, compute_what)
        self.regul2.update_w(mhat2, alphaLS, compute_what)




class SumRegularization():
    """ 
    Sum of independent regularizations for each med. param, 
    potentially connected by cross-gradient or VTV 
    """

    def __init__(self, regul1, regul2, 
    coeff_cg=0.0, 
    coeff_ncg=0.0, parameters_ncg=[],
    coeff_vtv=0.0, parameters_vtv=[],
    isprint=False):
        """ regul1, regul2 = regularization/prior objects for medium 1 and 2
        coeff_cg = regularization constant for cross-gradient term
        coeff_ncg = regularization constant for normalized cross-gradient term
        coeff_vtv = regularization constant for VTV term; if 0.0, no VTV
        """
        assert id(regul1) != id(regul2), "Need to define two distinct regul objects"

        self.regul1 = regul1
        self.regul2 = regul2
        self.coeff_cg = coeff_cg
        self.coeff_ncg = coeff_ncg
        self.coeff_vtv = coeff_vtv

        V1 = self.regul1.Vm
        V2 = self.regul2.Vm
        self.V1V2 = createMixedFS(V1, V2)
        self.a, self.b = Function(V1), Function(V2)
        self.ab = Function(self.V1V2)
        mpicomm = V1.mesh().mpi_comm()
        self.bd = BlockDiagonal(V1, V2, mpicomm)
        self.saa = self.bd.saa

        test, trial = TestFunction(self.V1V2), TrialFunction(self.V1V2)
        self.M = assemble(inner(test, trial)*dx)

        if self.coeff_cg > 0.0:
            self.crossgrad = crossgradient(self.V1V2)
            if isprint: print "[SumRegularization] Using cross-gradient"
        if self.coeff_ncg > 0.0:
            self.normalizedcrossgrad = normalizedcrossgradient(self.V1V2, parameters_ncg)
            if isprint: 
                print '[SumRegularization] Using normalized cross-gradient (eps={})'.format(
                self.normalizedcrossgrad.parameters['eps'])
        if self.coeff_vtv > 0.0:
            assert self.regul1.Vm is self.regul2.Vm
            self.vtv = V_TVPD(V1, parameters_vtv)
            if isprint: print "[SumRegularization] Using VTV"
        if isprint:
            print '[SumRegularization] Coeff cg={}, ncg={}, vtv={}'.format(
            self.coeff_cg, self.coeff_ncg, self.coeff_vtv)

        self.amgprecond = amg_solver()


    def isTV(self):
        return self.regul1.isTV() * self.regul2.isTV()
    def isPD(self):
        return self.regul1.isPD() * self.regul2.isPD()


    def costab(self, m1, m2):
        cost = self.regul1.cost(m1) + self.regul2.cost(m2)
        if self.coeff_cg > 0.0:
            cost += self.coeff_cg*self.crossgrad.costab(m1, m2)
        if self.coeff_ncg > 0.0:
            cost += self.coeff_ncg*self.normalizedcrossgrad.costab(m1, m2)
        if self.coeff_vtv > 0.0:
            cost += self.coeff_vtv*self.vtv.costab(m1, m2)
        return cost

    def costabvect(self, m1, m2):
        cost = self.regul1.costvect(m1) + self.regul2.costvect(m2)
        if self.coeff_cg > 0.0:
            cost += self.coeff_cg*self.crossgrad.costab(m1, m2)
        if self.coeff_ncg > 0.0:
            cost += self.coeff_ncg*self.normalizedcrossgrad.costab(m1, m2)
        if self.coeff_vtv > 0.0:
            cost += self.coeff_vtv*self.vtv.costabvect(m1, m2)
        return cost


    def gradab(self, m1, m2):
        grad1 = self.regul1.grad(m1)
        grad2 = self.regul2.grad(m2)
        grad = self.saa.assign(grad1, grad2)
        if self.coeff_cg > 0.0:
            grad.axpy(self.coeff_cg, self.crossgrad.gradab(m1, m2))
        if self.coeff_ncg > 0.0:
            grad.axpy(self.coeff_ncg, self.normalizedcrossgrad.gradab(m1, m2))
        if self.coeff_vtv > 0.0:
            grad.axpy(self.coeff_vtv, self.vtv.gradab(m1, m2))
        return grad

    def gradabvect(self, m1, m2):
        """ relies on gradvect metod from regularization instead of grad
        gradvect takes a Vector() as input argument """
        grad1 = self.regul1.gradvect(m1)
        grad2 = self.regul2.gradvect(m2)
        grad = self.saa.assign(grad1, grad2)
        if self.coeff_cg > 0.0:
            grad.axpy(self.coeff_cg, self.crossgrad.gradab(m1, m2))
        if self.coeff_ncg > 0.0:
            grad.axpy(self.coeff_ncg, self.normalizedcrossgrad.gradab(m1, m2))
        if self.coeff_vtv > 0.0:
            grad.axpy(self.coeff_vtv, self.vtv.gradabvect(m1, m2))
        return grad


    def _blockdiagprecond(self):
        """ assemble a block-diagonal preconditioner
        with preconditioners for each regularization term """
        R1 = self.regul1.precond
        R2 = self.regul2.precond
        return self.bd.assemble(R1, R2)

    def assemble_hessianab(self, m1, m2):
        self.regul1.assemble_hessian(m1)
        self.regul2.assemble_hessian(m2)
        self.precond = self._blockdiagprecond()
        if self.coeff_cg > 0.0:
            self.crossgrad.assemble_hessianab(m1, m2)
            self.precond += self.crossgrad.Hprecond*self.coeff_cg
        if self.coeff_ncg > 0.0:
            self.normalizedcrossgrad.assemble_hessianab(m1, m2)
            self.precond += self.M*(1e-2*self.coeff_ncg)
            #self.precond += self.normalizedcrossgrad.Hprecond*self.coeff_ncg
        if self.coeff_vtv > 0.0:
            self.vtv.assemble_hessianab(m1, m2)
            self.precond += self.vtv.regTV.precond*self.coeff_vtv

    def hessianab(self, m1, m2):
        Hx1 = self.regul1.hessian(m1)
        Hx2 = self.regul2.hessian(m2)
        Hx = self.saa.assign(Hx1, Hx2)
        if self.coeff_cg > 0.0:
            Hx.axpy(self.coeff_cg, self.crossgrad.hessianab(m1, m2))
        if self.coeff_ncg > 0.0:
            Hx.axpy(self.coeff_ncg, self.normalizedcrossgrad.hessianab(m1, m2))
        if self.coeff_vtv > 0.0:
            Hx.axpy(self.coeff_vtv, self.vtv.hessianab(m1, m2))
        return Hx


    def getprecond(self):
#        if self.coeff_cg + self.coeff_ncg > 0.0:
#            solver = PETScKrylovSolver('gmres', self.amgprecond)
#            solver.parameters["maximum_iterations"] = 10000
#        else:
        solver = PETScKrylovSolver('cg', self.amgprecond)
        solver.parameters["maximum_iterations"] = 2000
        solver.parameters["absolute_tolerance"] = 1e-24
        solver.parameters["relative_tolerance"] = 1e-24
        solver.parameters["error_on_nonconvergence"] = True 
        solver.parameters["nonzero_initial_guess"] = False 
        solver.set_operator(self.precond)
        return solver


    def update_w(self, mhat, alphaLS, compute_what=True):
        """ update dual variable in direction what 
        and update re-scaled version """
        mhat1, mhat2 = self.saa.split(mhat)
        self.regul1.update_w(mhat1, alphaLS, compute_what)
        self.regul2.update_w(mhat2, alphaLS, compute_what)
        if self.coeff_vtv > 0.0:
            self.vtv.update_w(mhat, alphaLS, compute_what)



#----------------------------------------------------------------------
#----------------------------------------------------------------------
# Cross-gradient penalty functionals

class crossgradient():
    """ Define cross-gradient joint regularization """
    #TODO: introduce constant eps

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
        return assemble(self.cost)  # this is global (in parallel)

    def gradab(self, ma_in, mb_in):
        """ ma_in, mb_in = Function(V) """
        setfct(self.a, ma_in)
        setfct(self.b, mb_in)
        return assemble(self.grad)

    def assemble_hessianab(self, a, b):
        setfct(self.a, a)
        setfct(self.b, b)
        self.H = assemble(self.hessian)
        self.Hprecond = assemble(self.precond)

    def hessianab(self, ahat, bhat):
        """ ahat, bhat = Vector(V) """
        setfct(self.ahat, ahat)
        setfct(self.bhat, bhat)
        assign(self.abhat.sub(0), self.ahat)
        assign(self.abhat.sub(1), self.bhat)
        return self.H * self.abhat.vector()



class normalizedcrossgradient():
    """
    Defined normalized cross-gradient 
    |nabla m1 x nabla m2|^2/(|nabla m1|^2|nabla m2|^2)
    """

    def __init__(self, VV, parameters=[]):
        self.parameters = {}
        self.parameters['eps']          = 1e-4
        self.parameters['correctcost']  = True
        self.parameters.update(parameters)
        eps = Constant(self.parameters['eps'])

        self.ab = Function(VV)
        self.a, self.b = self.ab.split(deepcopy=True)
        normgrada = sqrt(inner(nabla_grad(self.a), nabla_grad(self.a)) + eps)
        ngrada = nabla_grad(self.a) / normgrada
        normgradb = sqrt(inner(nabla_grad(self.b), nabla_grad(self.b)) + eps)
        ngradb = nabla_grad(self.b) / normgradb

        # cost
        if self.parameters['correctcost']:
            meshtmp = UnitSquareMesh(VV.mesh().mpi_comm(), 10, 10)
            Vtmp = FunctionSpace(meshtmp, 'CG', 1)
            x = SpatialCoordinate(meshtmp)
            correctioncost = 1./assemble(sqrt(4.0*x[0]*x[0])*dx)
            print '[NCG] Correction cost with factor={}'.format(correctioncost)
        else:
            correctioncost = 1.0
        self.cost = 0.5*correctioncost*(1.0 - inner(ngrada, ngradb)*inner(ngrada, ngradb))*dx
        # gradient
        testa, testb = TestFunction(VV)
        grada = - inner(ngrada, ngradb)* \
        (inner(ngradb/normgrada, nabla_grad(testa)) - 
        inner(ngrada, ngradb)* \
        inner(ngrada/normgrada, nabla_grad(testa)))*dx
        gradb = - inner(ngrada, ngradb)* \
        (inner(ngrada/normgradb, nabla_grad(testb)) - 
        inner(ngrada, ngradb)* \
        inner(ngradb/normgradb, nabla_grad(testb)))*dx
        self.grad = grada + gradb
        # Hessian
        self.ahat, self.bhat = self.ab.split(deepcopy=True)
        self.abhat = Function(VV)
        triala, trialb = TrialFunction(VV)
        wkform11 = - ((inner(ngradb/normgrada, nabla_grad(testa))
        - inner(ngrada,ngradb)*inner(ngrada/normgrada, nabla_grad(testa)))*
        (inner(ngradb/normgrada, nabla_grad(triala)) -
        inner(ngrada,ngradb)*inner(ngrada/normgrada, nabla_grad(triala)))
        +
        inner(ngrada,ngradb)*(
        -inner(ngradb/normgrada, nabla_grad(testa))*
        inner(ngrada/normgrada, nabla_grad(triala))
        -inner(ngradb/normgrada, nabla_grad(triala))*
        inner(ngrada/normgrada, nabla_grad(testa))
        -inner(ngrada/normgrada,ngradb/normgrada)*
        inner(nabla_grad(testa), nabla_grad(triala))
        + 3*inner(ngrada,ngradb)*
        inner(ngrada/normgrada,nabla_grad(testa))*
        inner(ngrada/normgrada,nabla_grad(triala))
        ) )*dx
        #
        wkform12 = - ((inner(ngradb/normgrada, nabla_grad(testa))
        - inner(ngrada,ngradb)*inner(ngrada/normgrada, nabla_grad(testa)))*
        (inner(ngrada/normgradb, nabla_grad(trialb)) -
        inner(ngrada,ngradb)*inner(ngradb/normgradb, nabla_grad(trialb)))
        +
        inner(ngrada,ngradb)*(
        inner(nabla_grad(testa)/normgrada, nabla_grad(trialb)/normgradb)
        - inner(ngradb/normgrada, nabla_grad(testa))*
        inner(ngradb/normgradb, nabla_grad(trialb))
        - inner(ngrada/normgrada, nabla_grad(testa))*(
        inner(ngrada/normgradb, nabla_grad(trialb))
        - inner(ngrada, ngradb)*inner(ngradb/normgradb, nabla_grad(trialb)))
        ) )*dx
        #
        wkform22 = - ((inner(ngrada/normgradb, nabla_grad(testb))
        - inner(ngrada,ngradb)*inner(ngradb/normgradb, nabla_grad(testb)))*
        (inner(ngrada/normgradb, nabla_grad(trialb)) -
        inner(ngrada,ngradb)*inner(ngradb/normgradb, nabla_grad(trialb)))
        +
        inner(ngrada,ngradb)*(
        -inner(ngrada/normgradb, nabla_grad(testb))*
        inner(ngradb/normgradb, nabla_grad(trialb))
        -inner(ngrada/normgradb, nabla_grad(trialb))*
        inner(ngradb/normgradb, nabla_grad(testb))
        -inner(ngrada/normgradb,ngradb/normgradb)*
        inner(nabla_grad(testb), nabla_grad(trialb))
        + 3*inner(ngrada,ngradb)*
        inner(ngradb/normgradb,nabla_grad(testb))*
        inner(ngradb/normgradb,nabla_grad(trialb))
        ) )*dx
        #
        wkform21 = -((inner(ngrada/normgradb, nabla_grad(testb))
        - inner(ngrada,ngradb)*inner(ngradb/normgradb, nabla_grad(testb)))*
        (inner(ngradb/normgrada, nabla_grad(triala)) -
        inner(ngrada,ngradb)*inner(ngrada/normgrada, nabla_grad(triala)))
        +
        inner(ngrada,ngradb)*(
        inner(nabla_grad(testb)/normgradb, nabla_grad(triala)/normgrada)
        - inner(ngrada/normgradb, nabla_grad(testb))*
        inner(ngrada/normgrada, nabla_grad(triala))
        - inner(ngradb/normgradb, nabla_grad(testb))*(
        inner(ngradb/normgrada, nabla_grad(triala))
        - inner(ngrada, ngradb)*inner(ngrada/normgrada, nabla_grad(triala)))
        ) )*dx
        #
        self.hessian = wkform11 + wkform22 + wkform12 + wkform21
        self.precond = wkform11 + wkform22 #+ wkform12 + wkform21


    def costab(self, ma_in, mb_in):
        setfct(self.a, ma_in)
        setfct(self.b, mb_in)
        return assemble(self.cost)
        

    def gradab(self, ma_in, mb_in):
        """ ma_in, mb_in = Function(V) """
        setfct(self.a, ma_in)
        setfct(self.b, mb_in)
        return assemble(self.grad)


    def assemble_hessianab(self, a, b):
        setfct(self.a, a)
        setfct(self.b, b)
        self.H = assemble(self.hessian)
        self.Hprecond = assemble(self.precond)

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
        """ Vm = FunctionSpace for the parameters m1, and m2 """
        self.parameters = {}
        self.parameters['k']        = 1.0
        self.parameters['eps']      = 1e-2
        self.parameters['correctcost']  = True
        self.parameters.update(parameters)
        k = self.parameters['k']
        eps = self.parameters['eps']

        self.m1 = Function(Vm)
        self.m2 = Function(Vm)
        self.m1h = Function(Vm)
        self.m2h = Function(Vm)
        VmVm = createMixedFS(Vm, Vm)
        self.m12h = Function(VmVm)
        testm = TestFunction(VmVm)
        testm1, testm2 = testm
        trialm = TrialFunction(VmVm)
        trialm1, trialm2 = trialm

        # cost function
        normm1 = inner(nabla_grad(self.m1), nabla_grad(self.m1))
        normm2 = inner(nabla_grad(self.m2), nabla_grad(self.m2))
        TVnormsq = normm1 + normm2 + Constant(eps)
        TVnorm = sqrt(TVnormsq)
        if self.parameters['correctcost']:
            meshtmp = UnitSquareMesh(Vm.mesh().mpi_comm(), 10, 10)
            Vtmp = FunctionSpace(meshtmp, 'CG', 1)
            x = SpatialCoordinate(meshtmp)
            correctioncost = 1./assemble(sqrt(4.0*x[0]*x[0])*dx)
            print '[VTV] correction cost with factor={}'.format(correctioncost)
        else:
            correctioncost = 1.0
        self.wkformcost = Constant(k*correctioncost) * TVnorm * dx

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

        # for preconditioning
        self.amgprecond = amg_solver()
        M = assemble(inner(testm, trialm)*dx)
        factM = 1e-2*k
        self.sMass = M*factM


    def isTV(self): return True
    def isPD(self): return False


    def costab(self, m1, m2):
        """ Compute value of cost function at (m1,m2) """
        setfct(self.m1, m1)
        setfct(self.m2, m2)
        return assemble(self.wkformcost)

    def costabvect(self, m1, m2):   return self.costab(m1, m2)


    def gradab(self, m1, m2):
        """ returns gradient at (m1,m2) as a vector """
        setfct(self.m1, m1)
        setfct(self.m2, m2)
        return assemble(self.gradm)

    def gradabvect(self, m1, m2):   return self.gradab(m1, m2)


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


    def getprecond(self):
        """ precondition by TV + small fraction of mass matrix """
        solver = PETScKrylovSolver('cg', self.amgprecond)
        solver.parameters["maximum_iterations"] = 2000
        solver.parameters["relative_tolerance"] = 1e-24
        solver.parameters["absolute_tolerance"] = 1e-24
        solver.parameters["error_on_nonconvergence"] = True 
        solver.parameters["nonzero_initial_guess"] = False 
        solver.set_operator(self.H + self.sMass)
        return solver


class V_TV():
    """ Definite Vectorial Total Variation regularization from Total Variation class """

    def __init__(self, Vm, parameters=[]):
        """ Vm = FunctionSpace for the parameters m1, and m2 """
        self.parameters = {'k':1.0, 'eps':1e-2}
        self.parameters.update(parameters)
        VmVm = createMixedFS(Vm, Vm)
        self.parameters['Vm'] = VmVm

        self.regTV = TV(self.parameters)

        self.m1, self.m2 = Function(Vm), Function(Vm)
        self.m = Function(VmVm)


    def isTV(self): return True
    def isPD(self): return False


    def costab(self, m1, m2):
        assign(self.m.sub(0), m1)
        assign(self.m.sub(1), m2)
        return self.regTV.cost(self.m)

    def costabvect(self, m1, m2):
        setfct(self.m1, m1)
        setfct(self.m2, m2)
        return self.costab(self.m1, self.m2)


    def gradab(self, m1, m2):
        assign(self.m.sub(0), m1)
        assign(self.m.sub(1), m2)
        return self.regTV.grad(self.m)

    def gradabvect(self, m1, m2):
        setfct(self.m1, m1)
        setfct(self.m2, m2)
        return self.gradab(self.m1, self.m2)


    def assemble_hessianab(self, m1, m2):
        setfct(self.m1, m1)
        setfct(self.m2, m2)
        assign(self.m.sub(0), self.m1)
        assign(self.m.sub(1), self.m2)
        self.regTV.assemble_hessian(self.m)

    def hessianab(self, m1h, m2h):
        """ m1h, m2h = Vector(V) """
        setfct(self.m1, m1h)
        setfct(self.m2, m2h)
        assign(self.m.sub(0), self.m1)
        assign(self.m.sub(1), self.m2)
        return self.regTV.hessian(self.m.vector())

    def getprecond(self):
        return self.regTV.getprecond()



class V_TVPD():
    """ Definite Vectorial Total Variation regularization from Total Variation class """

    def __init__(self, Vm, parameters=[]):
        """ Vm = FunctionSpace for the parameters m1, and m2 """
        self.parameters = {}
        self.parameters['k'] = 1.0
        self.parameters['eps'] = 1e-2
        self.parameters['rescaledradiusdual'] = 1.0
        self.parameters['print'] = False
        self.parameters.update(parameters)

        VmVm = createMixedFS(Vm, Vm)
        self.parameters['Vm'] = VmVm
        Vw = FunctionSpace(Vm.mesh(), 'DG', 0)
        VwVw = createMixedFS(Vw, Vw)
        self.parameters['Vw'] = VwVw

        self.regTV = TVPD(self.parameters)

        self.m1, self.m2 = Function(Vm), Function(Vm)
        self.m = Function(VmVm)

        self.w_loc = Function(VwVw)
        self.factorw = Function(Vw)
        self.factorww = Function(VwVw)

        tmp = interpolate(Constant("1.0"), Vw)
        self.one = tmp.vector()


    def isTV(self): return True
    def isPD(self): return True


    def costab(self, m1, m2):
        assign(self.m.sub(0), m1)
        assign(self.m.sub(1), m2)
        return self.regTV.cost(self.m)

    def costabvect(self, m1, m2):
        setfct(self.m1, m1)
        setfct(self.m2, m2)
        return self.costab(self.m1, self.m2)


    def gradab(self, m1, m2):
        assign(self.m.sub(0), m1)
        assign(self.m.sub(1), m2)
        return self.regTV.grad(self.m)

    def gradabvect(self, m1, m2):
        setfct(self.m1, m1)
        setfct(self.m2, m2)
        return self.gradab(self.m1, self.m2)


    def assemble_hessianab(self, m1, m2):
        setfct(self.m1, m1)
        setfct(self.m2, m2)
        assign(self.m.sub(0), self.m1)
        assign(self.m.sub(1), self.m2)
        self.regTV.assemble_hessian(self.m)

    def hessianab(self, m1h, m2h):
        """ m1h, m2h = Vector(V) """
        setfct(self.m1, m1h)
        setfct(self.m2, m2h)
        assign(self.m.sub(0), self.m1)
        assign(self.m.sub(1), self.m2)
        return self.regTV.hessian(self.m.vector())

    def getprecond(self):
        return self.regTV.getprecond()


    def compute_what(self, mhat):
        self.regTV.compute_what(mhat)


    def update_w(self, mhat, alphaLS, compute_what=True):
        """ update dual variable in direction what 
        and update re-scaled version """
        # Update wx and wy
        if compute_what:    self.compute_what(mhat)
        self.regTV.wx.vector().axpy(alphaLS, self.regTV.wxhat.vector())
        self.regTV.wy.vector().axpy(alphaLS, self.regTV.wyhat.vector())

        # Update rescaled variables
        rescaledradiusdual = self.parameters['rescaledradiusdual']    
        # wx**2
        as_backend_type(self.regTV.wxsq).vec().pointwiseMult(\
            as_backend_type(self.regTV.wx.vector()).vec(),\
            as_backend_type(self.regTV.wx.vector()).vec())
        # wy**2
        as_backend_type(self.regTV.wysq).vec().pointwiseMult(\
            as_backend_type(self.regTV.wy.vector()).vec(),\
            as_backend_type(self.regTV.wy.vector()).vec())
        # |w|
        self.w_loc.vector().zero()
        self.w_loc.vector().axpy(1.0, self.regTV.wxsq + self.regTV.wysq)
        normw1, normw2 = self.w_loc.split(deepcopy=True)
        normw = normw1.vector() + normw2.vector()
        as_backend_type(normw).vec().sqrtabs()
        # |w|/r
        as_backend_type(normw).vec().pointwiseDivide(\
            as_backend_type(normw).vec(),\
            as_backend_type(self.one*rescaledradiusdual).vec())
        # max(1.0, |w|/r)
        count = pointwiseMaxCount(self.factorw.vector(), normw, 1.0)
        # rescale wx and wy
        assign(self.factorww.sub(0), self.factorw)
        assign(self.factorww.sub(1), self.factorw)
        as_backend_type(self.regTV.wxrs.vector()).vec().pointwiseDivide(\
            as_backend_type(self.regTV.wx.vector()).vec(),\
            as_backend_type(self.factorww.vector()).vec())
        as_backend_type(self.regTV.wyrs.vector()).vec().pointwiseDivide(\
            as_backend_type(self.regTV.wy.vector()).vec(),\
            as_backend_type(self.factorww.vector()).vec())

        minf = self.factorw.vector().min()
        maxf = self.factorw.vector().max()
        if self.parameters['print']:
            print ('[V_TVPD] perc. dual entries rescaled={:.2f} %, ' +\
            'min(factorw)={}, max(factorw)={}').format(\
            100.*float(count)/self.factorw.vector().size(), minf, maxf)




#----------------------------------------------------------------------
#----------------------------------------------------------------------
class NuclearNormSVD2D():

    def __init__(self, mesh, parameters_in=[], isprint=False):
        self.V = FunctionSpace(mesh, 'CG', 1)
        self.Vd = VectorFunctionSpace(mesh, 'DG', 0)
        self.VV = createMixedFS(self.V, self.V)

        self.mpicomm = mesh.mpi_comm()

        self.parameters = {}
        self.parameters['eps'] = 0.0
        self.parameters['k'] = 1.0
        self.parameters.update(parameters_in)
        self.eps = self.parameters['eps']
        self.k = self.parameters['k']

        self.tmpm1 = Function(self.V)
        self.tmpm2 = Function(self.V)
        self.gradm1 = Function(self.Vd)
        self.gradm2 = Function(self.Vd)
        self.test1, self.test2 = TestFunction(self.VV)

        # evaluation points = centroids
        self.x = []
        self.vol = []
        for cell in cells(mesh):
            self.x.append(\
            cell.get_vertex_coordinates().reshape((-1,2)).mean(axis=0))
            self.vol.append(cell.volume())

        # pre-assemble int of grad(test) over each cell
        indfct = TrialFunction(FunctionSpace(mesh, 'DG', 0))
        self.Gx1test = assemble(indfct*(self.test1.dx(0))*dx)
        self.Gx2test = assemble(indfct*(self.test2.dx(0))*dx)
        self.Gy1test = assemble(indfct*(self.test1.dx(1))*dx)
        self.Gy2test = assemble(indfct*(self.test2.dx(1))*dx)

        if isprint:
            print '[NuclearNormSVD2D] eps={}, k={}'.format(self.eps, self.k)
        

    def isTV(self): return False
    def isPD(self): return False


    def costab(self, m1, m2):
        self.gradm1 = project(nabla_grad(m1), self.Vd)
        self.gradm2 = project(nabla_grad(m2), self.Vd)

        cost = 0.0
        for x, vol in zip(self.x, self.vol):
            G = np.array([self.gradm1(x), self.gradm2(x)]).T
            u, s, v = np.linalg.svd(G)
            sqrts2eps = np.sqrt(s**2 + self.eps)
            cost += vol * sqrts2eps.sum()

        cost_global = MPI.sum(self.mpicomm, cost)
        return self.k*cost_global

    def costabvect(self, m1, m2):
        setfct(self.tmpm1, m1)
        setfct(self.tmpm2, m2)
        return self.costab(self.tmpm1, self.tmpm2)


    def gradab(self, m1, m2):
        self.gradm1 = project(nabla_grad(m1), self.Vd)
        self.gradm2 = project(nabla_grad(m2), self.Vd)

        uwv00, uwv00ind = [], []
        uwv10, uwv10ind = [], []
        uwv01, uwv01ind = [], []
        uwv11, uwv11ind = [], []
        for ii, x in enumerate(self.x):
            G = np.array([self.gradm1(x), self.gradm2(x)]).T
            u, s, v = np.linalg.svd(G)
            sqrts2eps = np.sqrt(s**2 + self.eps)
            W = np.diag(s/sqrts2eps)
            uwv = u.dot(W.dot(v))

            uwv00.append(uwv[0][0])
            uwv00ind.append(ii)

            uwv10.append(uwv[1][0])
            uwv10ind.append(ii)

            uwv01.append(uwv[0][1])
            uwv01ind.append(ii)

            uwv11.append(uwv[1][1])
            uwv11ind.append(ii)

        Grad = Function(self.VV)
        grad = Grad.vector()
        rhsG = Vector()
        self.Gx1test.init_vector(rhsG, 1)

        rhsG.set_local(np.array(uwv00), np.array(uwv00ind, dtype=np.intc))
        rhsG.apply('insert')
        grad.axpy(1.0, self.Gx1test * rhsG)

        rhsG.zero()
        rhsG.set_local(np.array(uwv10), np.array(uwv10ind, dtype=np.intc))
        rhsG.apply('insert')
        grad.axpy(1.0, self.Gy1test * rhsG)

        rhsG.set_local(np.array(uwv01), np.array(uwv01ind, dtype=np.intc))
        rhsG.apply('insert')
        grad.axpy(1.0, self.Gx2test * rhsG)

        rhsG.set_local(np.array(uwv11), np.array(uwv11ind, dtype=np.intc))
        rhsG.apply('insert')
        grad.axpy(1.0, self.Gy2test * rhsG)

        return grad*self.k

    def gradabvect(self, m1, m2):
        setfct(self.tmpm1, m1)
        setfct(self.tmpm2, m2)
        return self.gradab(self.tmpm1, self.tmpm2)


    def assemble_hessianab(self, m1, m2):
        pass



class NuclearNormformula():

    def __init__(self, mesh, parameters=[], isprint=False):
        self.parameters = {}
        self.parameters['eps']          = 0.0
        self.parameters['k']            = 1.0
        self.parameters['correctcost']  = True
        self.parameters.update(parameters)
        eps = self.parameters['eps']
        k = self.parameters['k']

        self.V = FunctionSpace(mesh, 'CG', 1)
        self.tmp1, self.tmp2 = Function(self.V), Function(self.V)

        self.VV = VectorFunctionSpace(mesh, 'CG', 1, 2)
        self.m = Function(self.VV)
        self.mtest = TestFunction(self.VV)
        self.mtrial = TrialFunction(self.VV)
        self.m1, self.m2 = split(self.m)
        self.mh = Function(self.VV)

        normg1 = inner(nabla_grad(self.m1), nabla_grad(self.m1))
        normg2 = inner(nabla_grad(self.m2), nabla_grad(self.m2))
        if self.parameters['correctcost']:
            meshtmp = UnitSquareMesh(mesh.mpi_comm(), 10, 10)
            Vtmp = FunctionSpace(meshtmp, 'CG', 1)
            x = SpatialCoordinate(meshtmp)
            self.correctioncost = 1./assemble(sqrt(4.0*x[0]*x[0])*dx)
            print '[NuclearNormformula] Correction cost with factor={}'.format(self.correctioncost)
        else:
            self.correctioncost = 1.0
        self.cost = 1./np.sqrt(2.0) * Constant(k) * (\
        sqrt(normg1 + normg2 + Constant(np.sqrt(eps)) + 
        sqrt((normg1 - normg2)**2 + Constant(eps) +
        4.0*inner(nabla_grad(self.m1), nabla_grad(self.m2))**2))
        + sqrt(normg1 + normg2 + Constant(np.sqrt(eps)*(1.0+1e-15)) - 
        sqrt((normg1 - normg2)**2 + Constant(eps) +
        4.0*inner(nabla_grad(self.m1), nabla_grad(self.m2))**2)))*dx

        self.grad = derivative(self.cost, self.m, self.mtest)

        self.hessian = derivative(self.grad, self.m, self.mtrial)

        self.amgprecond = amg_solver()

        M = assemble(inner(self.mtest, self.mtrial)*dx)
        factM = 1e-2*k
        self.sMass = M*factM

        if isprint:
            print '[NuclearNormformula] eps={}, k={}'.format(eps, k)


    def isTV(self): return False
    def isPD(self): return False


    def costab(self, m1, m2):
        assign(self.m.sub(0), m1)
        assign(self.m.sub(1), m2)
        return assemble(self.cost)*self.correctioncost

    def costabvect(self, m1, m2):
        setfct(self.tmp1, m1)
        setfct(self.tmp2, m2)
        return self.costab(self.tmp1, self.tmp2)


    def gradab(self, m1, m2):
        assign(self.m.sub(0), m1)
        assign(self.m.sub(1), m2)
        return assemble(self.grad)

    def gradabvect(self, m1, m2):
        setfct(self.tmp1, m1)
        setfct(self.tmp2, m2)
        return self.gradab(self.tmp1, self.tmp2)


    def assemble_hessianab(self, m1, m2):
        setfct(self.tmp1, m1)
        setfct(self.tmp2, m2)
        assign(self.m.sub(0), self.tmp1)
        assign(self.m.sub(1), self.tmp2)
        self.H = assemble(self.hessian)

    def hessianab(self, m1h, m2h):
        """ m1h, m2h = Vector(V) """
        setfct(self.tmp1, m1h)
        setfct(self.tmp2, m2h)
        assign(self.mh.sub(0), self.tmp1)
        assign(self.mh.sub(1), self.tmp2)
        return self.H * self.mh.vector()


    def getprecond(self):
        """ precondition by TV + small fraction of mass matrix """
        #TODO: does not appear to be a great way to apply preconditioner (DIVERGED_ITS)
        solver = PETScKrylovSolver('cg', self.amgprecond)
        solver.parameters["maximum_iterations"] = 3000
        solver.parameters["relative_tolerance"] = 1e-24
        solver.parameters["absolute_tolerance"] = 1e-24
        solver.parameters["error_on_nonconvergence"] = True 
        solver.parameters["nonzero_initial_guess"] = False 
        self.precond = self.H + self.sMass
        solver.set_operator(self.precond)
        return solver

