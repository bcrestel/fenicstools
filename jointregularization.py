import sys
import numpy as np

from dolfin import inner, nabla_grad, dx, \
Function, TestFunction, TrialFunction, assemble, \
PETScKrylovSolver, assign
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
        return grad

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
        self.Hdiag = assemble(self.precond)

    def hessianab(self, ahat, bhat):
        """ ahat, bhat = Vector(V) """
        setfct(self.ahat, ahat)
        setfct(self.bhat, bhat)
        assign(self.abhat.sub(0), self.ahat)
        assign(self.abhat.sub(1), self.bhat)
        return self.H * self.abhat.vector()
