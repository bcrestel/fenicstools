import sys
import numpy as np

from dolfin import inner, nabla_grad, dx, \
Function, TestFunction, TrialFunction, assemble, \
PETScKrylovSolver, assign
from miscfenics import setfct


class Tikhonovab():
    """ Define Tikhonov regularization for a and b parameters """

    def __init__(self, parameters):
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
            self.precond = gamma*K + 1e-10*M
        else:
            self.precond = self.R
        self.a, self.b = Function(V), Function(V)
        self.ab = Function(VV)
        self.abv = self.ab.vector()

    def costab(self, ma_in, mb_in):
        """ ma_in, mb_in = Function(V) """
        setfct(self.a, ma_in)
        self.a.vector().axpy(-1.0, self.m0av)
        setfct(self.b, mb_in)
        self.b.vector().axpy(-1.0, self.m0bv)
        assign(self.ab.sub(0), self.a)
        assign(self.ab.sub(1), self.b)
        return 0.5 * self.abv.inner(self.R * self.abv)

    def gradab(self, ma_in, mb_in):
        """ ma_in, mb_in = Function(V) """
        setfct(self.a, ma_in)
        self.a.vector().axpy(-1.0, self.m0av)
        setfct(self.b, mb_in)
        self.b.vector().axpy(-1.0, self.m0bv)
        assign(self.ab.sub(0), self.a)
        assign(self.ab.sub(1), self.b)
        return self.R * self.abv

    def assemble_hessian(self, m_in):
        pass

    def hessianab(self, ahat, bhat):
        """ ahat, bhat = Vector(V) """
        setfct(self.a, ahat)
        setfct(self.b, bhat)
        assign(self.ab.sub(0), self.a)
        assign(self.ab.sub(1), self.b)
        return self.R * self.abv

#    def mult(self, abhat, y):
#        setfct(self.MGab, abhat)
#        ahat, bhat = self.MGab.split(deepcopy=True)
#        out = self.hessianab(ahat.vector(), bhat.vector())
#        y.zero()
#        y.axpy(1.0, out)

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
        test = TestFunction(V)
        self.grada = inner( \
        inner(nabla_grad(self.b), nabla_grad(self.b))*nabla_grad(self.a) - \
        inner(nabla_grad(self.a), nabla_grad(self.b))*nabla_grad(self.b), \
        test)*dx
        self.gradb = inner( \
        inner(nabla_grad(self.a), nabla_grad(self.a))*nabla_grad(self.b) - \
        inner(nabla_grad(self.a), nabla_grad(self.b))*nabla_grad(self.a), \
        test)*dx

    def costab(self, ma_in, mb_in):
        """ ma_in, mb_in = Function(V) """
        setfct(self.a, ma_in)
        setfct(self.b, mb_in)
        return assemble(self.cost)

    def gradab(self, ma_in, mb_in):
        """ ma_in, mb_in = Function(V) """
        setfct(self.a, ma_in)
        setfct(self.b, mb_in)
        MGa = assemble(self.grada)
        MGb = assemble(self.gradb)
        assign(self.MG.sub(0), MGa)
        assign(self.MG.sub(1), MGb)
        return self.MG.vector()
