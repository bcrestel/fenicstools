import sys
import numpy as np

from dolfin import inner, nabla_grad, dx, \
Function, TestFunction, TrialFunction, assemble, \
PETScKrylovSolver, assign
from miscfenics import setfct


class Tikhonovab():
    """ Define Tikhonov regularization for a and b parameters """

    def __init__(self, parameters):
        Vm = parameters['Vm']
        gamma = parameters['gamma']
        beta = parameters['beta']
        VV = Vm*Vm
        test, trial = TestFunction(VV), TrialFunction(VV)
        K = assemble(inner(nabla_grad(trial), nabla_grad(test))*dx)
        M = assemble(inner(trial, test)*dx)
        self.R = gamma*K + beta*M
        if beta < 1e-14:
            self.precond = gamma*K + 1e-14*M
        else:
            self.precond = self.R
        self.a, self.b = Function(Vm), Function(Vm)
        self.ab = Function(VV)
        self.abv = self.ab.vector()

    def costab(self, ma_in, mb_in):
        """ ma_in, mb_in = Function(V) """
        assign(self.ab.sub(0), ma_in)
        assign(self.ab.sub(1), mb_in)
        return 0.5 * self.abv.inner(self.R * self.abv)

    def gradab(self, ma_in, mb_in):
        """ ma_in, mb_in = Function(V) """
        assign(self.ab.sub(0), ma_in)
        assign(self.ab.sub(1), mb_in)
        return self.R * self.abv

    def assemble_hessian(self, m_in):
        pass

    def hessianab(self, ahat, bhat):
        """ ahat, bhat = Vector(V) """
        setfct(self.a, ahat)
        setfct(self.b, bhat)
        return self.gradab(self.a, self.b)

#    def mult(self, abhat, y):
#        setfct(self.MGab, abhat)
#        ahat, bhat = self.MGab.split(deepcopy=True)
#        out = self.hessianab(ahat.vector(), bhat.vector())
#        y.zero()
#        y.axpy(1.0, out)

    def getprecond(self):
        """
        # no converge w/o preconditioning
        solver = PETScKrylovSolver("cg", "none")
        solver.set_operator(self) 
        solver.parameters["maximum_iterations"] = 1000
        solver.parameters["relative_tolerance"] = 1e-10
        solver.parameters["error_on_nonconvergence"] = True 
        solver.parameters["nonzero_initial_guess"] = False 
        "" "
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
        solver.set_operator(self.precond)
        return solver
        

