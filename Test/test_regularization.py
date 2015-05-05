import unittest
import numpy as np

from dolfin import *
from regularization import TikhonovH1

class TestTikhonovH1(unittest.TestCase):

    def setUp(self):
        mesh = UnitSquareMesh(5,5,'crossed')
        self.Vm = FunctionSpace(mesh, 'Lagrange', 2)
        self.m = Function(self.Vm)
        self.m.vector()[:] = np.random.randn(self.Vm.dim())
        self.Regulg = TikhonovH1({'Vm': self.Vm, 'gamma': 1e-5})
        self.Regulb = TikhonovH1({'Vm': self.Vm, 'gamma': 0.0, 'beta':1e-5})
        self.Regul = TikhonovH1({'Vm': self.Vm, 'gamma': 1e-5, \
        'beta':1e-10, 'm0': self.m})
        self.M = []
        for ii in range(10):    self.M.append(Function(self.Vm))
        self.lenm = len(self.M[0].vector().array())
        for mm in self.M:
            mm.vector()[:] = np.random.randn(self.lenm)

    def test00a_inst(self):
        """Default instantiation and check default values"""
        Regul = TikhonovH1({'Vm': self.Vm, 'gamma': 1e-5})
        error = Regul.beta + np.linalg.norm(Regul.m0.vector().array())
        self.assertTrue(error < 1e-16, error)

    def test00b_inst(self):
        """Default instantiation"""
        Regul = TikhonovH1({'Vm': self.Vm, 'gamma': 1e-5, \
        'beta': 1e-7, 'm0': self.m})

    def test01_gamma(self):
        """Check gamma applied consistently"""
        Regul10 = TikhonovH1({'Vm': self.Vm, 'gamma': 1e-10})
        error = 0.0
        for mm in self.M:
            r1 = (self.Regulg.grad(mm)).array()
            r10 = (Regul10.grad(mm)).array()
            err = np.linalg.norm(r1 - 1e5*r10)/np.linalg.norm(r1)
            error = max(error, err)
        self.assertTrue(error < 2e-16, error)

    def test01_beta(self):
        """Check beta applied consistently"""
        Regul10 = TikhonovH1({'Vm': self.Vm, 'gamma': 0.0, 'beta':1e-10})
        error = 0.0
        for mm in self.M:
            r1 = (self.Regulb.grad(mm)).array()
            r10 = (Regul10.grad(mm)).array()
            err = np.linalg.norm(r1 - 1e5*r10)/np.linalg.norm(r1)
            error = max(error, err)
        self.assertTrue(error < 2e-16, error)
        
    def test02_getR(self):
        """Check getter for R"""
        gR = self.Regul.get_R()
        error = 0.0
        for mm in self.M:
            r1 = (self.Regul.hessian(mm.vector())).array()
            r2 = (gR * mm.vector()).array()
            err = np.linalg.norm(r1 - r2)/np.linalg.norm(r1)
            error = max(error, err)
        self.assertTrue(error < 3e-16, error)

    def test02_precond(self):
        """Check preconditioner when beta is defined"""
        prec = self.Regul.get_precond()
        gR = self.Regul.get_R()
        error = 0.0
        for mm in self.M:
            r1 = (prec * mm.vector()).array()
            r2 = (gR * mm.vector()).array()
            err = np.linalg.norm(r1 - r2)/np.linalg.norm(r1)
            error = max(error, err)
        self.assertTrue(error < 1e-16, error)

    def test02_precond2(self):
        """Check preconditioner when beta is defined"""
        prec = self.Regulg.get_precond()
        gR = self.Regulg.get_R()
        gM = self.Regulg.M
        gRM = gR + 1e-14*gM
        error = 0.0
        for mm in self.M:
            r1 = (prec * mm.vector()).array()
            r2 = (gRM * mm.vector()).array()
            err = np.linalg.norm(r1 - r2)/np.linalg.norm(r1)
            error = max(error, err)
        self.assertTrue(error < 1e-16, error)


    def test03_costnull(self):
        """Check null space of regularization"""
        self.m.vector()[:] = np.ones(self.lenm)
        cost = self.Regulg.cost(self.m) 
        self.assertTrue(cost < 1e-16, cost)

    def test03_costposit(self):
        """Check cost is nonnegative"""
        mincost = 1.0
        for mm in self.M:
            cost = self.Regulg.cost(mm)
            mincost = min(mincost, cost)
        self.assertTrue(mincost > 0.0, mincost)
        
    def test04_grad(self):
        """Check cost and gradient are consistent"""
        error = 0.0
        h = 1e-5
        for mm in self.M:
            grad = self.Regul.grad(mm)
            mm_arr = mm.vector().array()
            for dm in self.M:
                dm_arr = dm.vector().array()
                dm_arr /= np.linalg.norm(dm_arr)
                gradxdm = np.dot(grad.array(), dm_arr)
                self.m.vector()[:] = mm_arr + h*dm_arr
                cost1 = self.Regul.cost(self.m)
                self.m.vector()[:] = mm_arr - h*dm_arr
                cost2 = self.Regul.cost(self.m)
                gradxdm_fd = (cost1-cost2) / (2.*h)
                err = abs(gradxdm - gradxdm_fd) / abs(gradxdm)
                error = max(error, err)
        self.assertTrue(error < 2e-7, error)

    def test05_hess(self):
        """Check gradient and Hessian do the same"""
        error = 0.0
        for mm in self.M:
            gradm = self.Regul.grad(mm).array()
            hessm = self.Regul.hessian(mm.vector()-self.Regul.m0.vector())\
            .array()
            err = np.linalg.norm(gradm-hessm)/np.linalg.norm(gradm)
            error = max(error, err)
        self.assertTrue(error < 1e-16, error)
        
    def test06_hess(self):
        """Check gradient and hessian are consistent"""
        error = 0.0
        h = 1e-5
        for mm in self.M:
            for dm in self.M:
                Hdm = self.Regul.hessian(dm.vector()).array()
                self.m.vector()[:] = mm.vector().array() + \
                h*dm.vector().array()
                G1dm = self.Regul.grad(self.m).array()
                self.m.vector()[:] = mm.vector().array() - \
                h*dm.vector().array()
                G2dm = self.Regul.grad(self.m).array()
                HFDdm = (G1dm-G2dm)/(2*h)
                err = np.linalg.norm(HFDdm-Hdm)/np.linalg.norm(Hdm)
                error = max(error, err)
        self.assertTrue(error < 1e-8, error)
