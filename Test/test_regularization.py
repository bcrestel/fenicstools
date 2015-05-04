import unittest
import numpy as np

from dolfin import *
from regularization import TikhonovH1

class TestTikhonovH1(unittest.TestCase):

    def setUp(self):
        mesh = UnitSquareMesh(5,5,'crossed')
        self.Vm = FunctionSpace(mesh, 'Lagrange', 1)
        self.Regul1 = TikhonovH1({'Vm': self.Vm, 'gamma': 1.0})
        self.M = []
        for ii in range(10):    self.M.append(Function(self.Vm))
        self.lenm = len(self.M[0].vector().array())
        for mm in self.M:
            mm.vector()[:] = np.random.randn(self.lenm)
        self.m = Function(self.Vm)

    def test00_inst(self):
        """Default instantiation"""
        Regul = TikhonovH1({'Vm': self.Vm, 'gamma': 10.})

    def test01_gamma(self):
        """Check gamma applied consistently"""
        Regul10 = TikhonovH1({'Vm': self.Vm, 'gamma': 10.})
        error = 0.0
        for mm in self.M:
            r1 = (self.Regul1.grad(mm)).array()
            r10 = (Regul10.grad(mm)).array()
            err = np.linalg.norm(10.*r1 - r10)/np.linalg.norm(r10)
            error = max(error, err)
        self.assertTrue(error < 1e-16)
        
    def test02_getR(self):
        """Check getter for R"""
        gR = self.Regul1.get_R()
        error = 0.0
        for mm in self.M:
            r1 = (self.Regul1.grad(mm)).array()
            r2 = (gR * mm.vector()).array()
            err = np.linalg.norm(r1 - r2)/np.linalg.norm(r1)
            error = max(error, err)
        self.assertTrue(error < 1e-16)

    def test03_costnull(self):
        """Check null space of regularization"""
        self.m.vector()[:] = np.ones(self.lenm)
        cost = self.Regul1.cost(self.m) 
        self.assertTrue(cost < 1e-16, cost)

    def test03_costposit(self):
        """Check cost is nonnegative"""
        mincost = 1.0
        for mm in self.M:
            cost = self.Regul1.cost(mm)
            mincost = min(mincost, cost)
        self.assertTrue(mincost > 0.0, mincost)
        
    def test04_grad(self):
        """Check cost and gradient are consistent"""
        error = 0.0
        h = 1e-5
        for mm in self.M:
            grad = self.Regul1.grad(mm)
            mm_arr = mm.vector().array()
            for dm in self.M:
                dm_arr = dm.vector().array()
                dm_arr /= np.linalg.norm(dm_arr)
                gradxdm = np.dot(grad.array(), dm_arr)
                self.m.vector()[:] = mm_arr + h*dm_arr
                cost1 = self.Regul1.cost(self.m)
                self.m.vector()[:] = mm_arr - h*dm_arr
                cost2 = self.Regul1.cost(self.m)
                gradxdm_fd = (cost1-cost2) / (2.*h)
                err = abs(gradxdm - gradxdm_fd) / abs(gradxdm)
                error = max(error, err)
        self.assertTrue(error < 1e-7, error)
        
    def test05_hess(self):
        """Check gradient and hessian are consistent"""
        error = 0.0
        h = 1e-5
        for mm in self.M:
            hess = self.Regul1.hessian(mm.vector())
            mm_arr = mm.vector().array()
            for dm in self.M:
                dm_arr = dm.vector().array()
                dm_arr /= np.linalg.norm(dm_arr)
                hessxdm = np.dot(hess.array(), dm_arr)
                self.m.vector()[:] = mm_arr + h*dm_arr
                grad1 = self.Regul1.grad(self.m).array()
                self.m.vector()[:] = mm_arr - h*dm_arr
                grad2 = self.Regul1.grad(self.m).array()
                hessxdm_fd = (grad1-grad2) / (2.*h)
                err = np.linalg.norm(hessxdm - hessxdm_fd) / \
                np.linalg.norm(hessxdm)
                error = max(error, err)
        self.assertTrue(error < 1e-7, error)
