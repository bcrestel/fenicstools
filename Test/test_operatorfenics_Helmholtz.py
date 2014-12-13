import unittest
import numpy as np

from dolfin import *
from operatorfenics import OperatorHelmholtz
set_log_active(False)

class TestHelmholtz(unittest.TestCase):
    
    def setUp(self):
        mesh = UnitSquareMesh(1,1,'crossed')
        self.V = FunctionSpace(mesh, 'Lagrange', 1)
        self.Vm = FunctionSpace(mesh, 'Lagrange', 1)
        self.m = Function(self.Vm)
        self.lenm = len(self.m.vector().array())
        u0 = Expression('0')
        def u0_boundary(x, on_boundary):
            return False
        self.bc = DirichletBC(self.V, u0, u0_boundary)
        self.Data = {'k':10}
        self.Am0 = np.array([[1.,0.,-1.,0.,0.],[0.,1.,-1.,0.,0.],\
        [-1.,-1.,4.,-1.,-1.],[0.,0.,-1.,1.,0.],[0.,0.,-1.,0.,1.]])
        
    def test00_inst(self):
        """Instantiate object from class"""
        myclass = OperatorHelmholtz(self.V, self.Vm, self.bc, self.Data)

    def test01_m0(self):
        """m default to zero"""
        myclass = OperatorHelmholtz(self.V, self.Vm, self.bc, self.Data)
        maxm = max(abs(myclass.m.vector().array()))
        self.assertFalse(maxm > 1e-14)

    def test02_Am0(self):
        """A properly assembled and evaluated as an array"""
        myclass = OperatorHelmholtz(self.V, self.Vm, self.bc, self.Data)
        maxAerr = max(abs( (self.Am0-myclass.A.array()).reshape((25,1)) ))
        self.assertFalse(maxAerr > 1e-14)

    def test04_Am1(self):
        """Non diff part of operator scales as expected"""
        myclass = OperatorHelmholtz(self.V, self.Vm, self.bc, self.Data)
        myclass.update_m(np.ones(self.lenm))
        Adiff = self.Am0 - myclass.A.array()
        myclass.update_m(10.*np.ones(self.lenm))
        Adiff2 = self.Am0 - myclass.A.array()
        maxAerr = max(abs( (10.*Adiff - Adiff2).reshape((25,1)) ))
        self.assertFalse(maxAerr > 1e-12)


if __name__ == '__main__':
    unittest.main()
