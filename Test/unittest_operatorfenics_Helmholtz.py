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

    def test99_assemble(self):
        """Check diff way to assemble syst are consistent"""
        mesh = UnitSquareMesh(10,10)
        V = FunctionSpace(mesh, 'Lagrange', 2)
        u0 = Expression('0')
        def u0_bdy(x, on_boundary): return on_boundary
        bc = DirichletBC(V, u0, u0_bdy)
        Data = {'k': 10.0}
        OpEll = OperatorHelmholtz(V, V, bc, Data)
        f = Expression('1')
        OpEll.assemble_A()
        A1 = OpEll.A
        v = TestFunction(V)
        L = f*v*dx
        b1 = assemble(L)
        bc.apply(b1)
        u1 = Function(V)
        solve(A1, u1.vector(), b1)

        A2, b2 = OpEll.assemble_Ab(f)
        u2 = Function(V)
        solve(A2, u2.vector(), b2)
        diffa = A1.array() - A2.array()
        self.assertTrue(max(abs(u1.vector().array() - \
        u2.vector().array())) < 1e-14 and max(abs(u1.vector().array())) > 1e-3)


if __name__ == '__main__':
    unittest.main()
