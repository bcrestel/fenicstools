import unittest
import numpy as np

from dolfin import *
from operatorfenics import OperatorMass
from exceptionsfenics import WrongInstanceError
set_log_active(False)

class TestMass(unittest.TestCase):
    
    def setUp(self):
        mesh = UnitSquareMesh(1,1,'crossed')
        self.V = FunctionSpace(mesh, 'Lagrange', 1)
        self.Vm = FunctionSpace(mesh, 'Lagrange', 1)
        self.m = Function(self.Vm)
        self.lenm = len(self.m.vector().array())
        u0 = Expression('0')
        def u0_boundary(x, on_boundary):    return False
        def u0_boundary2(x, on_boundary):    return on_boundary
        self.bc = DirichletBC(self.V, u0, u0_boundary)
        self.bc2 = DirichletBC(self.V, u0, u0_boundary2)
        self.Am0 = np.array([[4.,1.,2.,1.,0.],[1.,4.,2.,0.,1.],\
        [2.,2.,8.,2.,2.],[1.,0.,2.,4.,1.],[0.,1.,2.,1.,4.]])/48.
        self.Am02 = np.array([[48.,0.,0.,0.,0.],[0.,48.,0.,0.,0.],\
        [2.,2.,8.,2.,2.],[0.,0.,0.,48.,0.],[0.,0.,0.,0.,48.]])/48.
        
    def test00_inst(self):
        """Instantiate an object from class"""
        myclass = OperatorMass(self.V, self.Vm, self.bc)

    def test01_Am0(self):
        """Basic operator properly assembled"""
        myclass = OperatorMass(self.V, self.Vm, self.bc)
        maxAerr = max(abs((self.Am0-myclass.A.array()).reshape((25,1))))
        self.assertTrue(maxAerr < 1e-12)

    def test01_Am02(self):
        """Basic operator properly assembled with Dirichlet bc"""
        myclass = OperatorMass(self.V, self.Vm, self.bc2)
        maxAerr = max(abs((self.Am02-myclass.A.array()).reshape((25,1))))
        self.assertTrue(maxAerr < 1e-12)

    def test02_mupdate(self):
        """Update parameter from Function"""
        myclass = OperatorMass(self.V, self.Vm, self.bc)
        inputarray = np.random.randn(self.lenm)
        self.m.vector()[:] = inputarray
        myclass.update_m(self.m)
        self.assertTrue(np.array_equal(myclass.m.vector().array(), inputarray))

    def test02_mupdate2(self):
        """Update parameter from ndarray"""
        myclass = OperatorMass(self.V, self.Vm, self.bc)
        inputarray = np.random.randn(self.lenm)
        myclass.update_m(inputarray)
        self.assertTrue(np.array_equal(myclass.m.vector().array(), inputarray))

    def test02_mupdate3(self):
        """Raises exception when wrong parameter format used"""
        myclass = OperatorMass(self.V, self.Vm, self.bc)
        self.assertRaises(WrongInstanceError, myclass.update_m, self.m.vector())

    def test99_assemble(self):
        """Check diff way to assemble syst are consistent"""
        mesh = UnitSquareMesh(10,10)
        V = FunctionSpace(mesh, 'Lagrange', 2)
        u0 = Expression('0')
        def u0_bdy(x, on_boundary): return on_boundary
        bc = DirichletBC(V, u0, u0_bdy)
        OpEll = OperatorMass(V, V, bc)
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
