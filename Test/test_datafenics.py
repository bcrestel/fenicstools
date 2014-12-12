import unittest
import numpy as np

from dolfin import *
from datafenics import CompData
from exceptionsfenics import WrongInstanceError
set_log_active(False)

class TestCompData(unittest.TestCase):
    def setUp(self):
        mesh = UnitSquareMesh(5,5,'crossed')
        self.Vm = FunctionSpace(mesh, 'Lagrange', 2)
        self.m = Function(self.Vm)
        self.lenm = len(self.m.vector().array())
        self.m.vector()[:] = np.random.randn(self.lenm)
        
    def test01_inst(self):
        """Defaut instantiation"""
        compdata = CompData()

    def test01_inst1(self):
        """Instantiation with m only"""
        compdata = CompData(self.m)

    def test01_inst2(self):
        """Instantiation with m and Vm"""
        compdata = CompData(np.random.randn(self.lenm), self.Vm)

    def test01_instexcep1(self):
        """Exception when Vm not properly defined"""
        marr = np.random.randn(self.lenm)
        self.assertRaises(WrongInstanceError, CompData, marr, self.m)

    def test01_instexcep2(self):
        """Exception when m not properly defined"""
        marr = np.random.randn(self.lenm)
        self.assertRaises(WrongInstanceError, CompData, marr)

    def test01_instexcep3(self):
        """Exception when m not properly defined with Vm"""
        self.assertRaises(WrongInstanceError, CompData, self.m, self.Vm)
        
    def test02_copy(self):
        """Copy object w/ same object m"""
        compdata = CompData(np.random.randn(self.lenm), self.Vm)
        compdata2 = compdata.copy()
        
    def test02_copy2(self):
        """Copy object creates different object"""
        compdata = CompData(np.random.randn(self.lenm), self.Vm)
        compdata2 = compdata.copy()
        self.assertFalse(id(compdata.m) == id(compdata2.m))

    def test02_copy3(self):
        """Copied objects have same entries"""
        compdata = CompData(np.random.randn(self.lenm), self.Vm)
        compdata2 = compdata.copy()
        self.assertTrue(np.array_equal(compdata.m.vector().array(), \
        compdata2.m.vector().array()))
