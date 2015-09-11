import unittest
import numpy as np

from dolfin import UnitSquareMesh, FunctionSpace, TestFunction, TrialFunction, \
assemble, dx, LUSolver, Function

from linalg.lumpedmatrixsolver import LumpedMatrixSolver


class TestLumpedMass(unittest.TestCase):

    def setUp(self):
        mesh = UnitSquareMesh(5, 5, 'crossed')
        self.V = FunctionSpace(mesh, 'Lagrange', 5)
        self.u = Function(self.V)
        self.uM = Function(self.V)
        self.uMdiag = Function(self.V)
        test = TestFunction(self.V)
        trial = TrialFunction(self.V)
        m = test*trial*dx
        self.M = assemble(m)
        self.solver = LUSolver()
        self.solver.parameters['reuse_factorization'] = True
        self.solver.parameters['symmetric'] = True
        self.solver.set_operator(self.M)
        

    def test00(self):
        """ Create a lumped solver """
        myobj = LumpedMatrixSolver(self.V)

    def test01_set(self):
        """ Set operator """
        myobj = LumpedMatrixSolver(self.V)
        myobj.set_operator(self.M)

    def test01_entries(self):
        """ Lump matrix """
        myobj = LumpedMatrixSolver(self.V)
        myobj.set_operator(self.M)
        err = 0.0
        for index, ii in enumerate(self.M.array()):
            err += abs(ii.sum() - myobj.Mdiag[index])
        self.assertTrue(err < index*1e-16)


    def test02(self):
        """ Invert lumped matrix """
        myobj = LumpedMatrixSolver(self.V)
        myobj.set_operator(self.M)
        err = 0.0
        for ii in range(len(myobj.Mdiag.array())):
            err += abs(1./myobj.Mdiag[ii] - myobj.invMdiag[ii])
        self.assertTrue(err < ii*1e-16)


    def test03_basic(self):
        """ solve """
        myobj = LumpedMatrixSolver(self.V)
        myobj.set_operator(self.M)
        myobj.solve(self.uMdiag.vector(), myobj.Mdiag)
        diff = myobj.one.array() - self.uMdiag.vector().array()
        self.assertTrue( np.sqrt((diff**2).sum()) < 1e-14 )


#    def test03(self):
#        """ solve """
#        myobj = LumpedMatrixSolver(self.V)
#        myobj.set_operator(self.M)
#        for ii in range(10):
#            self.u.vector()[:] = np.random.randn(self.V.dim())
#            normu = np.sqrt((self.u.vector().array()**2).sum())
#            b = self.M * self.u.vector()
#            self.solver.solve(self.uM.vector(), b)
#            myobj.solve(self.uMdiag.vector(), b)   
#            diff = self.uM.vector().array() - self.uMdiag.vector().array()
#            diffM = self.uM.vector().array() - self.u.vector().array()
#            diffMdiag = self.u.vector().array() - self.uMdiag.vector().array()
#            print np.sqrt((diffM**2).sum())/normu, \
#            np.sqrt((diffMdiag**2).sum())/normu, np.sqrt((diff**2).sum())/normu
#
#
#    def test04(self):
#        """ solve """
#        mesh = UnitSquareMesh(100, 100, 'crossed')
#        V = FunctionSpace(mesh, 'Lagrange', 1)
#        u = Function(V)
#        uM = Function(V)
#        uMdiag = Function(V)
#        test = TestFunction(V)
#        trial = TrialFunction(V)
#        m = test*trial*dx
#        M = assemble(m)
#        solver = LUSolver()
#        solver.parameters['reuse_factorization'] = True
#        solver.parameters['symmetric'] = True
#        solver.set_operator(M)
#        myobj = LumpedMatrixSolver(V)
#        myobj.set_operator(M)
#        for ii in range(10):
#            u.vector()[:] = np.random.randn(V.dim())
#            normu = np.sqrt((u.vector().array()**2).sum())
#            b = M * u.vector()
#            solver.solve(uM.vector(), b)
#            myobj.solve(uMdiag.vector(), b)   
#            diff = uM.vector().array() - uMdiag.vector().array()
#            diffM = uM.vector().array() - u.vector().array()
#            diffMdiag = u.vector().array() - uMdiag.vector().array()
#            print np.sqrt((diffM**2).sum())/normu, \
#            np.sqrt((diffMdiag**2).sum())/normu, np.sqrt((diff**2).sum())/normu
