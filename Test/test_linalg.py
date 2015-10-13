import unittest
import numpy as np

from dolfin import UnitSquareMesh, FunctionSpace, TestFunction, TrialFunction, \
assemble, dx, LUSolver, Function, interpolate, Expression, inner, nabla_grad, \
Vector

from linalg.lumpedmatrixsolver import LumpedMatrixSolver, LumpedMatrixSolverS
from linalg.miscroutines import get_diagonal


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
        self.ones = np.ones(self.V.dim())
        

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

    def test10(self):
        """ Create a lumped solver """
        myobj = LumpedMatrixSolverS(self.V)

    def test11_set(self):
        """ Set operator """
        myobj = LumpedMatrixSolverS(self.V)
        myobj.set_operator(self.M)

    def test11_entries(self):
        """ Lump matrix """
        myobj = LumpedMatrixSolverS(self.V)
        myobj.set_operator(self.M)
        Msum = np.dot(self.ones, self.M.array().dot(self.ones))
        err = abs(myobj.Mdiag.array().dot(self.ones) - \
        Msum) / Msum
        self.assertTrue(err < 1e-14, err)


    def test12(self):
        """ Invert lumped matrix """
        myobj = LumpedMatrixSolverS(self.V)
        myobj.set_operator(self.M)
        err = 0.0
        for ii in range(len(myobj.Mdiag.array())):
            err += abs(1./myobj.Mdiag[ii] - myobj.invMdiag[ii])
        self.assertTrue(err < ii*1e-16)


    def test13_basic(self):
        """ solve """
        myobj = LumpedMatrixSolverS(self.V)
        myobj.set_operator(self.M)
        myobj.solve(self.uMdiag.vector(), myobj.Mdiag)
        diff = myobj.one.array() - self.uMdiag.vector().array()
        self.assertTrue( np.sqrt((diff**2).sum()) < 1e-14 )

###################################################################
###################################################################
###################################################################
class TestGetDiagonal(unittest.TestCase):

    def setUp(self):
        mesh = UnitSquareMesh(5, 5, 'crossed')
        self.V = FunctionSpace(mesh, 'Lagrange', 2)
        u = interpolate(Expression('1 + 7*(pow(pow(x[0] - 0.5,2) +' + \
        ' pow(x[1] - 0.5,2),0.5) > 0.2)'), self.V)
        test = TestFunction(self.V)
        trial = TrialFunction(self.V)
        m = test*trial*dx
        self.M = assemble(m)
        k = inner(u*nabla_grad(test), nabla_grad(trial))*dx
        self.K = assemble(k)

    def test00(self):
        """ Get diagonal of M """
        md = get_diagonal(self.M)

    def test10(self):
        """ Get diagonal of K """
        kd = get_diagonal(self.K)

    def test01(self):
        """ Check diagonal content of M """
        md = get_diagonal(self.M)
        bi = Function(self.V)
        for index, ii in enumerate(md):
            b = bi.vector().copy()
            b[index] = 1.0
            Mb = (self.M*b).inner(b)
            self.assertTrue(abs(ii - Mb)/abs(Mb) < 1e-16)

    def test11(self):
        """ Check diagonal content of K """
        kd = get_diagonal(self.K)
        bi = Function(self.V)
        for index, ii in enumerate(kd):
            b = bi.vector().copy()
            b[index] = 1.0
            Kb = (self.K*b).inner(b)
            self.assertTrue(abs(ii - Kb)/abs(Kb) < 1e-16)
