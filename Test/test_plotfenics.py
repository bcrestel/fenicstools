import unittest
import numpy as np
import os
from os.path import isdir, isfile

from dolfin import *
from plotfenics import PlotFenics
from exceptionsfenics import NoOutdirError, NoVarnameError, NoIndicesError
set_log_active(False)

MYOUTDIR = 'Test/Plots/'

class TestPlots(unittest.TestCase):
    os.system('rm -rf {0}'.format(MYOUTDIR))
    
    def setUp(self):
        mesh = UnitSquareMesh(5,5,'crossed')
        V = FunctionSpace(mesh, 'Lagrange', 1)
        Vm = FunctionSpace(mesh, 'Lagrange', 1)
        self.m = Function(Vm)
        self.lenm = len(self.m.vector().array())
        u0 = Expression('0')
        def u0_boundary(x, on_boundary):
            return False
        bc = DirichletBC(V, u0, u0_boundary)
        self.myoutdir = MYOUTDIR
        self.indexlist = [1,3,5,6,7]

    def test00_inst(self):
        """Default instantiation"""
        myplot = PlotFenics()

    def test00_except(self):
        """Raise exception when dir not defined"""
        myplot = PlotFenics()
        self.assertRaises(NoOutdirError, myplot.plot_vtk, self.m)

    def test00_except2(self):
        """Raise exception when dir not defined"""
        myplot = PlotFenics()
        self.assertRaises(NoOutdirError, myplot.gather_vtkplots)

    def test01(self):
        """Instantiate class when dir does not exist"""
        if isdir(self.myoutdir):
            os.system('rm -rf {0}'.format(self.myoutdir))
        myplot = PlotFenics(self.myoutdir)
        self.assertTrue(isdir(self.myoutdir))

    def test01_alt(self):
        """Set directory"""
        if isdir(self.myoutdir):
            os.system('rm -rf {0}'.format(self.myoutdir))
        myplot = PlotFenics()
        myplot.set_outdir(self.myoutdir)
        self.assertTrue(isdir(self.myoutdir))

    def test02(self):
        """Instantiate class when dir exists"""
        myplot = PlotFenics(self.myoutdir)
        myplot.set_outdir(self.myoutdir + 'Test1/')
        self.assertTrue(myplot.outdir == self.myoutdir + 'Test1/')

    def test02_alt(self):
        """Need varname defined to plot"""
        myplot = PlotFenics(self.myoutdir)
        self.m.vector()[:] = np.ones(self.lenm)
        self.assertRaises(NoVarnameError, myplot.plot_vtk, self.m)

    def test03(self):
        """Need index list defined to gather"""
        myplot = PlotFenics(self.myoutdir)
        myplot.set_varname('testm')
        self.assertRaises(NoIndicesError, myplot.gather_vtkplots)

    def test04(self):
        """Plot parameter m"""
        myplot = PlotFenics(self.myoutdir)
        myplot.set_varname('testm')
        self.m.vector()[:] = 20 * np.ones(self.lenm, dtype='float')
        myplot.plot_vtk(self.m, 20)
        self.assertTrue(isfile(self.myoutdir+'testm_20.pvd') and \
        isfile(self.myoutdir+'testm_20000000.vtu'))

    def test05(self):
        """Indices list properly created"""
        myplot = PlotFenics(self.myoutdir)
        myplot.set_varname('testm')
        for ii in self.indexlist:
            self.m.vector()[:] = ii * np.ones(self.lenm, dtype='float')
            myplot.plot_vtk(self.m, ii)
        self.assertTrue(myplot.indices == self.indexlist)

    def test05_bis(self):
        """All files are still here"""
        self.assertTrue(isfile(self.myoutdir+'testm_20.pvd') and \
        isfile(self.myoutdir+'testm_7.pvd'))

    def test06(self):
        """Gather plots"""
        myplot = PlotFenics(self.myoutdir)
        myplot.set_varname('testm')
        myplot.indices = self.indexlist
        myplot.gather_vtkplots()
        self.assertTrue(isfile(self.myoutdir+'testm.pvd'))

    def test07(self):
        """pvd are deleted after gathering them"""
        self.assertFalse(isfile(self.myoutdir+'testm_7.pvd'))

    def test08(self):
        """Fenics_vtu_correction variable"""
        myplot = PlotFenics(self.myoutdir)
        self.assertTrue(isfile(self.myoutdir+'testm_20'+\
        myplot.Fenics_vtu_correction+'.vtu'))

    def test09(self):
        """Indices are reset when varname changed"""
        myplot = PlotFenics(self.myoutdir)
        myplot.set_varname('testm')
        indexlist = [1,3]
        for ii in indexlist:
            self.m.vector()[:] = ii * np.ones(self.lenm, dtype='float')
            myplot.plot_vtk(self.m, ii)
        myplot.set_varname('testm2')
        self.assertTrue((myplot.varname == 'testm2') and \
        (myplot.indices == []))


if __name__ == '__main__':
    unittest.main()
