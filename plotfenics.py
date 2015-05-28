import os
from os.path import isdir

from dolfin import File
from exceptionsfenics import *

class PlotFenics:
    """
    Class can be used to plot fenics variables to vtk file
    """
    Fenics_vtu_correction = '000000'

    # Instantiation
    def __init__(self, Outputfolder=None):
        self.outdir = Outputfolder
        self.set_outdir(self.outdir)
        self.indices = []
        self.varname = []

    def set_outdir(self, new_dir_in):
        """set output directory and creates it if needed"""
        if new_dir_in == None:  new_dir = 'Outputs/Plots/'
        else:   new_dir = new_dir_in
        self.outdir = new_dir
        if not isdir(new_dir):  os.makedirs(new_dir)

    def reset_indices(self):
        self.indices = []

    def set_varname(self, varname):
        self.varname = varname
        self.reset_indices()

    def plot_vtk(self, variable, index=0):
        """Plot variable in vtk file given by filename"""
        self._check_outdir()
        self._check_varname()
        fname = self.varname + '_{0}'.format(index)
        File(self.outdir + fname + '.pvd') << variable
        (self.indices).append(index)

    def gather_vtkplots(self):
        """Create pvd file to load all files in paraview"""
        self._check_outdir()
        self._check_indices()
        with open(self.outdir+self.varname+'.pvd', 'w') as fout:
            fout.write('<?xml version="1.0"?>\n<VTKFile type="Collection"' +\
            ' version="0.1">\n\t<Collection>\n')
            for step, ii in enumerate(self.indices):
                myline = ('<DataSet timestep="{0}" part="0" ' +\
                'file="{1}_{2}{3}.vtu" />\n').format(step, self.varname, ii,\
                self.Fenics_vtu_correction)
                fout.write(myline)
            fout.write('\t</Collection>\n</VTKFile>')

    # Check methods
    def _check_outdir(self):
        if self.outdir == []:
            raise NoOutdirError("outdir not defined")

    def _check_varname(self):
        if self.varname == []:
            raise NoVarnameError("varname not defined")

    def _check_indices(self):
        if self.indices == []:
            raise NoIndicesError("No indices defined")
