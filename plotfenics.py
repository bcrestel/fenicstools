import os
from os.path import isdir
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np

from dolfin import File, MPI
from exceptionsfenics import *
from miscfenics import setfct


class PlotFenics:
    """
    Class can be used to plot fenics variables to vtk file
    """
    Fenics_vtu_correction = '000000'

    # Instantiation
    def __init__(self, comm, Outputfolder=None):
        self.mpirank = MPI.rank(comm)
        mpisize = MPI.size(comm)
        if Outputfolder == None:    self.set_outdir('Output/', comm)
        else:   self.set_outdir(Outputfolder, comm)
        if mpisize == 1:    self.extensionvtu = 'vtu'
        else:   self.extensionvtu = 'pvtu'
        self.indices = []
        self.varname = []

    def set_outdir(self, new_dir, comm):
        """set output directory and creates it if needed"""
        if not new_dir[-1] == '/':  new_dir += '/'
        self.outdir = new_dir
        if self.mpirank == 0 and not isdir(new_dir):  os.makedirs(new_dir)
        MPI.barrier(comm)

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

    def plot_timeseries(self, timeseries, varname, index, skip, function):
        """ Plot every 'skip' entry of entry 'index' of a 'timeseries'.
        'function' = dl.Function """
        self.set_varname(varname)
        for ii, sol in enumerate(timeseries):
            if ii%skip == 0:
                setfct(function, sol[index])
                self.plot_vtk(function, ii)
        self.gather_vtkplots()

    def gather_vtkplots(self):
        """Create pvd file to load all files in paraview"""
        self._check_outdir()
        self._check_indices()
        if self.mpirank == 0:
            with open(self.outdir+self.varname+'.pvd', 'w') as fout:
                fout.write('<?xml version="1.0"?>\n<VTKFile type="Collection"' +\
                ' version="0.1">\n\t<Collection>\n')
                for step, ii in enumerate(self.indices):
                    myline = ('<DataSet timestep="{0}" part="0" ' +\
                    'file="{1}_{2}{3}.{4}" />\n').format(step, self.varname, ii,\
                    self.Fenics_vtu_correction, self.extensionvtu)
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


def plotobservations(times, Bp, dd, nbplots=None):
    """ Plot observations Bp along with data dd
    Input:
        times = numpy arrays (time steps)
        Bp, dd = numpy arrays (time-series)
    """
    assert Bp.shape == dd.shape, \
    "Both input arrays should have same dimensions"
    nbobspts, timesteps = Bp.shape
    assert len(times) == timesteps, \
    "Time steps and time-series sizes do not coincide"
    
    if nbplots is None:
        nbplots = nbobspts

    griddim = int(np.ceil(np.sqrt(nbplots)))
    fig = plt.figure()
    fig.set_size_inches(20., 15.)
    fignb = 1
    listplots = [int(ii) for ii in np.linspace(0, nbobspts-1, nbplots)]
    for ii in listplots:
        ax = fig.add_subplot(griddim, griddim, fignb)
        fignb += 1
        ax.plot(times, dd[ii,:], 'k--')
        ax.plot(times, Bp[ii,:], 'b')
        ax.set_title('obs'+str(ii+1))

    return fig
