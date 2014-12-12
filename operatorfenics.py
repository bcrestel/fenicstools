import abc
import numpy as np
import matplotlib.pyplot as plt

from dolfin import *
from Plotfenics import Plotfenics
set_log_active(False)

class OperatorA:
    """
    Defines a linear operator A(m) in the Fenics module
    Methods:
        A = operator A(m)
        m = parameter
    """
    __metaclass__ = abc.ABCMeta

    # Instantiation
    def __init__(self, V, Vm, bc, Data=[], eigensolver_param=[]):
        # parameter & bc
        self.m = Function(Vm)
        self.mupdate = 0
        self.bc = bc
        # Define test and trial functions
        self.trial = TrialFunction(V)
        self.test = TestFunction(V)
        self.uplot = Function(V)
        self.dimA = len((self.uplot).vector().array())
        # Add pb specific data
        self.Data = Data
        # Plots
        self.Plots = Plotfenics()
        self.makeplots = False
        # Define weak form to assemble A
        self._wkforma()
        # Assemble PDE operator A 
        self.update_A()
        # Set up eigensolver
        self.Apetsc = PETScMatrix()
        self.eigensolver = SLEPcEigenSolver(self.Apetsc)
        for key in eigensolver_param:
            self.eigensolver.parameters[key] = eigensolver_param[key]

    @abc.abstractmethod
    def _set_Data(self, Data):
        self.Data = Data

    @abc.abstractmethod
    def _wkforma(self):
        self.a = []

    def set_plotdir(self, newdir):
        self.Plots.set_outdir(newdir)
        self.makeplots = True

    # Update param
    def update_plotsname(self):
        self.Plots.set_varname('eig{0}'.format(self.mupdate))

    def update_Data(self, Data):
        self._set_Data(Data)
        self.update_A()

    def update_m(self, m):
        self.m.assign(m)
        self.mupdate += 1
        self.update_A()

    def update_A(self):
        self.A = assemble(self.a)
        (self.bc).apply(self.A)
        # Reset eigenvalue param
        self.nbeigenpairs = 0
        self.realeigenpairs = []
        self.cplxeigenpairs = []
        # Update Plot name
        self.update_plotsname()
        
    # Linear Algebra
    def compute_eigenpair(self, n=0):
        (self.Apetsc).assign(self.A)
        if n == 0:  (self.eigensolver).solve()
        else:   (self.eigensolver).solve(n)
        self.nbeigenpairs = (self.eigensolver).get_number_converged()

        realeigvalues = []
        realeigvectors = []
        cplxeigvalues = []
        cplxeigvectors = []
        [(realeigvalues.append((self.eigensolver).get_eigenpair(ii)[0]),\
        realeigvectors.append((self.eigensolver).get_eigenpair(ii)[1]),\
        cplxeigvalues.append((self.eigensolver).get_eigenpair(ii)[2]),\
        cplxeigvectors.append((self.eigensolver).get_eigenpair(ii)[3])) \
        for ii in range(self.nbeigenpairs)]
        self.realeigenpairs = [realeigvalues, realeigvectors]
        self.cplxeigenpairs = [cplxeigvalues, cplxeigvectors]

    def plot_eigenpairs(self, indices=[]):
        assert self.makeplots, "Need to define output directory"
        # Compute eigenpairs if not done already
        if self.nbeigenpairs == 0:
            if indices == []:   indices = range(self.dimA)
            elif isinstance(indices, int):  indices = [indices]
            self.compute_eigenpair(max(indices))
        # Plot Eigenvectors
        for ii in indices:
            (self.uplot).vector()[:] = (self.realeigenpairs)[1][ii]
            (self.Plots).plot_vtk(uplot, ii)
        (self.Plots).gather_vtkplots()
        # Plot Eigenvalues
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot((self.eigenpairs)[0], 'o')
        fig.suptitle('Eigenvalues for m{0}'.format(self.mupdate))
        fig.savefig(self.Outdir+self.Plots.varname+'.eps') 


###########################################################
# Derived Classes
###########################################################
class OperatorA_Mass(OperatorA):
    """
    Operator A for Mass matrix
    Derived from class OperatorA
    """
    def _set_Data(self, Data):
        self.Data = Data

    def _wkforma(self):
        self.a = inner(self.trial, self.test)*dx 


###########################################################
class OperatorA_Helmholtz(OperatorA):
    """
    Operator A for Helmholtz equation
    Derived from class OperatorA
    """
    def _set_Data(self, Data):
        if not Data.has_key('k'):   raise WrongKeyInInputDataError
        self.Data = Data

    def _wkforma(self):
        kk = (self.Data)['k']
        self.a = inner(nabla_grad(self.trial), nabla_grad(self.test))*dx -\
        inner(kk**2*(self.m)*(self.trial), self.test)*dx

    def update_plotsname(self):
        kf = round((self.Data['k'])*10./(2*np.pi), 1)
        self.Plots.set_varname('eig{0}k{1}'.format(self.mupdate, int(kf)))
