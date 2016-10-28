import abc
import sys
from os.path import splitext
import numpy as np

from dolfin import TrialFunction, TestFunction, Function, Vector, \
PETScKrylovSolver, LUSolver, set_log_active, LinearOperator, Expression, \
assemble, inner, nabla_grad, dx, MPI 
from exceptionsfenics import WrongInstanceError
from plotfenics import PlotFenics
from fenicstools.optimsolver import compute_searchdirection, bcktrcklinesearch


class ObjectiveFunctional(LinearOperator):
    """
    Provides data misfit, gradient and Hessian information for the data misfit
    part of a time-independent symmetric inverse problem.
    """
    __metaclass__ = abc.ABCMeta

    # Instantiation
    def __init__(self, V, Vm, bc, bcadj, \
    RHSinput=[], ObsOp=[], UD=[], Regul=[], Data=[], plot=False, \
    mycomm=None):
        # Define test, trial and all other functions
        self.trial = TrialFunction(V)
        self.test = TestFunction(V)
        self.mtrial = TrialFunction(Vm)
        self.mtest = TestFunction(Vm)
        self.rhs = Function(V)
        self.m = Function(Vm)
        self.mcopy = Function(Vm)
        self.srchdir = Function(Vm)
        self.delta_m = Function(Vm)
        self.MG = Function(Vm)
        self.MGv = self.MG.vector()
        self.Grad = Function(Vm)
        self.Gradnorm = 0.0
        self.lenm = len(self.m.vector().array())
        self.u = Function(V)
        self.ud = Function(V)
        self.diff = Function(V)
        self.p = Function(V)
        # Define weak forms to assemble A, C and E
        self._wkforma()
        self._wkformc()
        self._wkforme()
        # Store other info:
        self.ObsOp = ObsOp
        self.UD = UD
        self.reset()    # Initialize U, C and E to []
        self.Data = Data
        self.GN = 1.0   # GN = 0.0 => GN Hessian; = 1.0 => full Hessian
        # Operators and bc
        LinearOperator.__init__(self, self.delta_m.vector(), \
        self.delta_m.vector()) 
        self.bc = bc
        self.bcadj = bcadj
        self._assemble_solverM(Vm)
        self.assemble_A()
        self.assemble_RHS(RHSinput)
        self.Regul = Regul
        if Regul != []:
            self.PD = self.Regul.isPD()
        # Counters, tolerances and others
        self.nbPDEsolves = 0    # Updated when solve_A called
        self.nbfwdsolves = 0    # Counter for plots
        self.nbadjsolves = 0    # Counter for plots
        self._set_plots(plot)
        # MPI:
        self.mycomm = mycomm

    def copy(self):
        """Define a copy method"""
        V = self.trial.function_space()
        Vm = self.mtrial.function_space()
        newobj = self.__class__(V, Vm, self.bc, self.bcadj, [], self.ObsOp, \
        self.UD, self.Regul, self.Data, False)
        newobj.RHS = self.RHS
        newobj.update_m(self.m)
        return newobj

    def mult(self, mhat, y):
        """mult(self, mhat, y): do y = Hessian * mhat
        member self.GN sets full Hessian (=1.0) or GN Hessian (=0.0)"""
        N = self.Nbsrc # Number of sources
        y[:] = np.zeros(self.lenm)
        for C, E in zip(self.C, self.E):
            # Solve for uhat
            C.transpmult(mhat, self.rhs.vector())
            self.bcadj.apply(self.rhs.vector())
            self.solve_A(self.u.vector(), -self.rhs.vector())
            # Solve for phat
            E.transpmult(mhat, self.rhs.vector())
            Etmhat = self.rhs.vector().array()
            self.rhs.vector().axpy(1.0, self.ObsOp.incradj(self.u))
            self.bcadj.apply(self.rhs.vector())
            self.solve_A(self.p.vector(), -self.rhs.vector())
            # Compute Hessian*x:
            y.axpy(1.0/N, C * self.p.vector())
            y.axpy(self.GN/N, E * self.u.vector())
        y.axpy(1.0, self.Regul.hessian(mhat))

    # Getters
    def getm(self): return self.m
    def getmarray(self):    return self.m.vector().array()
    def getmcopyarray(self):    return self.mcopy.vector().array()
    def getVm(self):    return self.mtrial.function_space()
    def getMGarray(self):   return self.MG.vector().array()
    def getMGvec(self):   return self.MGv
    def getGradarray(self):   return self.Grad.vector().array()
    def getGradnorm(self):  return self.Gradnorm
    def getsrchdirarray(self):    return self.srchdir.vector().array()
    def getsrchdirvec(self):    return self.srchdir.vector()
    def getsrchdirnorm(self):
        return np.sqrt((self.MM*self.getsrchdirvec()).inner(self.getsrchdirvec()))
    def getgradxdir(self): return self.gradxdir
    def getcost(self):  return self.cost, self.misfit, self.regul
    def getprecond(self):
        return self.Regul.getprecond()
#        Prec = PETScKrylovSolver("richardson", "amg")
#        Prec.parameters["maximum_iterations"] = 1
#        Prec.parameters["error_on_nonconvergence"] = False
#        Prec.parameters["nonzero_initial_guess"] = False
#        Prec.set_operator(self.Regul.get_precond())
#        return Prec
    def getMass(self):    return self.MM

    # Setters
    def setsrchdir(self, arr):  self.srchdir.vector()[:] = arr
    def setgradxdir(self, valueloc):   
        """Sum all local results for Grad . Srch_dir"""
        try:
            valueglob = MPI.sum(self.mycomm, valueloc)
        except:
            valueglob = valueloc
        self.gradxdir = valueglob

    # Solve
    def solvefwd(self, cost=False):
        """Solve fwd operators for given RHS"""
        self.nbfwdsolves += 1
        #if self.ObsOp.noise:    self.noise = 0.0
        if self.plot:
            self.plotu = PlotFenics(self.plotoutdir)
            self.plotu.set_varname('u{0}'.format(self.nbfwdsolves))
        if cost:    self.misfit = 0.0
        for ii, rhs in enumerate(self.RHS):
            self.solve_A(self.u.vector(), rhs)
            if self.plot:   self.plotu.plot_vtk(self.u, ii)
            u_obs, noiselevel = self.ObsOp.obs(self.u)
            self.U.append(u_obs)
            #if self.ObsOp.noise:    self.noise += noiselevel
            if cost:
                self.misfit += self.ObsOp.costfct(u_obs, self.UD[ii])
            self.C.append(assemble(self.c))
        if cost:
            self.misfit /= len(self.U)
            self.regul = self.Regul.cost(self.m)
            self.cost = self.misfit + self.regul
#        if self.ObsOp.noise and self.myrank == 0:
#            print 'Total noise in data misfit={:.5e}\n'.\
#            format(self.noise*.5/len(self.U))
#            self.ObsOp.noise = False    # Safety
        if self.plot:   self.plotu.gather_vtkplots()

    def solvefwd_cost(self):
        """Solve fwd operators for given RHS and compute cost fct"""
        self.solvefwd(True)

    def solveadj(self, grad=False):
        """Solve adj operators"""
        self.nbadjsolves += 1
        if self.plot:
            self.plotp = PlotFenics(self.plotoutdir)
            self.plotp.set_varname('p{0}'.format(self.nbadjsolves))
        self.Nbsrc = len(self.UD)
        if grad:    self.MG.vector()[:] = np.zeros(self.lenm)
        for ii, C in enumerate(self.C):
            self.ObsOp.assemble_rhsadj(self.U[ii], self.UD[ii], \
            self.rhs, self.bcadj)
            self.solve_A(self.p.vector(), self.rhs.vector())
            if self.plot:   self.plotp.plot_vtk(self.p, ii)
            self.E.append(assemble(self.e))
            if grad:    self.MG.vector().axpy(1.0/self.Nbsrc, \
                        C * self.p.vector())
        if grad:
            self.MG.vector().axpy(1.0, self.Regul.grad(self.m))
            self.solverM.solve(self.Grad.vector(), self.MG.vector())
            self.Gradnorm = np.sqrt(self.Grad.vector().inner(self.MG.vector()))
        if self.plot:   self.plotp.gather_vtkplots()

    def solveadj_constructgrad(self):
        """Solve adj operators and assemble gradient"""
        self.solveadj(True)

    # Assembler
    def assemble_A(self):
        """Assemble operator A(m)"""
        self.A = assemble(self.a)
        self.bc.apply(self.A)
        self.set_solver()

    def solve_A(self, b, f):
        """Solve system of the form A.b = f, 
        with b and f in form to be used in solver."""
        self.solver.solve(b, f)
        self.nbPDEsolves += 1

    def assemble_RHS(self, RHSin):
        """Assemble RHS for fwd solve"""
        if RHSin == []: self.RHS = None
        else:
            self.RHS = []
            for rhs in RHSin:
                if isinstance(rhs, Expression):
                    L = rhs*self.test*dx
                    b = assemble(L)
                    self.bc.apply(b)
                    self.RHS.append(b)
                else:   raise WrongInstanceError("rhs should be Expression")

    def _assemble_solverM(self, Vm):
        self.MM = assemble(inner(self.mtrial, self.mtest)*dx)
        self.solverM = LUSolver()
        self.solverM.parameters['reuse_factorization'] = True
        self.solverM.parameters['symmetric'] = True
        self.solverM.set_operator(self.MM)

    def _set_plots(self, plot):
        self.plot = plot
        if self.plot:
            filename, ext = splitext(sys.argv[0])
            self.plotoutdir = filename + '/Plots/'
            self.plotvarm = PlotFenics(self.plotoutdir)
            self.plotvarm.set_varname('m')

    def plotm(self, index):
        if self.plot:   self.plotvarm.plot_vtk(self.m, index)

    def gatherm(self):
        if self.plot:   self.plotvarm.gather_vtkplots()

    # Update param
    def update_Data(self, Data):
        """Update Data member"""
        self.Data = Data
        self.assemble_A()
        self.reset()

    def update_m(self, m):
        """Update values of parameter m"""
        if isinstance(m, np.ndarray):
            self.m.vector()[:] = m
        elif isinstance(m, Function):
            self.m.assign(m)
        elif isinstance(m, float):
            self.m.vector()[:] = m
        elif isinstance(m, int):
            self.m.vector()[:] = float(m)
        else:   raise WrongInstanceError('Format for m not accepted')
        self.assemble_A()
        self.reset()

    def backup_m(self):
        self.mcopy.assign(self.m)

    def restore_m(self):
        self.update_m(self.mcopy)

    def reset(self):
        """Reset U, C and E"""
        self.U = []
        self.C = []
        self.E = []

    def set_solver(self):
        """Reset solver for fwd operator"""
        self.solver = LUSolver()
        self.solver.parameters['reuse_factorization'] = True
        self.solver.set_operator(self.A)

    def addPDEcount(self, increment=1):
        """Increase 'nbPDEsolves' by 'increment'"""
        self.nbPDEsolves += increment

    def resetPDEsolves(self):
        self.nbPDEsolves = 0

    # Additional methods for compatibility with CG solver:
    def init_vector(self, x, dim):
        """Initialize vector x to be compatible with parameter
         Does not work in dolfin 1.3.0"""
        self.MM.init_vector(x, 0)

    def init_vector130(self):
        """Initialize vector x to be compatible with parameter"""
        return Vector(Function(self.mcopy.function_space()).vector())

    # Abstract methods
    @abc.abstractmethod
    def _wkforma(self): self.a = []

    @abc.abstractmethod
    def _wkformc(self): self.c = []

    @abc.abstractmethod
    def _wkforme(self): self.e = []


    def inversion(self, initial_medium, target_medium, mpicomm, \
    parameters_in=[], myplot=None):
        """ solve inverse problem with that objective function """

        parameters = {'tolgrad':1e-10, 'tolcost':1e-14, 'maxnbNewtiter':50}
        parameters.update(parameters_in)
        maxnbNewtiter = parameters['maxnbNewtiter']
        tolgrad = parameters['tolgrad']
        tolcost = parameters['tolcost']
        mpirank = MPI.rank(mpicomm)

        self.update_m(initial_medium)
        self._plotm(myplot, 'init')

        if mpirank == 0:
            print '\t{:12s} {:10s} {:12s} {:12s} {:12s} {:10s} \t{:10s} {:12s} {:12s}'.format(\
            'iter', 'cost', 'misfit', 'reg', '|G|', 'medmisf', 'a_ls', 'tol_cg', 'n_cg')
        dtruenorm = np.sqrt(target_medium.vector().\
        inner(self.MM*target_medium.vector()))

        self.solvefwd_cost()
        for it in xrange(maxnbNewtiter):
            self.solveadj_constructgrad()   # compute gradient

            if it == 0:   gradnorm0 = self.Gradnorm
            diff = self.m.vector() - target_medium.vector()
            medmisfit = np.sqrt(diff.inner(self.MM*diff))
            if mpirank == 0:
                print '{:12d} {:12.4e} {:12.2e} {:12.2e} {:11.4e} {:10.2e} ({:4.2f})'.\
                format(it, self.cost, self.misfit, self.regul, \
                self.Gradnorm, medmisfit, medmisfit/dtruenorm),
            self._plotm(myplot, str(it))
            self._plotgrad(myplot, str(it))

            if self.Gradnorm < gradnorm0*tolgrad or self.Gradnorm < 1e-12:
                if mpirank == 0:
                    print '\nGradient sufficiently reduced -- optimization stopped'
                break

            # Compute search direction:
            tolcg = min(0.5, np.sqrt(self.Gradnorm/gradnorm0))
            self.assemble_hessian() # for regularization
            cgiter, cgres, cgid, tolcg = compute_searchdirection(self, 'Newt', tolcg)
            if self.PD: self.regularization.compute_dw(self.srchdir)
            self._plotsrchdir(myplot, str(it))

            # Line search:
            cost_old = self.cost
            statusLS, LScount, alpha = bcktrcklinesearch(self, 12)
            if mpirank == 0:
                print '{:11.3f} {:12.2e} {:10d}'.format(alpha, tolcg, cgiter)
            if self.PD: self.regularization.update_w(alpha, not mpirank)

            if np.abs(self.cost-cost_old)/np.abs(cost_old) < tolcost:
                if mpirank == 0:
                    print 'Cost function stagnates -- optimization stopped'
                break


    def assemble_hessian(self):
        self.Regul.assemble_hessian(self.m)

    def _plotm(self, myplot, index):
        """ plot media during inversion """
        if not myplot == None:
            myplot.set_varname('m'+index)
            myplot.plot_vtk(self.m)

    def _plotgrad(self, myplot, index):
        """ plot grad during inversion """
        if not myplot == None:
            myplot.set_varname('Grad_m'+index)
            myplot.plot_vtk(self.Grad)

    def _plotsrchdir(self, myplot, index):
        """ plot srchdir during inversion """
        if not myplot == None:
            myplot.set_varname('srchdir_m'+index)
            myplot.plot_vtk(self.srchdir)

###########################################################
# Derived Classes
###########################################################

class ObjFctalElliptic(ObjectiveFunctional):
    """
    Operator for elliptic equation div (m grad u)
    <m grad u, grad v>
    """
    def _wkforma(self):
        self.a = inner(self.m*nabla_grad(self.trial), nabla_grad(self.test))*dx

    def _wkformc(self):
        self.c = inner(self.mtest*nabla_grad(self.u), nabla_grad(self.trial))*dx

    def _wkforme(self):
        self.e = inner(self.mtest*nabla_grad(self.p), nabla_grad(self.trial))*dx

#class OperatorHelmholtz(OperatorPDE):
#    """
#    Operator for Helmholtz equation
#    <grad u, grad v> - k^2 m u v
#    """
#    def _wkforma(self):
#        kk = self.Data['k']
#        self.a = inner(nabla_grad(self.trial), nabla_grad(self.test))*dx -\
#        inner(kk**2*(self.m)*(self.trial), self.test)*dx


