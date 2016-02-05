import numpy as np

from dolfin import LinearOperator, Function, TestFunction,\
assemble, inner, nabla_grad, dx,\
LUSolver()
from miscfenics import setfct, isequal

class ObjectiveAcoustic(LinearOperator):
    """
    Computes data misfit, gradient and Hessian evaluation for the seismic
    inverse problem using acoustic wave data
    """

    # CONSTRUCTORS:
    def __init__(self, acousticwavePDE):
        """ 
        Input:
            acousticwavePDE should be an instantiation from class AcousticWave
        """
        self.PDE = acousticwavePDE
        self.fwdsource = self.PDE.ftime
        self.MG = Function(self.PDE.Vl)
        self.MGv = self.MG.vector()
        self.Grad = Function(self.PDE.Vl)
        self.Gradv = self.Grad.vector()
        self.srchdir = Function(self.PDE.Vl)
        LinearOperator.__init__(self, self.MG.vector(), self.MG.vector())
        self.obsop = None   # Observation operator
        self.dd = None  # observations
        self.mtest = TestFunction(self.PDE.Vl)
        self.p, self.v = Function(self.PDE.V), Function(self.PDE.V)
        self.wkformgrad = inner(self.mtest*nabla_grad(self.p), nabla_grad(self.v))*dx
        # Mass matrix:
        Mass = assemble(self.PDE.weak_m)
        self.solverM = LUSolver()
        self.solverM.parameters['reuse_factorization'] = True
        self.solverM.parameters['symmetric'] = True
        self.solverM.set_operator(Mass)


    def copy(self):
        """(hard) copy constructor"""
        newobj = self.__class__(self.PDE.copy())
        setfct(newobj.MG, self.MG)
        setfct(newobj.srchdir, self.srchdir)
        newobj.obsop = self.obsop
        return newobj


    # FORWARD PROBLEM + COST:
    def solvefwd(self, cost=False):
        self.PDE.set_fwd()
        self.PDE.ftime = self.fwdsource
        self.solfwd,_ = self.PDE.solve()
        # observations:
        self.Bp = np.zeros((len(self.obsop.PtwiseObs.Points),len(self.solfwd)))
        self.times = np.zeros(len(self.solfwd))
        for index, sol in enumerate(self.solfwd):
            setfct(self.p, sol[0])
            self.Bp[:,index] = self.obsop.obs(self.p)
            self.times[index] = sol[1]
        if cost:
            assert not self.dd == None, "Provide observations"
            self.misfit = self.obsop.costfct(self.Bp, self.dd, self.times)

    def solvefwd_cost(self):    self.solvefwd(True)


    # ADJOINT PROBLEM + GRAD:
    def solveadj(self, grad=False):
        self.PDE.set_adj()
        self.obsop.assemble_rhsadj(self.Bp, self.dd, self.times, self.PDE.bc)
        self.PDE.ftime = self.obsop.ftimeadj
        self.soladj,_ = self.PDE.solve()
        if grad:
            self.MGv.zero()
            factors = np.ones(self.ftimes.size)
            factors[0], factors[-1] = 0.5, 0.5
            #TODO: add boundary term for abs bc
            for fwd, adj, fact in zip(self.solfwd, reversed(self.soladj), factors):
                ttf, tta = fwd[1], adj[1]
                assert isequal(ttf, tta, 1e-14), "Check time steps in fwd and adj"
                setfct(self.p, fwd[0])
                setfct(self.v, adj[0])
                self.MGv.axpy(fact, assemble(self.wkformgrad))
            self.solverM.solve(self.Gradv, self.Mgv)

    def solveadj_constructgrad(self):   self.solveadj(True)


    # HESSIAN:
    def mult(self, mhat, y):
        """
        mult(self, mhat, y): return y = Hessian * mhat
        member self.GN sets full Hessian (=1.0) or GN Hessian (=0.0)
        inputs:
            y, mhat = Function(V).vector()
        """
        #TODO:


    # SETTERS + UPDATE:
    def update_PDE(self, parameters): self.PDE.update(parameters)
    def update_m(self, lam):    self.update_PDE({'lambda':lam})
    def set_abc(self, mesh, class_bc_abc, lumpD):  
        self.PDE.set_abc(mesh, class_bc_abc, lumpD)
    def backup_m(self): self.lam_bkup = self.getmarray()
    def setsrcterm(self, ftime):    self.PDE.ftime = ftime


    # GETTERS:
    def getmcopyarray(self):    return self.lam_bkup
    def getmarray(self):    return self.PDE.lam.vector().array()
    def getMGarray(self):   return self.MG.vector().array()

