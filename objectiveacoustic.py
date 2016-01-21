from dolfin import LinearOperator, Function, TestFunction,\
assemble, inner, nabla_grad, dx
from miscfenics import setfct, isequal

class ObjectiveAcoustic(LinearOperator):
    """Computes data misfit, gradient and Hessian evaluation for the seismic
inverse problem using acoustic wave data"""

    # CONSTRUCTORS:
    def __init__(self, acousticwavePDE):
        """default constructor"""
        self.PDE = acousticwavePDE
        self.fwdsource = self.PDE.ftime
        self.MG = Function(self.PDE.Vl)
        self.MGv = self.MG.vector()
        self.srchdir = Function(self.PDE.Vl)
        LinearOperator.__init__(self, self.MG.vector(), self.MG.vector())
        self.ObsOp = None   # Observation operator
        self.dd = None  # observations
        self.mtest = TestFunction(self.PDE.Vl)
        self.p, self.v = Function(self.PDE.V), Function(self.PDE.V)
        self.wkformgrad = inner(self.mtest*nabla_grad(self.p), nabla_grad(self.v))*dx


    def copy(self):
        """(hard) copy constructor"""
        newobj = self.__class__(self.PDE.copy())
        setfct(newobj.MG, self.MG)
        setfct(newobj.srchdir, self.srchdir)
        newobj.ObsOp = self.ObsOp
        return newobj


    # FORWARD PROBLEM + COST:
    def solvefwd(self, cost=False):
        self.PDE.set_fwd()
        self.PDE.ftime = self.fwdsource
        self.solfwd,_ = self.PDE.solve()
        if cost:
            assert not self.dd == None, "Provide observations"
            self.Bp = np.zeros(self.dd.shape)
            self.times = np.zeros(self.dd.shape[1])
            for index, sol in enumerate(self.solfwd):
                setfct(self.p, sol[0])
                self.Bp[:,index] = self.ObsOp.obs(self.p)
                self.times[index] = sol[1]
            self.misfit = self.ObsOp.costfct(self.Bp, self.dd, self.times)

    def solvefwd_cost(self):    self.solvefwd(True)


    # ADJOINT PROBLEM + GRAD:
    def solveadj(self, grad=False):
        self.PDE.set_adj()
        self.ObsOp.assemble_rhsadj(self.Bp, self.dd, self.times, self.PDE.bc)
        self.PDE.ftime = self.ObsOp.ftimeadj
        self.soladj,_ = self.PDE.solve()
        if grad:
            self.MGv.zero()
            factors = np.ones(self.ftimes.size)
            factors[0], factors[-1] = 0.5, 0.5
            for fwd, adj, fact in zip(self.solfwd, reversed(self.soladj), factors):
                ttf, tta = fwd[1], adj[1]
                assert isequal(ttf, tta, 1e-14), "Check time steps in fwd and adj"
                setfct(self.p, fwd[0])
                setfct(self.v, adj[0])
                self.MGv.axpy(fact, assemble(self.wkformgrad))
            #TODO: add boundary term for abs bc

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
    def update_PDE(self, parameters_m): self.PDE.update(parameters_m)
    def update_m(self, lam):    self.PDE.update({'lam':lam})
    def set_abc(self, mesh, class_bc_abc, lumpD):  
        self.PDE.set_abc(mesh, class_bc_abc, lumpD)
    def backup_m(self): self.lam_bkup = self.getmarray()
    def setsrcterm(self, ftime):    self.PDE.ftime = ftime


    # GETTERS:
    def getmcopyarray(self):    return self.lam_bkup
    def getmarray(self):    return self.PDE.lam.vector().array()
    def getMGarray(self):   return self.MG.vector().array()

