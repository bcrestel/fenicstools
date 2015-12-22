from dolfin import LinearOperator, Function
from miscfenics import setfct

class ObjectiveAcoustic(LinearOperator):
    """Computes data misfit, gradient and Hessian evaluation for the seismic
inverse problem using acoustic wave data"""

    # CONSTRUCTORS:
    def __init__(self, acousticwavePDE):
        """default constructor"""
        self.PDE = acousticwavePDE
        self.MG = Function(self.PDE.Vl)
        self.srchdir = Function(self.PDE.Vl)
        LinearOperator.__init__(self, self.MG.vector(), self.MG.vector())
        self.ObsOp = None   # Observation operator
        self.dd = None  # observations


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
        self.solfwd, tmp = self.PDE.solve()
        if cost == True:
            assert not self.dd == None, "Provide observations"
            sol0, tmp = self.ObsOp.obs(self.solfwd[0][0])
            sollast, tmp = self.ObsOp.obs(self.solfwd[-1][0])
            self.misfit = .5*(self.ObsOp.costfc(sol0, self.dd[0]) + \
            self.ObsOp.costfc(sollast, self.dd[-1]))
            for pp, dd in zip(self.solfwd[1:-1], self.dd[1:-1]):
                solii, tmp = self.ObsOp.obs(pp[0])
                self.misfit += self.ObsOp.costfc(solii, dd)
            self.misfit *= self.PDE.Dt

    def solvefwd_cost(self):    self.solvefwd(True)


    # ADJOINT PROBLEM + GRAD:
    def solveadj(self, grad=False):
        self.PDE.set_adj()
        #TODO

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

