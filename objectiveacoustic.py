from dolfin import LinearOperator
from acousticwave import AcousticWave
from miscfenics import setfct

class ObjectiveAcoustic(LinearOperator):
    """Computes data misfit, gradient and Hessian evaluation for the seismic
inverse problem using acoustic wave data"""

    # CONSTRUCTORS:
    def __init__(self, functionspaces):
        """default constructor"""
        self.acousticPDE = AcousticWave(functionspaces)
        self.MG = Function(self.acousticPDE.Vl)
        self.srchdir = Function(self.acousticPDE.Vl)
        LinearOperator.__init__(self, self.acousticPDE.MG.vector(), \
        self.acousticPDE.MG.vector())
        self.ObsOp = None


    def copy(self):
        """(hard) copy constructor"""
        V = self.acousticPDE.V
        Vl = self.acousticPDE.Vl
        Vr = self.acousticPDE.Vr
        newobj = self.__class__({'V':V, 'Vl':Vl, 'Vr':Vr})
        newobj.acousticPDE.bc = self.acousticPDE.bc
        if self.acousticPDE.abc == True:
            newobj.acousticPDE.set_abc(V.mesh(), self.acousticPDE.class_bc_abc)
        newobj.acousticPDE.update({'lambda':self.acousticPDE.lam, \
        'rho':self.acousticPDE.rho, 't0':self.acousticPDE.t0, \
        'tf':self.acousticPDE.tf, 'Dt':self.acousticPDE.Dt, \
        'u0init':self.acousticPDE.u0init, 'utinit':self.acousticPDE.utinit, \
        'u1init':self.acousticPDE.u1init})
        return newobj


    # FORWARD PROBLEM + COST:
    def solvefwd(self, cost=False):
        self.acousticPDE.set_fwd()
        self.solution, tmp = self.acousticPDE.solve()
        if cost == True:    #TODO:

    def solvefwd_cost(self):    self.solvefwd(True)


    # ADJOINT PROBLEM + GRAD:
    def solveadj(self, grad=False):
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
    def update_PDE(self, parameters_m): self.acousticPDE.update(parameters_m)

    def set_abc(self, mesh, class_bc_abc):  
        self.acousticPDE.set_abc(mesh, class_bc_abc)

    def update_m(self, lam):    self.acousticPDE.update({'lam':lam})

    def backup_m(self): self.lam-bkup = self.getmarray()

    def setsrcterm(self, ftime):    self.acousticPDE.ftime = ftime


    # GETTERS:
    def getmcopyarray(self):    return self.lam-bkup
    def getmarray(self):    return self.acousticPDE.lam.vector().array()
    def getMGarray(self):   return self.MG.vector().array()

