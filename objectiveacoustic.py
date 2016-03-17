import numpy as np

from dolfin import LinearOperator, Function, TestFunction, TrialFunction, \
assemble, inner, nabla_grad, dx,\
LUSolver
from miscfenics import setfct, isequal

class ObjectiveAcoustic(LinearOperator):
    """
    Computes data misfit, gradient and Hessian evaluation for the seismic
    inverse problem using acoustic wave data
    """
    #TODO: add support for multiple sources

    # CONSTRUCTORS:
    def __init__(self, acousticwavePDE):
        """ 
        Input:
            acousticwavePDE should be an instantiation from class AcousticWave
        """
        self.PDE = acousticwavePDE
        self.PDE.exact = None
        self.fwdsource = self.PDE.ftime
        self.MG = Function(self.PDE.Vl)
        self.MGv = self.MG.vector()
        self.Grad = Function(self.PDE.Vl)
        self.Gradv = self.Grad.vector()
        self.srchdir = Function(self.PDE.Vl)
        self.delta_m = Function(self.PDE.Vl)
        LinearOperator.__init__(self, self.MG.vector(), self.MG.vector())
        self.obsop = None   # Observation operator
        self.dd = None  # observations
        # gradient
        self.lamtest, self.lamtrial = TestFunction(self.PDE.Vl), TrialFunction(self.PDE.Vl)
        self.p, self.v = Function(self.PDE.V), Function(self.PDE.V)
        self.wkformgrad = inner(self.lamtest*nabla_grad(self.p), nabla_grad(self.v))*dx
        # incremental rhs
        self.lamhat = Function(self.PDE.Vl)
        self.ptrial, self.ptest = TrialFunction(self.PDE.V), TestFunction(self.PDE.V)
        self.wkformrhsincr = inner(self.lamhat*nabla_grad(self.ptrial), nabla_grad(self.ptest))*dx
        # Hessian
        self.phat, self.vhat = Function(self.PDE.V), Function(self.PDE.V)
        self.wkformhess = inner(self.lamtest*nabla_grad(self.phat), nabla_grad(self.v))*dx \
        + inner(self.lamtest*nabla_grad(self.p), nabla_grad(self.vhat))*dx
        # Mass matrix:
        weak_m =  inner(self.lamtrial,self.lamtest)*dx
        Mass = assemble(weak_m)
        self.solverM = LUSolver()
        self.solverM.parameters['reuse_factorization'] = True
        self.solverM.parameters['symmetric'] = True
        self.solverM.set_operator(Mass)
        # Time-integration factors
        self.factors = np.ones(self.PDE.times.size)
        self.factors[0], self.factors[-1] = 0.5, 0.5
        self.factors *= self.PDE.Dt


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
        for index, sol in enumerate(self.solfwd):
            setfct(self.p, sol[0])
            self.Bp[:,index] = self.obsop.obs(self.p)
        if cost:
            assert not self.dd == None, "Provide observations"
            self.misfit = self.obsop.costfct(self.Bp, self.dd, self.PDE.times)
            self.cost = self.misfit #TODO: add regularization

    def solvefwd_cost(self):    self.solvefwd(True)


    # ADJOINT PROBLEM + GRAD:
    #TODO: profile; gradient computation step slow
    def solveadj(self, grad=False):
        self.PDE.set_adj()
        self.obsop.assemble_rhsadj(self.Bp, self.dd, self.PDE.times, self.PDE.bc)
        self.PDE.ftime = self.obsop.ftimeadj
        self.soladj,_ = self.PDE.solve()
        if grad:
            self.MGv.zero()
            #TODO: add boundary term for abs bc
            for fwd, adj, fact in zip(self.solfwd, reversed(self.soladj), self.factors):
                ttf, tta = fwd[1], adj[1]
                assert isequal(ttf, tta, 1e-16), \
                'tfwd={}, tadj={}, reldiff={}'.format(ttf, tta, abs(ttf-tta)/ttf)
                setfct(self.p, fwd[0])
                setfct(self.v, adj[0])
                self.MGv.axpy(fact, assemble(self.wkformgrad))
            self.solverM.solve(self.Gradv, self.MGv)

    def solveadj_constructgrad(self):   self.solveadj(True)


    # HESSIAN:
    def ftimeincrfwd(self, tt):
        """ Compute rhs for incremental forward at time tt """
        try:
            index = int(np.where(isequal(self.PDE.times, tt, 1e-14))[0])
        except:
            print 'Error in ftimeincrfwd at time {}'.format(tt)
            print np.min(np.abs(self.PDE.times-tt))
            sys.exit(0)
        setfct(self.p, self.solfwd[index][0])
        return -1.0*(self.C*self.p.vector()).array()

    def ftimeincradj(self, tt):
        """ Compute rhs for incremental adjoint at time tt """
        try:
            index = int(np.where(isequal(self.PDE.times, tt, 1e-14))[0])
        except:
            print 'Error in ftimeincradj at time {}'.format(tt)
            print np.min(np.abs(self.PDE.times-tt))
            sys.exit(0)
        # lamhat * grad(ptilde).grad(v)
        setfct(self.v, self.soladj[index][0])
        setfct(self.vhat, self.C * self.v.vector())
        # B* B phat
        setfct(self.phat, self.solincrfwd[index][0])
        self.vhat.vector().axpy(1.0, self.obsop.incradj(self.phat, tt))
        return -1.0*self.vhat.vector().array()
        
    def mult(self, lamhat, y):
        """
        mult(self, lamhat, y): return y = Hessian * lamhat
        inputs:
            y, lamhat = Function(V).vector()
        """
        #TODO: testing + add boundary terms for abs abc
        setfct(self.lamhat, lamhat)
        self.C = assemble(self.wkformrhsincr)
        # solve for phat
        self.PDE.set_fwd()
        self.PDE.ftime = self.ftimeincrfwd
        self.solincrfwd,_ = self.PDE.solve()
        # solve for vhat
        self.PDE.set_adj()
        self.PDE.ftime = self.ftimeincradj
        self.solincradj,_ = self.PDE.solve()
        # Compute Hessian*x
        y.zero()
        for fwd, adj, incrfwd, incradj, fact in \
        zip(self.solfwd, reversed(self.soladj), \
        self.solincrfwd, reversed(self.solincradj), self.factors):
            ttf, tta = incrfwd[1], incradj[1]
            assert isequal(ttf, tta, 1e-16), \
            'tfwd={}, tadj={}, reldiff={}'.format(ttf, tta, abs(ttf-tta)/ttf)
            setfct(self.p, fwd[0])
            setfct(self.v, adj[0])
            setfct(self.phat, incrfwd[0])
            setfct(self.vhat, incradj[0])
            y.axpy(fact, assemble(self.wkformhess))


    # SETTERS + UPDATE:
    def update_PDE(self, parameters): self.PDE.update(parameters)
    def update_m(self, lam):    self.update_PDE({'lambda':lam})
    def set_abc(self, mesh, class_bc_abc, lumpD):  
        self.PDE.set_abc(mesh, class_bc_abc, lumpD)
    def backup_m(self): self.lam_bkup = self.getmarray()
    def restore_m(self):    self.update_m(self.lam_bkup)
    def setsrcterm(self, ftime):    self.PDE.ftime = ftime


    # GETTERS:
    def getmcopyarray(self):    return self.lam_bkup
    def getmarray(self):    return self.PDE.lam.vector().array()
    def getMGarray(self):   return self.MGv.array()

