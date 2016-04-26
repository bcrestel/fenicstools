import numpy as np

from dolfin import LinearOperator, Function, TestFunction, TrialFunction, \
assemble, inner, nabla_grad, dx, sqrt, LUSolver
from miscfenics import setfct, isequal, ZeroRegularization

class ObjectiveAcoustic(LinearOperator):
    """
    Computes data misfit, gradient and Hessian evaluation for the seismic
    inverse problem using acoustic wave data
    """
    #TODO: add support for multiple sources

    # CONSTRUCTORS:
    def __init__(self, acousticwavePDE, regularization=None):
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
        if regularization == None:  self.regularization = ZeroRegularization()
        else:   self.regularization = regularization
        self.alpha_reg = 1.0
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
        self.invDt = 1./self.PDE.Dt
        # Absorbing BCs
        if self.PDE.abc:
            #TODO: should probably be tested in other situations
            if self.PDE.lumpD:
                print '*** Warning: Damping matrix D is lumped. ',\
                'Make sure gradient is consistent.'
            self.vD, self.pD, self.p1D, self.p2D = Function(self.PDE.V), \
            Function(self.PDE.V), Function(self.PDE.V), Function(self.PDE.V)
            self.wkformgradD = inner(0.5*sqrt(self.PDE.rho/self.PDE.lam)\
            *self.pD, self.vD*self.lamtest)*self.PDE.ds(1)
            self.wkformDprime = inner(0.5*sqrt(self.PDE.rho/self.PDE.lam)\
            *self.lamhat*self.ptrial, self.ptest)*self.PDE.ds(1)
            self.dp, self.dph, self.vhatD = Function(self.PDE.V), \
            Function(self.PDE.V), Function(self.PDE.V)
            self.p1hatD, self.p2hatD = Function(self.PDE.V), Function(self.PDE.V)
            self.wkformhessD = inner(-0.25*sqrt(self.PDE.rho)/(self.PDE.lam*sqrt(self.PDE.lam))\
            *self.lamhat*self.dp, self.vD*self.lamtest)*self.PDE.ds(1) \
            + inner(0.5*sqrt(self.PDE.rho/self.PDE.lam)\
            *self.dph, self.vD*self.lamtest)*self.PDE.ds(1)\
            + inner(0.5*sqrt(self.PDE.rho/self.PDE.lam)\
            *self.dp, self.vhatD*self.lamtest)*self.PDE.ds(1)


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
            self.cost_reg = self.regularization.cost(self.PDE.lam)
            self.cost = self.misfit + self.alpha_reg*self.cost_reg

    def solvefwd_cost(self):    self.solvefwd(True)


    # ADJOINT PROBLEM + GRAD:
    #@profile
    def solveadj(self, grad=False):
        self.PDE.set_adj()
        self.obsop.assemble_rhsadj(self.Bp, self.dd, self.PDE.times, self.PDE.bc)
        self.PDE.ftime = self.obsop.ftimeadj
        self.soladj,_ = self.PDE.solve()
        if grad:
            self.MGv.zero()
            if self.PDE.abc:
                self.vD.vector().zero(); self.pD.vector().zero();
                self.p1D.vector().zero(); self.p2D.vector().zero();
            index = 0
            for fwd, adj, fact in \
            zip(self.solfwd, reversed(self.soladj), self.factors):
                ttf, tta = fwd[1], adj[1]
                assert isequal(ttf, tta, 1e-16), \
                'tfwd={}, tadj={}, reldiff={}'.format(ttf, tta, abs(ttf-tta)/ttf)
                setfct(self.p, fwd[0])
                setfct(self.v, adj[0])
                self.MGv.axpy(fact, assemble(self.wkformgrad))
#                self.MGv.axpy(fact, assemble(self.wkformgrad, \
#                form_compiler_parameters={'optimize':True,\
#                'representation':'quadrature'}))
                if self.PDE.abc:
                    if index%2 == 0:
                        self.p2D.vector().axpy(1.0, self.p.vector())
                        setfct(self.pD, self.p2D)
                        self.MGv.axpy(1.0*0.5*self.invDt, assemble(self.wkformgradD))
                        #self.MGv.axpy(fact*0.5*self.invDt, assemble(self.wkformgradD))
                        setfct(self.p2D, -1.0*self.p.vector())
                        setfct(self.vD, self.v)
                    else:
                        self.p1D.vector().axpy(1.0, self.p.vector())
                        setfct(self.pD, self.p1D)
                        self.MGv.axpy(1.0*0.5*self.invDt, assemble(self.wkformgradD))
                        #self.MGv.axpy(fact*0.5*self.invDt, assemble(self.wkformgradD))
                        setfct(self.p1D, -1.0*self.p.vector())
                        setfct(self.vD, self.v)
                index += 1
            self.MGv.axpy(self.alpha_reg, self.regularization.grad(self.PDE.lam))
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
        # lamhat * grad(p).grad(vtilde)
        assert isequal(tt, self.solfwd[index][1], 1e-16)
        setfct(self.p, self.solfwd[index][0])
        setfct(self.v, self.C*self.p.vector())
        # D'.dot(p)
        if self.PDE.abc and index > 0:
                setfct(self.p, \
                self.solfwd[index+1][0] - self.solfwd[index-1][0])
                self.v.vector().axpy(.5*self.invDt, self.Dp*self.p.vector())
        return -1.0*self.v.vector().array()

    def ftimeincradj(self, tt):
        """ Compute rhs for incremental adjoint at time tt """
        try:
            indexf = int(np.where(isequal(self.PDE.times, tt, 1e-14))[0])
            indexa = int(np.where(isequal(self.PDE.times[::-1], tt, 1e-14))[0])
        except:
            print 'Error in ftimeincradj at time {}'.format(tt)
            print np.min(np.abs(self.PDE.times-tt))
            sys.exit(0)
        # lamhat * grad(ptilde).grad(v)
        assert isequal(tt, self.soladj[indexa][1], 1e-16)
        setfct(self.v, self.soladj[indexa][0])
        setfct(self.vhat, self.C*self.v.vector())
        # B* B phat
        assert isequal(tt, self.solincrfwd[indexf][1], 1e-16)
        setfct(self.phat, self.solincrfwd[indexf][0])
        self.vhat.vector().axpy(1.0, self.obsop.incradj(self.phat, tt))
        # D'.dot(v)
        if self.PDE.abc and indexa > 0:
                setfct(self.v, \
                self.soladj[indexa-1][0] - self.soladj[indexa+1][0])
                self.vhat.vector().axpy(-.5*self.invDt, self.Dp*self.v.vector())
        return -1.0*self.vhat.vector().array()
        
    def mult(self, lamhat, y):
        """
        mult(self, lamhat, y): return y = Hessian * lamhat
        inputs:
            y, lamhat = Function(V).vector()
        """
        self.regularization.assemble_hessian(lamhat)
        setfct(self.lamhat, lamhat)
        self.C = assemble(self.wkformrhsincr)
        if self.PDE.abc:    self.Dp = assemble(self.wkformDprime)
        # solve for phat
        self.PDE.set_fwd()
        self.PDE.ftime = self.ftimeincrfwd
        self.solincrfwd,_ = self.PDE.solve()
        # solve for vhat
        self.PDE.set_adj()
        self.PDE.ftime = self.ftimeincradj
        self.solincradj,_ = self.PDE.solve()
        # Compute Hessian*lamhat
        y.zero()
        index = 0
        if self.PDE.abc:
            self.vD.vector().zero(); self.vhatD.vector().zero(); 
            self.p1D.vector().zero(); self.p2D.vector().zero();
            self.p1hatD.vector().zero(); self.p2hatD.vector().zero();
        for fwd, adj, incrfwd, incradj, fact in \
        zip(self.solfwd, reversed(self.soladj), \
        self.solincrfwd, reversed(self.solincradj), self.factors):
            ttf, tta, ttf2 = incrfwd[1], incradj[1], fwd[1]
            assert isequal(ttf, tta, 1e-16), 'tfwd={}, tadj={}, reldiff={}'.\
            format(ttf, tta, abs(ttf-tta)/ttf)
            assert isequal(ttf, ttf2, 1e-16), 'tfwd={}, tadj={}, reldiff={}'.\
            format(ttf, ttf2, abs(ttf-ttf2)/ttf)
            setfct(self.p, fwd[0])
            setfct(self.v, adj[0])
            setfct(self.phat, incrfwd[0])
            setfct(self.vhat, incradj[0])
            y.axpy(fact, assemble(self.wkformhess))
            if self.PDE.abc:
                if index%2 == 0:
                    self.p2D.vector().axpy(1.0, self.p.vector())
                    self.p2hatD.vector().axpy(1.0, self.phat.vector())
                    setfct(self.dp, self.p2D)
                    setfct(self.dph, self.p2hatD)
                    y.axpy(1.0*0.5*self.invDt, assemble(self.wkformhessD))
                    setfct(self.p2D, -1.0*self.p.vector())
                    setfct(self.p2hatD, -1.0*self.phat.vector())
                else:
                    self.p1D.vector().axpy(1.0, self.p.vector())
                    self.p1hatD.vector().axpy(1.0, self.phat.vector())
                    setfct(self.dp, self.p1D)
                    setfct(self.dph, self.p1hatD)
                    y.axpy(1.0*0.5*self.invDt, assemble(self.wkformhessD))
                    setfct(self.p1D, -1.0*self.p.vector())
                    setfct(self.p1hatD, -1.0*self.phat.vector())
                setfct(self.vD, self.v)
                setfct(self.vhatD, self.vhat)
            index += 1
        # add regularization term
        y.axpy(self.alpha_reg, self.regularization.hessian(lamhat))


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

