import numpy as np

from dolfin import LinearOperator, Function, TestFunction, TrialFunction, \
assemble, inner, nabla_grad, dx, sqrt, LUSolver, assign
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
        # functions for gradient
        self.MG = Function(self.PDE.Vm*self.PDE.Vm)
        self.MGv = self.MG.vector()
        self.Grad = Function(self.PDE.Vm*self.PDE.Vm)
        self.srchdir = Function(self.PDE.Vm*self.PDE.Vm)
        self.delta_m = Function(self.PDE.Vm*self.PDE.Vm)
        self.ab = Function(self.PDE.Vm*self.PDE.Vm)
        self.ab_bkup = Function(self.PDE.Vm*self.PDE.Vm)
        #LinearOperator.__init__(self, self.MG.vector(), self.MG.vector())
        self.obsop = None   # Observation operator
        self.dd = None  # observations
        # regularization
        if regularization == None:  
            print '*** Warning: Using zero regularization'
            self.regularization = ZeroRegularization()
        else:   self.regularization = regularization
        self.alpha_reg = 1.0
        # gradient a
        self.mtest, self.mtrial = TestFunction(self.PDE.Vm), TrialFunction(self.PDE.Vm)
        self.p, self.q = Function(self.PDE.V), Function(self.PDE.V)
        self.ppprhs = Function(self.PDE.V)
        self.ptmp = Function(self.PDE.V)
        #TODO: not consistent with lumped mass matrix -- needs to think about that
        self.wkformgrada = inner(self.mtest*self.p, self.q)*dx
        # gradient b
        self.wkformgradb = inner(self.mtest*nabla_grad(self.p), nabla_grad(self.q))*dx
        # incremental rhs a
        self.ahat, self.bhat = Function(self.PDE.Vm), Function(self.PDE.Vm)
        self.ptrial, self.ptest = TrialFunction(self.PDE.V), TestFunction(self.PDE.V)
        self.wkformrhsincra = inner(self.ahat*self.ptrial, self.ptest)*dx
        # incremental rhs b
        self.wkformrhsincrb = inner(self.bhat*nabla_grad(self.ptrial), nabla_grad(self.ptest))*dx
        # Hessian a
        self.phat, self.qhat = Function(self.PDE.V), Function(self.PDE.V)
        self.wkformhessa = inner(self.mtest*self.phat, self.q)*dx \
        + inner(self.mtest*self.p, self.qhat)*dx
        # Hessian b
        self.wkformhessb = inner(self.mtest*nabla_grad(self.phat), nabla_grad(self.q))*dx \
        + inner(self.mtest*nabla_grad(self.p), nabla_grad(self.qhat))*dx
        # Mass matrix:
        weak_m =  inner(self.mtrial,self.mtest)*dx
        Mass = assemble(weak_m)
        self.solverM = LUSolver("petsc")
        self.solverM.parameters['reuse_factorization'] = True
        self.solverM.parameters['symmetric'] = True
        self.solverM.set_operator(Mass)
        # Time-integration factors
        self.factors = np.ones(self.PDE.times.size)
        self.factors[0], self.factors[-1] = 0.5, 0.5
        self.factors *= self.PDE.Dt
        self.invDt = 1./self.PDE.Dt
        # Absorbing BCs
        #TODO: not implemented yet
#        if self.PDE.abc:
#            if self.PDE.lumpD:
#                print '*** Warning: Damping matrix D is lumped. ',\
#                'Make sure gradient is consistent.'
#            self.vD, self.pD, self.p1D, self.p2D = Function(self.PDE.V), \
#            Function(self.PDE.V), Function(self.PDE.V), Function(self.PDE.V)
#            self.wkformgradD = inner(0.5*sqrt(self.PDE.rho/self.PDE.lam)\
#            *self.pD, self.vD*self.lamtest)*self.PDE.ds(1)
#            self.wkformDprime = inner(0.5*sqrt(self.PDE.rho/self.PDE.lam)\
#            *self.bhat*self.ptrial, self.ptest)*self.PDE.ds(1)
#            self.dp, self.dph, self.vhatD = Function(self.PDE.V), \
#            Function(self.PDE.V), Function(self.PDE.V)
#            self.p1hatD, self.p2hatD = Function(self.PDE.V), Function(self.PDE.V)
#            self.wkformhessD = inner(-0.25*sqrt(self.PDE.rho)/(self.PDE.lam*sqrt(self.PDE.lam))\
#            *self.bhat*self.dp, self.vD*self.lamtest)*self.PDE.ds(1) \
#            + inner(0.5*sqrt(self.PDE.rho/self.PDE.lam)\
#            *self.dph, self.vD*self.lamtest)*self.PDE.ds(1)\
#            + inner(0.5*sqrt(self.PDE.rho/self.PDE.lam)\
#            *self.dp, self.vhatD*self.lamtest)*self.PDE.ds(1)


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
            self.cost_misfit = self.obsop.costfct(self.Bp, self.dd, self.PDE.times)
            self.cost_reg = self.regularization.costa(self.PDE.a) + \
            self.regularization.costb(self.PDE.b)
            self.cost = self.cost_misfit + self.alpha_reg*self.cost_reg

    def solvefwd_cost(self):    self.solvefwd(True)


    # ADJOINT PROBLEM + GRAD:
    def solveadj(self, grad=False):
        self.PDE.set_adj()
        self.obsop.assemble_rhsadj(self.Bp, self.dd, self.PDE.times, self.PDE.bc)
        self.PDE.ftime = self.obsop.ftimeadj
        self.soladj,_ = self.PDE.solve()
        if grad:
            self.MGv.zero()
            MGa, MGb = self.MG.split(deepcopy=True)
            MGav, MGbv = MGa.vector(), MGb.vector()
#            if self.PDE.abc:
#                self.vD.vector().zero(); self.pD.vector().zero();
#                self.p1D.vector().zero(); self.p2D.vector().zero();
            index = 0
            for fwd, adj, fact, fwdm, fwdp in \
            zip(self.solfwd, reversed(self.soladj), self.factors,\
            [[np.zeros(self.PDE.V.dim()), -self.PDE.Dt]]+self.solfwd[:-1], \
            self.solfwd[1:]+[[np.zeros(self.PDE.V.dim()), self.PDE.times[-1]+self.PDE.Dt]]):
                ttf, tta = fwd[1], adj[1]
                assert isequal(ttf, tta, 1e-16), \
                'tfwd={}, tadj={}, reldiff={}'.format(ttf, tta, abs(ttf-tta)/ttf)
                setfct(self.p, fwd[0])
                setfct(self.q, adj[0])
                MGbv.axpy(fact, assemble(self.wkformgradb))
#                # technique 1
#                ttfm, ttfp = fwdm[1], fwdp[1]
#                assert isequal(ttfm+self.PDE.Dt, tta, 1e-15), 'a={}, b={}, err={}'.\
#                format(ttfm+self.PDE.Dt, tta, np.abs(ttfm+self.PDE.Dt-tta)/tta)
#                assert isequal(ttfp-self.PDE.Dt, tta, 1e-15), 'a={}, b={}, err={}'.\
#                format(ttfp-self.PDE.Dt, tta, np.abs(ttfp-self.PDE.Dt-tta)/tta)
#                if index > 0:   # do not include term at t=0
#                    setfct(self.ptmp, fwdm[0])
#                    self.p.vector().axpy(-0.5, self.ptmp.vector())
#                    setfct(self.ptmp, fwdp[0])
#                    self.p.vector().axpy(-0.5, self.ptmp.vector())
#                else:
#                    setfct(self.ptmp, fwdp[0])
#                    self.p.vector().axpy(-1.0, self.ptmp.vector())
#                MGav.axpy(-2.0*self.invDt*self.invDt*fact, assemble(self.wkformgrada))
                # technique 3
                setfct(self.ppprhs, self.fwdsource(ttf))
                setfct(self.ptmp, fwd[0])
                self.ppprhs.vector().axpy(-1.0, self.PDE.K*self.ptmp.vector())
                self.PDE.solverM.solve(self.p.vector(), self.ppprhs.vector())
                MGav.axpy(fact, assemble(self.wkformgrada))

#                if self.PDE.abc:
#                    if index%2 == 0:
#                        self.p2D.vector().axpy(1.0, self.p.vector())
#                        setfct(self.pD, self.p2D)
#                        self.MGv.axpy(fact*0.5*self.invDt, assemble(self.wkformgradD))
#                        setfct(self.p2D, -1.0*self.p.vector())
#                        setfct(self.vD, self.v)
#                    else:
#                        self.p1D.vector().axpy(1.0, self.p.vector())
#                        setfct(self.pD, self.p1D)
#                        self.MGv.axpy(fact*0.5*self.invDt, assemble(self.wkformgradD))
#                        setfct(self.p1D, -1.0*self.p.vector())
#                        setfct(self.vD, self.v)
                index += 1
            # add regularization
            Grada, Gradb = self.Grad.split(deepcopy=True)
            Gradav, Gradbv = Grada.vector(), Gradb.vector()
            MGav.axpy(self.alpha_reg, self.regularization.grada(self.PDE.a))
            self.solverM.solve(Gradav, MGav)
            MGbv.axpy(self.alpha_reg, self.regularization.gradb(self.PDE.b))
            self.solverM.solve(Gradbv, MGbv)
            # assemble
            assign(self.MG.sub(0), MGa)
            assign(self.Grad.sub(0), Grada)
            assign(self.MG.sub(1), MGb)
            assign(self.Grad.sub(1), Gradb)

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
        # bhat * grad(p).grad(vtilde)
        assert isequal(tt, self.solfwd[index][1], 1e-16)
        setfct(self.p, self.solfwd[index][0])
        setfct(self.q, self.C*self.p.vector())
        # D'.dot(p)
#        if self.PDE.abc and index > 0:
#                setfct(self.p, \
#                self.solfwd[index+1][0] - self.solfwd[index-1][0])
#                self.v.vector().axpy(.5*self.invDt, self.Dp*self.p.vector())
        return -1.0*self.q.vector().array()

    def ftimeincradj(self, tt):
        """ Compute rhs for incremental adjoint at time tt """
        try:
            indexf = int(np.where(isequal(self.PDE.times, tt, 1e-14))[0])
            indexa = int(np.where(isequal(self.PDE.times[::-1], tt, 1e-14))[0])
        except:
            print 'Error in ftimeincradj at time {}'.format(tt)
            print np.min(np.abs(self.PDE.times-tt))
            sys.exit(0)
        # bhat * grad(ptilde).grad(v)
        assert isequal(tt, self.soladj[indexa][1], 1e-16)
        setfct(self.q, self.soladj[indexa][0])
        setfct(self.qhat, self.C*self.q.vector())
        # B* B phat
        assert isequal(tt, self.solincrfwd[indexf][1], 1e-16)
        setfct(self.phat, self.solincrfwd[indexf][0])
        self.qhat.vector().axpy(1.0, self.obsop.incradj(self.phat, tt))
        # D'.dot(v)
#        if self.PDE.abc and indexa > 0:
#                setfct(self.v, \
#                self.soladj[indexa-1][0] - self.soladj[indexa+1][0])
#                self.vhat.vector().axpy(-.5*self.invDt, self.Dp*self.v.vector())
        return -1.0*self.qhat.vector().array()
        
    def mult(self, abhat, y):
        """
        mult(self, abhat, y): return y = Hessian * lamhat
        inputs:
            y, abhat = Function(V).vector()
        """
        # get ahat, bhat:
        setfct(self.ab, abhat)
        ahat, bhat = self.ab.split(deepcopy=True)
        setfct(self.ahat, ahat)
        setfct(self.bhat, bhat)
        self.regularization.assemble_hessian(abhat)
        self.C = assemble(self.wkformrhsincrb)
        #if self.PDE.abc:    self.Dp = assemble(self.wkformDprime)
        # solve for phat
        self.PDE.set_fwd()
        self.PDE.ftime = self.ftimeincrfwd
        self.solincrfwd,_ = self.PDE.solve()
        # solve for vhat
        self.PDE.set_adj()
        self.PDE.ftime = self.ftimeincradj
        self.solincradj,_ = self.PDE.solve()
        # Compute Hessian*abhat
        self.ab.vector().zero()
        yaF, ybF = self.ab.split(deepcopy=True)
        ya, yb = yaF.vector(), ybF.vector()
        index = 0
#        if self.PDE.abc:
#            self.vD.vector().zero(); self.vhatD.vector().zero(); 
#            self.p1D.vector().zero(); self.p2D.vector().zero();
#            self.p1hatD.vector().zero(); self.p2hatD.vector().zero();
        for fwd, adj, incrfwd, incradj, fact in \
        zip(self.solfwd, reversed(self.soladj), \
        self.solincrfwd, reversed(self.solincradj), self.factors):
            ttf, tta, ttf2 = incrfwd[1], incradj[1], fwd[1]
            assert isequal(ttf, tta, 1e-16), 'tfwd={}, tadj={}, reldiff={}'.\
            format(ttf, tta, abs(ttf-tta)/ttf)
            assert isequal(ttf, ttf2, 1e-16), 'tfwd={}, tadj={}, reldiff={}'.\
            format(ttf, ttf2, abs(ttf-ttf2)/ttf)
            setfct(self.p, fwd[0])
            setfct(self.q, adj[0])
            setfct(self.phat, incrfwd[0])
            setfct(self.qhat, incradj[0])
            yb.axpy(fact, assemble(self.wkformhessb))
#            if self.PDE.abc:
#                if index%2 == 0:
#                    self.p2D.vector().axpy(1.0, self.p.vector())
#                    self.p2hatD.vector().axpy(1.0, self.phat.vector())
#                    setfct(self.dp, self.p2D)
#                    setfct(self.dph, self.p2hatD)
#                    y.axpy(1.0*0.5*self.invDt, assemble(self.wkformhessD))
#                    setfct(self.p2D, -1.0*self.p.vector())
#                    setfct(self.p2hatD, -1.0*self.phat.vector())
#                else:
#                    self.p1D.vector().axpy(1.0, self.p.vector())
#                    self.p1hatD.vector().axpy(1.0, self.phat.vector())
#                    setfct(self.dp, self.p1D)
#                    setfct(self.dph, self.p1hatD)
#                    y.axpy(1.0*0.5*self.invDt, assemble(self.wkformhessD))
#                    setfct(self.p1D, -1.0*self.p.vector())
#                    setfct(self.p1hatD, -1.0*self.phat.vector())
#                setfct(self.vD, self.v)
#                setfct(self.vhatD, self.vhat)
            index += 1
        assign(self.ab.sub(0), yaF)
        assign(self.ab.sub(1), ybF)
        y.zero()
        y.axpy(1.0, self.ab.vector())
        # add regularization term
        y.axpy(self.alpha_reg, self.regularization.hessian(abhat))


    # SETTERS + UPDATE:
    def update_PDE(self, parameters): self.PDE.update(parameters)
    def update_m(self, medparam):
        """ medparam is a np.array containing both med parameters """
        setfct(self.ab, medparam)
        a, b = self.ab.split(deepcopy=True)
        self.update_PDE({'a':a, 'b':b})
    def set_abc(self, mesh, class_bc_abc, lumpD):  
        self.PDE.set_abc(mesh, class_bc_abc, lumpD)
    def backup_m(self): 
        """ back-up current value of med param a and b """
        assign(self.ab_bkup.sub(0), self.PDE.a)
        assign(self.ab_bkup.sub(1), self.PDE.b)
    def restore_m(self):    
        """ restore backed-up values of a and b """
        a, b = self.ab_bkup.split(deepcopy=True)
        self.update_m({'a':a, 'b':b})
    def setsrcterm(self, ftime):    self.PDE.ftime = ftime


    # GETTERS:
    def getmcopyarray(self):    return self.ab_bkup.vector().array()
    def getMGarray(self):   return self.MGv.array()

