import numpy as np

from dolfin import LinearOperator, Function, TestFunction, TrialFunction, \
assemble, inner, nabla_grad, dx, sqrt, LUSolver, assign, Constant, \
PETScKrylovSolver
from miscfenics import setfct, isequal, ZeroRegularization
from linalg.lumpedmatrixsolver import LumpedMassMatrixPrime

class ObjectiveAcoustic(LinearOperator):
    """
    Computes data misfit, gradient and Hessian evaluation for the seismic
    inverse problem using acoustic wave data
    """
    #TODO: add support for multiple sources

    # CONSTRUCTORS:
    def __init__(self, acousticwavePDE, invparam='ab', regularization=None):
        """ 
        Input:
            acousticwavePDE should be an instantiation from class AcousticWave
        """
        self.PDE = acousticwavePDE
        self.PDE.exact = None
        self.fwdsource = self.PDE.ftime
        # functions for gradient
        self.ab = Function(self.PDE.Vm*self.PDE.Vm)
        self.obsop = None   # Observation operator
        self.dd = None  # observations
        # decide whether ahat and bhat are used
        self.invparam = invparam
        if self.invparam == 'ab':
            self.MG = Function(self.PDE.Vm*self.PDE.Vm)
            self.MGv = self.MG.vector()
            self.Grad = Function(self.PDE.Vm*self.PDE.Vm)
            self.srchdir = Function(self.PDE.Vm*self.PDE.Vm)
            self.delta_m = Function(self.PDE.Vm*self.PDE.Vm)
            self.get_costreg = self.get_costreg_joint
            self.update_m = self.update_ab
            self.m_bkup = Function(self.PDE.Vm*self.PDE.Vm)
            self.backup_m = self.backup_ab
            self.restore_m = self.restore_ab
            self.ftimeincrfwda = self.ftimeincrfwd_componenta
            self.ftimeincrfwdb = self.ftimeincrfwd_componentb
            self.ftimeincradja = self.ftimeincradj_componenta
            self.ftimeincradjb = self.ftimeincradj_componentb
            self.hessiana = self.hessian_componenta
            self.hessianb = self.hessian_componentb
        else:
            self.MG = Function(self.PDE.Vm)
            self.MGv = self.MG.vector()
            self.Grad = Function(self.PDE.Vm)
            self.srchdir = Function(self.PDE.Vm)
            self.delta_m = Function(self.PDE.Vm)
            self.m_bkup = Function(self.PDE.Vm)
            if self.invparam == 'a':
                self.get_costreg = self.get_costreg_a
                self.update_m = self.update_a
                self.backup_m = self.backup_a
                self.restore_m = self.restore_a
                self.ftimeincrfwda = self.ftimeincrfwd_componenta
                self.ftimeincrfwdb = self.ftimepass
                self.ftimeincradja = self.ftimeincradj_componenta
                self.ftimeincradjb = self.ftimepass
                self.hessiana = self.hessian_componenta
                self.hessianb = self.hessian_componentbpass
            elif self.invparam == 'b':
                self.get_costreg = self.get_costreg_b
                self.update_m = self.update_b
                self.backup_m = self.backup_b
                self.restore_m = self.restore_b
                self.ftimeincrfwda = self.ftimeincrfwd_componentapass
                self.ftimeincrfwdb = self.ftimeincrfwd_componentb
                self.ftimeincradja = self.ftimeincradj_componentapass
                self.ftimeincradjb = self.ftimeincradj_componentb
                self.hessiana = self.hessian_componentapass
                self.hessianb = self.hessian_componentb
        LinearOperator.__init__(self, self.MGv, self.MGv)
        # regularization
        if regularization == None:  
            print '*** Warning: Using zero regularization'
            self.regularization = ZeroRegularization()
        else:   self.regularization = regularization
        self.alpha_reg = 1.0
        # gradient and Hessian
        self.p, self.q = Function(self.PDE.V), Function(self.PDE.V)
        self.phat, self.qhat = Function(self.PDE.V), Function(self.PDE.V)
        self.ahat, self.bhat = Function(self.PDE.Vm), Function(self.PDE.Vm)
        self.ptrial, self.ptest = TrialFunction(self.PDE.V), TestFunction(self.PDE.V)
        self.mtest, self.mtrial = TestFunction(self.PDE.Vm), TrialFunction(self.PDE.Vm)
        self.ppprhs = Function(self.PDE.V)
        self.ptmp = Function(self.PDE.V)
        if self.PDE.lump:
            self.Mprime = LumpedMassMatrixPrime(self.PDE.Vm, self.PDE.V, self.PDE.M.ratio)
            self.get_gradienta = self.get_gradienta_lumped
            self.get_hessiana = self.get_hessiana_lumped
            self.get_incra = self.get_incra_lumped
        else:
            self.wkformgrada = inner(self.mtest*self.p, self.q)*dx
            self.get_gradienta = self.get_gradienta_full
            self.wkformhessa = inner(self.phat*self.mtest, self.q)*dx \
            + inner(self.p*self.mtest, self.qhat)*dx
            self.get_hessiana = self.get_hessiana_full
            self.wkformrhsincra = inner(self.ahat*self.ptrial, self.ptest)*dx
            self.get_incra = self.get_incra_full
        self.wkformgradb = inner(self.mtest*nabla_grad(self.p), nabla_grad(self.q))*dx
        self.wkformrhsincrb = inner(self.bhat*nabla_grad(self.ptrial), nabla_grad(self.ptest))*dx
        self.wkformhessb = inner(nabla_grad(self.phat)*self.mtest, nabla_grad(self.q))*dx \
        + inner(nabla_grad(self.p)*self.mtest, nabla_grad(self.qhat))*dx
        # Mass matrix:
        if self.invparam == 'ab':
            # mass matrix will be block-diagonal (although rows and columns are mixed)
            self.mmtest, self.mmtrial = TestFunction(self.PDE.Vm*self.PDE.Vm), \
            TrialFunction(self.PDE.Vm*self.PDE.Vm)
        else:
            self.mmtest, self.mmtrial = self.mtest, self.mtrial
        weak_m =  inner(self.mmtrial, self.mmtest)*dx
        self.Mass = assemble(weak_m)
        self.solverM = LUSolver("petsc")
        self.solverM.parameters['reuse_factorization'] = True
        self.solverM.parameters['symmetric'] = True
        self.solverM.set_operator(self.Mass)
        # Time-integration factors
        self.factors = np.ones(self.PDE.times.size)
        self.factors[0], self.factors[-1] = 0.5, 0.5
        self.factors *= self.PDE.Dt
        self.invDt = 1./self.PDE.Dt
        # Absorbing BCs
        #TODO: ABCs not implemented yet for joint inversion
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
        setfct(newobj.Grad, self.Grad)
        setfct(newobj.srchdir, self.srchdir)
        newobj.obsop = self.obsop
        newobj.dd = self.dd
        return newobj


    # FORWARD PROBLEM + COST:
    #@profile
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
            assert not self.dd == None, "Provide data observations to compute cost"
            self.cost_misfit = self.obsop.costfct(self.Bp, self.dd, self.PDE.times)
            self.cost_reg = self.get_costreg()
            self.cost = self.cost_misfit + self.alpha_reg*self.cost_reg

    def solvefwd_cost(self):    self.solvefwd(True)

    def get_costreg_joint(self):    return self.regularization.costab(self.PDE.a, self.PDE.b)
    def get_costreg_a(self):    return self.regularization.cost(self.PDE.a)
    def get_costreg_b(self):    return self.regularization.cost(self.PDE.b)
        

    # ADJOINT PROBLEM + GRAD:
    #@profile
    def solveadj(self, grad=False):
        self.PDE.set_adj()
        self.obsop.assemble_rhsadj(self.Bp, self.dd, self.PDE.times, self.PDE.bc)
        self.PDE.ftime = self.obsop.ftimeadj
        self.soladj,_ = self.PDE.solve()
        if grad:
            if self.invparam == 'ab':
                # split gradient parts a and b
                self.MGv.zero()
                MGa, MGb = self.MG.split(deepcopy=True)
                MGav, MGbv = MGa.vector(), MGb.vector()
                # loop over time
                for fwd, adj, fact, fwdm, fwdp in \
                zip(self.solfwd, reversed(self.soladj), self.factors,\
                [[np.zeros(self.PDE.V.dim()), -self.PDE.Dt]]+self.solfwd[:-1], \
                self.solfwd[1:]+[[np.zeros(self.PDE.V.dim()), self.PDE.times[-1]+self.PDE.Dt]]):
                    self.gradient_componentb(fact, fwd, adj, MGbv)
                    self.gradient_componenta(fact, fwdm, fwdp, adj, MGav)
                # add regularization
                assign(self.MG.sub(0), MGa)
                assign(self.MG.sub(1), MGb)
                self.MGv.axpy(self.alpha_reg, \
                self.regularization.gradab(self.PDE.a, self.PDE.b))
            else:
                if self.invparam == 'a':
                    self.MGv.zero()
                    MGav = self.MGv
                    # loop over time
                    for fwd, adj, fact, fwdm, fwdp in \
                    zip(self.solfwd, reversed(self.soladj), self.factors,\
                    [[np.zeros(self.PDE.V.dim()), -self.PDE.Dt]]+self.solfwd[:-1], \
                    self.solfwd[1:]+[[np.zeros(self.PDE.V.dim()), self.PDE.times[-1]+self.PDE.Dt]]):
                        setfct(self.p, fwd[0])
                        setfct(self.q, adj[0])
                        self.gradient_componenta(fact, fwdm, fwdp, adj, MGav)
                    # add regularization
                    MGav.axpy(self.alpha_reg, self.regularization.grad(self.PDE.a))
                elif self.invparam == 'b':
                    self.MGv.zero()
                    MGbv = self.MGv
                    # loop over time
                    for fwd, adj, fact in \
                    zip(self.solfwd, reversed(self.soladj), self.factors):
                        self.gradient_componentb(fact, fwd, adj, MGbv)
                    # add regularization
                    MGbv.axpy(self.alpha_reg, self.regularization.grad(self.PDE.b))
            # compute Grad
            self.solverM.solve(self.Grad.vector(), self.MGv)
#            if self.PDE.abc:
#                self.vD.vector().zero(); self.pD.vector().zero();
#                self.p1D.vector().zero(); self.p2D.vector().zero();
#            index = 0
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
#                index += 1

    def solveadj_constructgrad(self):   self.solveadj(True)

    def gradient_componentb(self, fact, fwd, adj, MGbv):
        ttf, tta = fwd[1], adj[1]
        assert isequal(ttf, tta, 1e-16), \
        'tfwd={}, tadj={}, reldiff={}'.format(ttf, tta, abs(ttf-tta)/ttf)
        setfct(self.p, fwd[0])
        setfct(self.q, adj[0])
        MGbv.axpy(fact, assemble(self.wkformgradb))
    def gradient_componenta(self, fact, fwdm, fwdp, adj, MGav):
        tta = adj[1]
        ttfm, ttfp = fwdm[1], fwdp[1]
        assert isequal(ttfm+self.PDE.Dt, tta, 1e-15), 'a={}, b={}, err={}'.\
        format(ttfm+self.PDE.Dt, tta, np.abs(ttfm+self.PDE.Dt-tta)/tta)
        assert isequal(ttfp-self.PDE.Dt, tta, 1e-15), 'a={}, b={}, err={}'.\
        format(ttfp-self.PDE.Dt, tta, np.abs(ttfp-self.PDE.Dt-tta)/tta)
        setfct(self.ptmp, fwdm[0])
        self.p.vector().axpy(-0.5, self.ptmp.vector())
        setfct(self.ptmp, fwdp[0])
        self.p.vector().axpy(-0.5, self.ptmp.vector())
        MGav.axpy(-2.0*self.invDt*self.invDt*fact, self.get_gradienta()) 

    def get_gradienta_lumped(self):
        return self.Mprime.get_gradient(self.p.vector(), self.q.vector())
    def get_gradienta_full(self):
        return assemble(self.wkformgrada)

    # HESSIAN:
    def ftimepass(self):
        pass

    #@profile
    def ftimeincrfwd(self, tt):
        """ Compute rhs for incremental forward at time tt """
        try:
            index = int(np.where(isequal(self.PDE.times, tt, 1e-14))[0])
        except:
            print 'Error in ftimeincrfwd at time {}'.format(tt)
            print np.min(np.abs(self.PDE.times-tt))
            sys.exit(0)
        # bhat: bhat*grad(p).grad(qtilde)
        assert isequal(tt, self.solfwd[index][1], 1e-16)
        setfct(self.p, self.solfwd[index][0])
        self.q.vector().zero()
        self.ftimeincrfwdb()
        # ahat: ahat*p''*qtilde:
        self.ftimeincrfwda(index)
        # D'.dot(p)
#        if self.PDE.abc and index > 0:
#                setfct(self.p, \
#                self.solfwd[index+1][0] - self.solfwd[index-1][0])
#                self.v.vector().axpy(.5*self.invDt, self.Dp*self.p.vector())
        return -1.0*self.q.vector()

    def ftimeincrfwd_componentb(self):
        setfct(self.q, self.C*self.p.vector())
    def ftimeincrfwd_componenta(self, index):
        solfwdm = [[np.zeros(self.PDE.V.dim()), -self.PDE.Dt]]+self.solfwd[:-1]
        solfwdp = self.solfwd[1:]+[[np.zeros(self.PDE.V.dim()), self.PDE.times[-1]+self.PDE.Dt]]
        setfct(self.ptmp, solfwdm[index][0])
        self.p.vector().axpy(-0.5, self.ptmp.vector())
        setfct(self.ptmp, solfwdp[index][0])
        self.p.vector().axpy(-0.5, self.ptmp.vector())
        self.q.vector().axpy(-2.0*self.invDt*self.invDt, self.get_incra(self.p.vector()))
    def ftimeincrfwd_componentapass(self, index):
        pass

    #@profile
    def ftimeincradj(self, tt):
        """ Compute rhs for incremental adjoint at time tt """
        try:
            indexf = int(np.where(isequal(self.PDE.times, tt, 1e-14))[0])
            indexa = int(np.where(isequal(self.PDE.times[::-1], tt, 1e-14))[0])
        except:
            print 'Error in ftimeincradj at time {}'.format(tt)
            print np.min(np.abs(self.PDE.times-tt))
            sys.exit(0)
        # bhat: bhat*grad(ptilde).grad(v)
        assert isequal(tt, self.soladj[indexa][1], 1e-16)
        setfct(self.q, self.soladj[indexa][0])
        self.qhat.vector().zero()
        self.ftimeincradjb()
        # ahat: ahat*ptilde*q'':
        self.ftimeincradja(indexa)
        # B* B phat
        assert isequal(tt, self.solincrfwd[indexf][1], 1e-16)
        setfct(self.phat, self.solincrfwd[indexf][0])
        self.qhat.vector().axpy(1.0, self.obsop.incradj(self.phat, tt))
        # D'.dot(v)
#        if self.PDE.abc and indexa > 0:
#                setfct(self.v, \
#                self.soladj[indexa-1][0] - self.soladj[indexa+1][0])
#                self.vhat.vector().axpy(-.5*self.invDt, self.Dp*self.v.vector())
        return -1.0*self.qhat.vector()

    def ftimeincradj_componentb(self):
        setfct(self.qhat, self.C*self.q.vector())
    def ftimeincradj_componenta(self, indexa):
        soladjm = [[np.zeros(self.PDE.V.dim()), -self.PDE.Dt]]+self.soladj[:-1]
        soladjp = self.soladj[1:]+[[np.zeros(self.PDE.V.dim()), self.PDE.times[-1]+self.PDE.Dt]]
        setfct(self.ptmp, soladjm[indexa][0])
        self.q.vector().axpy(-0.5, self.ptmp.vector())
        setfct(self.ptmp, soladjp[indexa][0])
        self.q.vector().axpy(-0.5, self.ptmp.vector())
        self.qhat.vector().axpy(-2.0*self.invDt*self.invDt, self.get_incra(self.q.vector()))
    def ftimeincradj_componentapass(self, indexa):
        pass
        
    #@profile
    def mult(self, abhat, y):
        """
        mult(self, abhat, y): return y = Hessian * abhat
        inputs:
            y, abhat = Function(V).vector()
        """
        if self.invparam == 'ab':
            # get ahat, bhat:
            setfct(self.ab, abhat)
            ahat, bhat = self.ab.split(deepcopy=True)
            setfct(self.ahat, ahat)
            setfct(self.bhat, bhat)
        elif self.invparam == 'a':
            setfct(self.ahat, abhat)
            self.bhat.vector().zero()
        elif self.invparam == 'b':
            self.ahat.vector().zero()
            setfct(self.bhat, abhat)
        self.C = assemble(self.wkformrhsincrb)
        if not self.PDE.lump:   self.E = assemble(self.wkformrhsincra)
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
#        index = 0
#        if self.PDE.abc:
#            self.vD.vector().zero(); self.vhatD.vector().zero(); 
#            self.p1D.vector().zero(); self.p2D.vector().zero();
#            self.p1hatD.vector().zero(); self.p2hatD.vector().zero();
        for fwd, adj, incrfwd, incradj, fwdm, fwdp, incrfwdm, incrfwdp, fact in \
        zip(self.solfwd, reversed(self.soladj), \
        self.solincrfwd, reversed(self.solincradj), \
        [[np.zeros(self.PDE.V.dim()), -self.PDE.Dt]]+self.solfwd[:-1], \
        self.solfwd[1:]+[[np.zeros(self.PDE.V.dim()), self.PDE.times[-1]+self.PDE.Dt]], \
        [[np.zeros(self.PDE.V.dim()), -self.PDE.Dt]]+self.solincrfwd[:-1], \
        self.solincrfwd[1:]+[[np.zeros(self.PDE.V.dim()), self.PDE.times[-1]+self.PDE.Dt]], \
        self.factors):
            ttf, tta, ttf2 = incrfwd[1], incradj[1], fwd[1]
            assert isequal(ttf, tta, 1e-16), 'tfwd={}, tadj={}, reldiff={}'.\
            format(ttf, tta, abs(ttf-tta)/ttf)
            assert isequal(ttf, ttf2, 1e-16), 'tfwd={}, tadj={}, reldiff={}'.\
            format(ttf, ttf2, abs(ttf-ttf2)/ttf)
            # Hessian b
            setfct(self.p, fwd[0])
            setfct(self.q, adj[0])
            setfct(self.phat, incrfwd[0])
            setfct(self.qhat, incradj[0])
            self.hessianb(yb, fact)
            # Hessian a
            self.hessiana(ya, fact, fwdm, fwdp, incrfwdm, incrfwdp)
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
#            index += 1
        y.zero()
        if self.invparam == 'ab':
            assign(self.ab.sub(0), yaF)
            assign(self.ab.sub(1), ybF)
            y.axpy(1.0, self.ab.vector())
            # add regularization term
            y.axpy(self.alpha_reg, self.regularization.hessianab(self.ahat, self.bhat))
        elif self.invparam == 'a':
            y.axpy(1.0, ya)
            # add regularization term
            y.axpy(self.alpha_reg, self.regularization.hessian(abhat))
        elif self.invparam == 'b':
            y.axpy(1.0, yb)
            # add regularization term
            y.axpy(self.alpha_reg, self.regularization.hessian(abhat))

    def hessian_componenta(self, ya, fact, fwdm, fwdp, incrfwdm, incrfwdp):
        setfct(self.ptmp, fwdm[0])
        self.p.vector().axpy(-0.5, self.ptmp.vector())
        setfct(self.ptmp, fwdp[0])
        self.p.vector().axpy(-0.5, self.ptmp.vector())
        setfct(self.ptmp, incrfwdm[0])
        self.phat.vector().axpy(-0.5, self.ptmp.vector())
        setfct(self.ptmp, incrfwdp[0])
        self.phat.vector().axpy(-0.5, self.ptmp.vector())
        scalingdiff = -2.0*self.invDt*self.invDt
        setfct(self.p, self.p.vector()*scalingdiff)
        setfct(self.phat, self.phat.vector()*scalingdiff)
        ya.axpy(fact, self.get_hessiana())
    def hessian_componentapass(self, ya, fact, fwdm, fwdp, incrfwdm, incrfwdp):
        pass
    def hessian_componentb(self, yb, fact):
        yb.axpy(fact, assemble(self.wkformhessb))
    def hessian_componentbpass(self, yb, fact):
        pass

    def get_hessiana_full(self):
        return assemble(self.wkformhessa)
    def get_hessiana_lumped(self):
        return self.Mprime.get_gradient(self.phat.vector(), self.q.vector()) +\
        self.Mprime.get_gradient(self.p.vector(), self.qhat.vector())

    def get_incra_full(self, pvector):
        return self.E*pvector
    def get_incra_lumped(self, pvector):
        return self.Mprime.get_incremental(self.ahat.vector(), pvector)



    # SETTERS + UPDATE:
    def init_vector(self, x, dim):
        self.Mass.init_vector(x, dim)

    def update_PDE(self, parameters): self.PDE.update(parameters)

    def update_ab(self, medparam):
        """ medparam is a np.array containing both med parameters """
        setfct(self.ab, medparam)
        a, b = self.ab.split(deepcopy=True)
        self.update_PDE({'a':a, 'b':b})
    def update_a(self, medparam):
        self.update_PDE({'a':medparam})
    def update_b(self, medparam):
        self.update_PDE({'b':medparam})

    def set_abc(self, mesh, class_bc_abc, lumpD):  
        self.PDE.set_abc(mesh, class_bc_abc, lumpD)

    def backup_ab(self): 
        """ back-up current value of med param a and b """
        assign(self.m_bkup.sub(0), self.PDE.a)
        assign(self.m_bkup.sub(1), self.PDE.b)
    def backup_a(self):
        setfct(self.m_bkup, self.PDE.a)
    def backup_b(self):
        setfct(self.m_bkup, self.PDE.b)

    def restore_ab(self):    
        """ restore backed-up values of a and b """
        a, b = self.m_bkup.split(deepcopy=True)
        self.update_PDE({'a':a, 'b':b})
    def restore_a(self):
        self.update_PDE({'a':self.m_bkup})
    def restore_b(self):
        self.update_PDE({'b':self.m_bkup})

    def setsrcterm(self, ftime):    self.PDE.ftime = ftime

    # GETTERS:
    def getmcopyarray(self):    return self.m_bkup.vector().array()
    def getMGarray(self):   return self.MGv.array()
    def getprecond(self):
        return self.regularization.getprecond()

