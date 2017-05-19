import numpy as np

from dolfin import LinearOperator, Function, TestFunction, TrialFunction, \
assemble, inner, nabla_grad, dx, sqrt, LUSolver, assign, Constant, \
PETScKrylovSolver, MPI
from miscfenics import setfct, isequal, ZeroRegularization
from linalg.lumpedmatrixsolver import LumpedMassMatrixPrime
from fenicstools.optimsolver import compute_searchdirection, bcktrcklinesearch

class ObjectiveAcoustic(LinearOperator):
    """
    Computes data misfit, gradient and Hessian evaluation for the seismic
    inverse problem using acoustic wave data
    """
    # CONSTRUCTORS:
    def __init__(self, acousticwavePDE, sources, invparam='ab', regularization=None):
        """ 
        Input:
            acousticwavePDE should be an instantiation from class AcousticWave
        """
        self.PDE = acousticwavePDE
        self.PDE.exact = None
        self.fwdsource = sources
        # functions for gradient
        Vm = self.PDE.Vm
        V = self.PDE.V
        self.ab = Function(Vm*Vm)
        self.obsop = None   # Observation operator
        self.dd = None  # observations
        # decide whether ahat and bhat are used
        self.invparam = invparam
        if self.invparam == 'ab':
            self.MG = Function(Vm*Vm)
            self.MGv = self.MG.vector()
            self.Grad = Function(Vm*Vm)
            self.srchdir = Function(Vm*Vm)
            self.delta_m = Function(Vm*Vm)
            self.get_costreg = self.get_costreg_joint
            self.update_m = self.update_ab
            self.m_bkup = Function(Vm*Vm)
            self.backup_m = self.backup_ab
            self.restore_m = self.restore_ab
            self.assemble_hessian = self.assemble_hessianab
            self.ftimeincrfwda = self.ftimeincrfwd_componenta
            self.ftimeincrfwdb = self.ftimeincrfwd_componentb
            self.ftimeincradja = self.ftimeincradj_componenta
            self.ftimeincradjb = self.ftimeincradj_componentb
            self.hessiana = self.hessian_componenta
            self.hessianb = self.hessian_componentb
        else:
            self.MG = Function(Vm)
            self.MGv = self.MG.vector()
            self.Grad = Function(Vm)
            self.srchdir = Function(Vm)
            self.delta_m = Function(Vm)
            self.m_bkup = Function(Vm)
            if self.invparam == 'a':
                self.get_costreg = self.get_costreg_a
                self.update_m = self.update_a
                self.backup_m = self.backup_a
                self.restore_m = self.restore_a
                self.assemble_hessian = self.assemble_hessiana
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
                self.assemble_hessian = self.assemble_hessianb
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
        else:   
            self.regularization = regularization
            #self.TV = self.regularization.isTV()# does NOT seem to be used
            self.PD = self.regularization.isPD()
        self.alpha_reg = 1.0
        # gradient and Hessian
        self.p, self.q = Function(V), Function(V)
        self.phat, self.qhat = Function(V), Function(V)
        self.ahat, self.bhat = Function(Vm), Function(Vm)
        self.ptrial, self.ptest = TrialFunction(V), TestFunction(V)
        self.mtest, self.mtrial = TestFunction(Vm), TrialFunction(Vm)
        self.ppprhs = Function(V)
        self.ptmp = Function(V)
        if self.PDE.parameters['lumpM']:
            self.Mprime = LumpedMassMatrixPrime(Vm, V, self.PDE.M.ratio)
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
            self.mmtest, self.mmtrial = TestFunction(Vm*Vm), \
            TrialFunction(Vm*Vm)
        else:
            self.mmtest, self.mmtrial = self.mtest, self.mtrial
        weak_m =  inner(self.mmtrial, self.mmtest)*dx
        self.Mass = assemble(weak_m)
        # For serial only,
        #self.solverM = LUSolver("petsc")
        #self.solverM.parameters['reuse_factorization'] = True
        #self.solverM.parameters['symmetric'] = True
        # In parallel,
        self.solverM = PETScKrylovSolver("cg", "amg")
        self.solverM.parameters['report'] = False
        self.solverM.parameters['nonzero_initial_guess'] = True
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
#            self.vD, self.pD, self.p1D, self.p2D = Function(V), \
#            Function(V), Function(V), Function(V)
#            self.wkformgradD = inner(0.5*sqrt(self.PDE.rho/self.PDE.lam)\
#            *self.pD, self.vD*self.lamtest)*self.PDE.ds(1)
#            self.wkformDprime = inner(0.5*sqrt(self.PDE.rho/self.PDE.lam)\
#            *self.bhat*self.ptrial, self.ptest)*self.PDE.ds(1)
#            self.dp, self.dph, self.vhatD = Function(V), \
#            Function(V), Function(V)
#            self.p1hatD, self.p2hatD = Function(V), Function(V)
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
        Ricker = self.fwdsource[0]
        srcv = self.fwdsource[2]
        #TODO: modify to make fwdsource an iterable object that returns a source term
        # there should not be any source construction inside the solvefwd
        # function
        self.solfwd, self.Bp = [], []
        for ptsrc in self.fwdsource[1]:
            def srcterm(tt):
                srcv.zero()
                srcv.axpy(Ricker(tt), ptsrc)
                return srcv
            self.PDE.ftime = srcterm
            solfwd,_ = self.PDE.solve()
            self.solfwd.append(solfwd)
            # observations:
            Bp = np.zeros((len(self.obsop.PtwiseObs.Points),len(solfwd)))
            for index, sol in enumerate(solfwd):
                setfct(self.p, sol[0])
                Bp[:,index] = self.obsop.obs(self.p)
            self.Bp.append(Bp)
        if cost:
            assert not self.dd == None, "Provide data observations to compute cost"
            self.cost_misfit = 0.0
            for Bp, dd in zip(self.Bp, self.dd):
                self.cost_misfit += self.obsop.costfct(Bp, dd, self.PDE.times)
            self.cost_misfit /= len(self.Bp)
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
        self.soladj = []
        for Bp, dd in zip(self.Bp, self.dd):
            self.obsop.assemble_rhsadj(Bp, dd, self.PDE.times, self.PDE.bc)
            self.PDE.ftime = self.obsop.ftimeadj
            soladj,_ = self.PDE.solve()
            self.soladj.append(soladj)
        if grad:
            if self.invparam == 'ab':
                # split gradient parts a and b
                self.MGv.zero()
                MGa, MGb = self.MG.split(deepcopy=True)
                MGav, MGbv = MGa.vector(), MGb.vector()
                # loop over time
                for solfwd, soladj in zip(self.solfwd, self.soladj):
                    for fwd, adj, fact, fwdm, fwdp in \
                    zip(solfwd, reversed(soladj), self.factors,\
                    [[solfwd[0][0], -self.PDE.Dt]]+solfwd[:-1], \
                    solfwd[1:]+[[solfwd[0][0], self.PDE.times[-1]+self.PDE.Dt]]):
                        self.gradient_componentb(fact, fwd, adj, MGbv)
                        self.gradient_componenta(fact, fwdm, fwdp, adj, MGav)
                setfct(MGa, MGav/len(self.Bp))
                setfct(MGb, MGbv/len(self.Bp))
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
                    for solfwd, soladj in zip(self.solfwd, self.soladj):
                        for fwd, adj, fact, fwdm, fwdp in \
                        zip(solfwd, reversed(soladj), self.factors,\
                        [[solfwd[0][0], -self.PDE.Dt]]+solfwd[:-1], \
                        solfwd[1:]+[[solfwd[0][0], self.PDE.times[-1]+self.PDE.Dt]]):
                            setfct(self.p, fwd[0])
                            setfct(self.q, adj[0])
                            self.gradient_componenta(fact, fwdm, fwdp, adj, MGav)
                    setfct(self.MG, MGav/len(self.Bp))
                    # add regularization
                    self.MGv.axpy(self.alpha_reg, self.regularization.grad(self.PDE.a))
                elif self.invparam == 'b':
                    self.MGv.zero()
                    MGbv = self.MGv
                    # loop over time
                    for solfwd, soladj in zip(self.solfwd, self.soladj):
                        for fwd, adj, fact in \
                        zip(solfwd, reversed(soladj), self.factors):
                            self.gradient_componentb(fact, fwd, adj, MGbv)
                    setfct(self.MG, MGbv/len(self.Bp))
                    # add regularization
                    self.MGv.axpy(self.alpha_reg, self.regularization.grad(self.PDE.b))
            # compute Grad
            # When Grad very small (optim almost converged), first few residuals
            # may be flagged as diverging by PETSc. In that case, enlarge
            # divergence_limit
            try:
                self.solverM.solve(self.Grad.vector(), self.MGv)
            except:
                # Massive caveat: Hope that ALL processes throw an exception
                pseudoGradnorm = np.sqrt(self.MGv.inner(self.MGv))
                if pseudoGradnorm < 1e-8:
                    print '*** Warning: Increasing divergence_limit for Mass matrix solver'
                    self.solverM.parameters["divergence_limit"] = 1e6
                    self.solverM.solve(self.Grad.vector(), self.MGv)
                else:
                    print '*** Error: Problem with Mass matrix solver'
                    sys.exit(1)
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

    #TODO: gradient b 5 times more expensive that a with lumped mass matrix
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
        assert isequal(tt, self.solfwdi[index][1], 1e-16)
        setfct(self.p, self.solfwdi[index][0])
        self.q.vector().zero()
        self.ftimeincrfwdb()
        # ahat: ahat*p''*qtilde:
        self.ftimeincrfwda(index)
        # D'.dot(p)
#        if self.PDE.abc and index > 0:
#                setfct(self.p, \
#                self.solfwdi[index+1][0] - self.solfwdi[index-1][0])
#                self.v.vector().axpy(.5*self.invDt, self.Dp*self.p.vector())
        return -1.0*self.q.vector()

    def ftimeincrfwd_componentb(self):
        setfct(self.q, self.C*self.p.vector())
    def ftimeincrfwd_componenta(self, index):
        solfwdm = [[self.solfwdi[0][0], -self.PDE.Dt]]+self.solfwdi[:-1]
        solfwdp = self.solfwdi[1:]+[[self.solfwdi[0][0], self.PDE.times[-1]+self.PDE.Dt]]
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
        assert isequal(tt, self.soladji[indexa][1], 1e-16)
        setfct(self.q, self.soladji[indexa][0])
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
#                self.soladji[indexa-1][0] - self.soladji[indexa+1][0])
#                self.vhat.vector().axpy(-.5*self.invDt, self.Dp*self.v.vector())
        return -1.0*self.qhat.vector()

    def ftimeincradj_componentb(self):
        setfct(self.qhat, self.C*self.q.vector())
    def ftimeincradj_componenta(self, indexa):
        soladjm = [[self.soladji[0][0], -self.PDE.Dt]]+self.soladji[:-1]
        soladjp = self.soladji[1:]+[[self.soladji[0][0], self.PDE.times[-1]+self.PDE.Dt]]
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
        if not self.PDE.parameters['lumpM']:    self.E = assemble(self.wkformrhsincra)
        #if self.PDE.abc:    self.Dp = assemble(self.wkformDprime)
        # Compute Hessian*abhat
        self.ab.vector().zero()
        yaF, ybF = self.ab.split(deepcopy=True)
        ya, yb = yaF.vector(), ybF.vector()
        for self.solfwdi, self.soladji in zip(self.solfwd, self.soladj):
            # solve for phat
            self.PDE.set_fwd()
            self.PDE.ftime = self.ftimeincrfwd
            self.solincrfwd,_ = self.PDE.solve()
            # solve for vhat
            self.PDE.set_adj()
            self.PDE.ftime = self.ftimeincradj
            self.solincradj,_ = self.PDE.solve()
    #        index = 0
    #        if self.PDE.abc:
    #            self.vD.vector().zero(); self.vhatD.vector().zero(); 
    #            self.p1D.vector().zero(); self.p2D.vector().zero();
    #            self.p1hatD.vector().zero(); self.p2hatD.vector().zero();
            for fwd, adj, incrfwd, incradj, fwdm, fwdp, incrfwdm, incrfwdp, fact in \
            zip(self.solfwdi, reversed(self.soladji), \
            self.solincrfwd, reversed(self.solincradj), \
            [[self.solfwdi[0][0], -self.PDE.Dt]]+self.solfwdi[:-1], \
            self.solfwdi[1:]+[[self.solfwdi[0][0], self.PDE.times[-1]+self.PDE.Dt]], \
            [[self.solincrfwd[0][0], -self.PDE.Dt]]+self.solincrfwd[:-1], \
            self.solincrfwd[1:]+[[self.solincrfwd[0][0], self.PDE.times[-1]+self.PDE.Dt]], \
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
            y.axpy(1.0/len(self.Bp), self.ab.vector())
            # add regularization term
            y.axpy(self.alpha_reg, \
            self.regularization.hessianab(self.ahat.vector(), self.bhat.vector()))
        elif self.invparam == 'a':
            y.axpy(1.0/len(self.Bp), ya)
            # add regularization term
            y.axpy(self.alpha_reg, self.regularization.hessian(abhat))
        elif self.invparam == 'b':
            y.axpy(1.0/len(self.Bp), yb)
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
        """ medparam contains both med parameters """
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


    def assemble_hessianab(self):
        self.regularization.assemble_hessianab(self.PDE.a, self.PDE.b)
    def assemble_hessiana(self): 
        self.regularization.assemble_hessian(self.PDE.a)
    def assemble_hessianb(self): 
        self.regularization.assemble_hessian(self.PDE.b)


    # GETTERS:
    def getmcopy(self):         return self.m_bkup.vector()
    def getmcopyarray(self):    return self.getmcopy().array()
    def getMG(self):            return self.MGv
    def getMGarray(self):       return self.MGv.array()
    def getprecond(self):       return self.regularization.getprecond()



    def inversion(self, initial_medium, target_medium, mpicomm, \
    tolgrad=1e-10, tolcost=1e-14, maxnbNewtiter=50, myplot=None):
        """ solve inverse problem with that objective function """

        tolcg = 0.5
        mpirank = MPI.rank(mpicomm)
        if mpirank == 0:
            print '\t{:12s} {:10s} {:12s} {:12s} {:12s} {:10s} \t{:10s} {:12s} {:12s}'.format(\
            'iter', 'cost', 'misfit', 'reg', '|G|', 'medmisf', 'a_ls', 'tol_cg', 'n_cg')
        # set medium to initial_medium, then plot
        if self.invparam == 'a':    self.update_PDE({'a':initial_medium})
        elif self.invparam == 'b':    self.update_PDE({'b':initial_medium})
        elif self.invparam == 'ab':    
            a_init, b_init = initial_medium.split(deepcopy=True)
            self.update_PDE({'a':a_init, 'b':b_init})
        self._plotab(myplot, 'init')

        # start inversion
        dtruenorm = np.sqrt(target_medium.vector().\
        inner(self.Mass*target_medium.vector()))
        self.solvefwd_cost()
        for it in xrange(maxnbNewtiter):
            # compute gradient
            self.solveadj_constructgrad()
            gradnorm = np.sqrt(self.MGv.inner(self.Grad.vector()))
            if it == 0:   gradnorm0 = gradnorm
            if self.invparam == 'a':
                diff = self.PDE.a.vector() - target_medium.vector()
            elif self.invparam == 'b':
                diff = self.PDE.b.vector() - target_medium.vector()
            elif self.invparam == 'ab':
                assign(self.ab.sub(0), self.PDE.a)
                assign(self.ab.sub(1), self.PDE.b)
                diff = self.ab.vector() - target_medium.vector()
            medmisfit = np.sqrt(diff.inner(self.Mass*diff))
            if mpirank == 0:
                print '{:12d} {:12.4e} {:12.2e} {:12.2e} {:11.4e} {:10.2e} ({:4.2f})'.\
                format(it, self.cost, self.cost_misfit, self.cost_reg, \
                gradnorm, medmisfit, medmisfit/dtruenorm),
            # plots
            self._plotab(myplot, str(it))
            self._plotgrad(myplot, str(it))
            # stopping criterion (gradient)
            if gradnorm < gradnorm0 * tolgrad or gradnorm < 1e-12:
                if mpirank == 0:
                    print '\nGradient sufficiently reduced -- optimization stopped'
                break
            # compute search direction and plot
            #tolcg = min(0.5, np.sqrt(gradnorm/gradnorm0))
            tolcg = min(tolcg, np.sqrt(gradnorm/gradnorm0))
            self.assemble_hessian() # for nonlinear regularization functionals
            cgiter, cgres, cgid, tolcg = compute_searchdirection(self, 'Newt', tolcg)
            self._plotsrchdir(myplot, str(it))
            # perform line search
            cost_old = self.cost
            statusLS, LScount, alpha = bcktrcklinesearch(self, 12)
            cost = self.cost
            if mpirank == 0:
                print '{:11.3f} {:12.2e} {:10d}'.format(alpha, tolcg, cgiter)
            # perform line search for dual variable (TV-PD):
            if self.PD: self.regularization.update_w(self.srchdir.vector(), alpha)
            # stopping criterion (cost)
            if np.abs(cost-cost_old)/np.abs(cost_old) < tolcost:
                if mpirank == 0:
                    print 'Cost function stagnates -- optimization stopped'
                break


    def _plotab(self, myplot, index):
        """ plot media during inversion """
        if not myplot == None:
            if self.invparam == 'a' or self.invparam == 'ab':
                myplot.set_varname('a'+index)
                myplot.plot_vtk(self.PDE.a)
            if self.invparam == 'b' or self.invparam == 'ab':
                myplot.set_varname('b'+index)
                myplot.plot_vtk(self.PDE.b)

    def _plotgrad(self, myplot, index):
        """ plot grad during inversion """
        if not myplot == None:
            if self.invparam == 'a':
                myplot.set_varname('Grad_a'+index)
                myplot.plot_vtk(self.Grad)
            elif self.invparam == 'b':
                myplot.set_varname('Grad_b'+index)
                myplot.plot_vtk(self.Grad)
            elif self.invparam == 'ab':
                Ga, Gb = self.Grad.split(deepcopy=True)
                myplot.set_varname('Grad_a'+index)
                myplot.plot_vtk(Ga)
                myplot.set_varname('Grad_b'+index)
                myplot.plot_vtk(Gb)

    def _plotsrchdir(self, myplot, index):
        """ plot srchdir during inversion """
        if not myplot == None:
            if self.invparam == 'a':
                myplot.set_varname('srchdir_a'+index)
                myplot.plot_vtk(self.srchdir)
            elif self.invparam == 'b':
                myplot.set_varname('srchdir_b'+index)
                myplot.plot_vtk(self.srchdir)
            elif self.invparam == 'ab':
                Ga, Gb = self.srchdir.split(deepcopy=True)
                myplot.set_varname('srchdir_a'+index)
                myplot.plot_vtk(Ga)
                myplot.set_varname('srchdir_b'+index)
                myplot.plot_vtk(Gb)
