import numpy as np
from itertools import izip

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
    def __init__(self, mpicomm_global, acousticwavePDE, sources, \
    invparam='ab', regularization=None):
        """ 
        Input:
            acousticwavePDE should be an instantiation from class AcousticWave
        """
        self.mpicomm_global = mpicomm_global

        self.PDE = acousticwavePDE
        self.PDE.exact = None
        self.obsop = None   # Observation operator
        self.dd = None  # observations
        self.fwdsource = sources

        self.inverta = False
        self.invertb = False
        if 'a' in invparam:
            self.inverta = True
        if 'b' in invparam:
            self.invertb = True
        assert self.inverta + self.invertb > 0

        Vm = self.PDE.Vm
        V = self.PDE.V
        self.ab = Function(Vm*Vm)   # used for conversion (Vm,Vm)->Vm*Vm
        self.invparam = invparam
        self.MG = Function(Vm*Vm)
        self.MGv = self.MG.vector()
        self.Grad = Function(Vm*Vm)
        self.srchdir = Function(Vm*Vm)
        self.delta_m = Function(Vm*Vm)
        self.m_bkup = Function(Vm*Vm)
        LinearOperator.__init__(self, self.MGv, self.MGv)
        self.GN = False

        if regularization == None:  
            print '*** Warning: Using zero regularization'
            self.regularization = ZeroRegularization(Vm)
        else:   
            self.regularization = regularization
            self.PD = self.regularization.isPD()
        self.alpha_reg = 1.0

        self.p, self.q = Function(V), Function(V)
        self.phat, self.qhat = Function(V), Function(V)
        self.ahat, self.bhat = Function(Vm), Function(Vm)
        self.ptrial, self.ptest = TrialFunction(V), TestFunction(V)
        self.mtest, self.mtrial = TestFunction(Vm), TrialFunction(Vm)
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
            self.wkformhessaGN = inner(self.p*self.mtest, self.qhat)*dx
            self.get_hessiana = self.get_hessiana_full
            self.wkformrhsincra = inner(self.ahat*self.ptrial, self.ptest)*dx
            self.get_incra = self.get_incra_full
        self.wkformgradb = inner(self.mtest*nabla_grad(self.p), nabla_grad(self.q))*dx
        self.wkformgradbout = assemble(self.wkformgradb)
        self.wkformrhsincrb = inner(self.bhat*nabla_grad(self.ptrial), nabla_grad(self.ptest))*dx
        self.wkformhessb = inner(nabla_grad(self.phat)*self.mtest, nabla_grad(self.q))*dx \
        + inner(nabla_grad(self.p)*self.mtest, nabla_grad(self.qhat))*dx
        self.wkformhessbGN = inner(nabla_grad(self.p)*self.mtest, nabla_grad(self.qhat))*dx

        # Mass matrix:
        self.mmtest, self.mmtrial = TestFunction(Vm*Vm), TrialFunction(Vm*Vm)
        weak_m =  inner(self.mmtrial, self.mmtest)*dx
        self.Mass = assemble(weak_m)
        self.solverM = PETScKrylovSolver("cg", "jacobi")
        self.solverM.parameters["maximum_iterations"] = 2000
        self.solverM.parameters["absolute_tolerance"] = 1e-24
        self.solverM.parameters["relative_tolerance"] = 1e-24
        self.solverM.parameters["report"] = False
        self.solverM.parameters["error_on_nonconvergence"] = True 
        self.solverM.parameters["nonzero_initial_guess"] = False # True?
        self.solverM.set_operator(self.Mass)

        # Time-integration factors
        self.factors = np.ones(self.PDE.times.size)
        self.factors[0], self.factors[-1] = 0.5, 0.5
        self.factors *= self.PDE.Dt
        self.invDt = 1./self.PDE.Dt

        # Absorbing BCs
        if self.PDE.parameters['abc']:
            assert not self.PDE.parameters['lumpD']

            self.wkformgradaABC = inner(
            self.mtest*sqrt(self.PDE.b/self.PDE.a)*self.p, 
            self.q)*self.PDE.ds(1)
            self.wkformgradbABC = inner(
            self.mtest*sqrt(self.PDE.a/self.PDE.b)*self.p, 
            self.q)*self.PDE.ds(1)
            self.wkformgradaABCout = assemble(self.wkformgradaABC)
            self.wkformgradbABCout = assemble(self.wkformgradbABC)

            self.wkformincrrhsABC = inner(
            (self.ahat*sqrt(self.PDE.b/self.PDE.a)
             + self.bhat*sqrt(self.PDE.a/self.PDE.b))*self.ptrial,
            self.ptest)*self.PDE.ds(1)

            self.wkformhessaABC = inner(
            (self.bhat/sqrt(self.PDE.a*self.PDE.b) - 
            self.ahat*sqrt(self.PDE.b/(self.PDE.a*self.PDE.a*self.PDE.a)))
            *self.p*self.mtest, self.q)*self.PDE.ds(1)
            self.wkformhessbABC = inner(
            (self.ahat/sqrt(self.PDE.a*self.PDE.b) - 
            self.bhat*sqrt(self.PDE.a/(self.PDE.b*self.PDE.b*self.PDE.b)))
            *self.p*self.mtest, self.q)*self.PDE.ds(1)


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
        self.solfwd, self.solpfwd, self.solppfwd = [], [], [] 
        self.Bp = []

        #TODO: make fwdsource iterable to return source term
        Ricker = self.fwdsource[0]
        srcv = self.fwdsource[2]
        for ptsrc in self.fwdsource[1]:
            def srcterm(tt):
                srcv.zero()
                srcv.axpy(Ricker(tt), ptsrc)
                return srcv
            self.PDE.ftime = srcterm
            solfwd, solpfwd, solppfwd,_ = self.PDE.solve()
            self.solfwd.append(solfwd)
            self.solpfwd.append(solpfwd)
            self.solppfwd.append(solppfwd)

            Bp = np.zeros((len(self.obsop.PtwiseObs.Points),len(solfwd)))
            for index, sol in enumerate(solfwd):
                setfct(self.p, sol[0])
                Bp[:,index] = self.obsop.obs(self.p)
            self.Bp.append(Bp)

        if cost:
            assert not self.dd == None, "Provide data observations to compute cost"
            self.cost_misfit = 0.0
            for Bp, dd in izip(self.Bp, self.dd):
                self.cost_misfit += self.obsop.costfct(Bp, dd, self.PDE.times, self.factors)
            self.cost_misfit /= len(self.Bp)
            self.cost_reg = self.regularization.costab(self.PDE.a, self.PDE.b)
            self.cost = self.cost_misfit + self.alpha_reg*self.cost_reg

    def solvefwd_cost(self):    self.solvefwd(True)


    # ADJOINT PROBLEM + GRADIENT:
    #@profile
    def solveadj(self, grad=False):
        self.PDE.set_adj()
        self.soladj, self.solpadj, self.solppadj = [], [], []

        for Bp, dd in zip(self.Bp, self.dd):
            self.obsop.assemble_rhsadj(Bp, dd, self.PDE.times, self.PDE.bc)
            self.PDE.ftime = self.obsop.ftimeadj
            soladj,solpadj,solppadj,_ = self.PDE.solve()
            self.soladj.append(soladj)
            self.solpadj.append(solpadj)
            self.solppadj.append(solppadj)

        if grad:
            self.MG.vector().zero()
            MGa, MGb = self.MG.split(deepcopy=True)
            MGav, MGbv = MGa.vector(), MGb.vector()

            for solfwd, solpfwd, solppfwd, soladj in \
            izip(self.solfwd, self.solpfwd, self.solppfwd, self.soladj):

                for fwd, fwdp, fwdpp, adj, fact in \
                izip(solfwd, solpfwd, solppfwd, reversed(soladj), self.factors):
                    setfct(self.q, adj[0])
                    if self.inverta:
                        # gradient a
                        setfct(self.p, fwdpp[0])
                        MGav.axpy(fact, self.get_gradienta()) 
                    if self.invertb:
                        # gradient b
                        setfct(self.p, fwd[0])
                        assemble(form=self.wkformgradb, tensor=self.wkformgradbout)
                        MGbv.axpy(fact, self.wkformgradbout)

                    if self.PDE.parameters['abc']:
                        setfct(self.p, fwdp[0])
                        if self.inverta:
                            assemble(form=self.wkformgradaABC, tensor=self.wkformgradaABCout)
                            MGav.axpy(0.5*fact, self.wkformgradaABCout)
                        if self.invertb:
                            assemble(form=self.wkformgradbABC, tensor=self.wkformgradbABCout)
                            MGbv.axpy(0.5*fact, self.wkformgradbABCout)

            setfct(MGa, MGav/len(self.Bp))
            setfct(MGb, MGbv/len(self.Bp))
            self.MG.vector().zero()
            if self.inverta:
                assign(self.MG.sub(0), MGa)
            if self.invertb:
                assign(self.MG.sub(1), MGb)

            self.MG.vector().axpy(self.alpha_reg, \
            self.regularization.gradab(self.PDE.a, self.PDE.b))

            try:
                self.solverM.solve(self.Grad.vector(), self.MG.vector())
            except:
                # if |G|<<1, first residuals may diverge
                # caveat: Hope that ALL processes throw an exception
                pseudoGradnorm = np.sqrt(self.MGv.inner(self.MGv))
                if pseudoGradnorm < 1e-8:
                    print '*** Warning: Increasing divergence_limit for Mass matrix solver'
                    self.solverM.parameters["divergence_limit"] = 1e6
                    self.solverM.solve(self.Grad.vector(), self.MG.vector())
                else:
                    print '*** Error: Problem with Mass matrix solver'
                    sys.exit(1)

    def solveadj_constructgrad(self):   self.solveadj(True)

    def get_gradienta_lumped(self):
        return self.Mprime.get_gradient(self.p.vector(), self.q.vector())

    def get_gradienta_full(self):
        return assemble(self.wkformgrada)


    # HESSIAN:
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
#        assert isequal(tt, self.solfwdi[index][1], 1e-16)
        setfct(self.p, self.solfwdi[index][0])
        self.q.vector().zero()
        self.q.vector().axpy(1.0, self.C*self.p.vector())

        # ahat: ahat*p''*qtilde:
        setfct(self.p, self.solppfwdi[index][0])
        self.q.vector().axpy(1.0, self.get_incra(self.p.vector()))

        # ABC:
        if self.PDE.parameters['abc']:
            setfct(self.phat, self.solpfwdi[index][0])
            self.q.vector().axpy(0.5, self.Dp*self.phat.vector())

        return -1.0*self.q.vector()


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

        # B* B phat
#        assert isequal(tt, self.solincrfwd[indexf][1], 1e-16)
        setfct(self.phat, self.solincrfwd[indexf][0])
        self.qhat.vector().zero()
        self.qhat.vector().axpy(1.0, self.obsop.incradj(self.phat, tt))

        if not self.GN:
            # bhat: bhat*grad(ptilde).grad(v)
#            assert isequal(tt, self.soladji[indexa][1], 1e-16)
            setfct(self.q, self.soladji[indexa][0])
            self.qhat.vector().axpy(1.0, self.C*self.q.vector())

            # ahat: ahat*ptilde*q'':
            setfct(self.q, self.solppadji[indexa][0])
            self.qhat.vector().axpy(1.0, self.get_incra(self.q.vector()))

            # ABC:
            if self.PDE.parameters['abc']:
                setfct(self.phat, self.solpadji[indexa][0])
                self.qhat.vector().axpy(-0.5, self.Dp*self.phat.vector())

        return -1.0*self.qhat.vector()

    def get_incra_full(self, pvector):
        return self.E*pvector

    def get_incra_lumped(self, pvector):
        return self.Mprime.get_incremental(self.ahat.vector(), pvector)

        
    #@profile
    def mult(self, abhat, y):
        """
        mult(self, abhat, y): return y = Hessian * abhat
        inputs:
            y, abhat = Function(V).vector()
        """
        setfct(self.ab, abhat)
        ahat, bhat = self.ab.split(deepcopy=True)
        setfct(self.ahat, ahat)
        setfct(self.bhat, bhat)
        if not self.inverta:
            self.ahat.vector().zero()
        if not self.invertb:
            self.bhat.vector().zero()

        self.C = assemble(self.wkformrhsincrb)
        if not self.PDE.parameters['lumpM']:    self.E = assemble(self.wkformrhsincra)
        if self.PDE.parameters['abc']:  self.Dp = assemble(self.wkformincrrhsABC)

        # Compute Hessian*abhat
        self.ab.vector().zero()
        yaF, ybF = self.ab.split(deepcopy=True)
        ya, yb = yaF.vector(), ybF.vector()

        # iterate over sources:
        for self.solfwdi, self.solpfwdi, self.solppfwdi, \
        self.soladji, self.solpadji, self.solppadji \
        in izip(self.solfwd, self.solpfwd, self.solppfwd, \
        self.soladj, self.solpadj, self.solppadj):
            # incr. fwd
            self.PDE.set_fwd()
            self.PDE.ftime = self.ftimeincrfwd
            self.solincrfwd,solpincrfwd,self.solppincrfwd,_ = self.PDE.solve()

            # incr. adj
            self.PDE.set_adj()
            self.PDE.ftime = self.ftimeincradj
            solincradj,_,_,_ = self.PDE.solve()

            # assemble Hessian-vect product:
            for fwd, adj, fwdp, incrfwdp, \
            fwdpp, incrfwdpp, incrfwd, incradj, fact \
            in izip(self.solfwdi, reversed(self.soladji), self.solpfwdi, solpincrfwd, \
            self.solppfwdi, self.solppincrfwd, self.solincrfwd, reversed(solincradj), self.factors):
#                ttf, tta, ttf2 = incrfwd[1], incradj[1], fwd[1]
#                assert isequal(ttf, tta, 1e-16), 'tfwd={}, tadj={}, reldiff={}'.\
#                format(ttf, tta, abs(ttf-tta)/ttf)
#                assert isequal(ttf, ttf2, 1e-16), 'tfwd={}, tadj={}, reldiff={}'.\
#                format(ttf, ttf2, abs(ttf-ttf2)/ttf)

                setfct(self.q, adj[0])
                setfct(self.qhat, incradj[0])
                if self.invertb:
                    # Hessian b
                    setfct(self.p, fwd[0])
                    setfct(self.phat, incrfwd[0])
                    if self.GN:
                        yb.axpy(fact, assemble(self.wkformhessbGN))
                    else:
                        yb.axpy(fact, assemble(self.wkformhessb))

                if self.inverta:
                    # Hessian a
                    setfct(self.p, fwdpp[0])
                    setfct(self.phat, incrfwdpp[0])
                    ya.axpy(fact, self.get_hessiana())

                if self.PDE.parameters['abc']:
                    if not self.GN:
                        setfct(self.p, incrfwdp[0])
                        if self.inverta:
                            ya.axpy(0.5*fact, assemble(self.wkformgradaABC))
                        if self.invertb:
                            yb.axpy(0.5*fact, assemble(self.wkformgradbABC))

                    setfct(self.p, fwdp[0])
                    setfct(self.q, incradj[0])
                    if self.inverta:
                        ya.axpy(0.5*fact, assemble(self.wkformgradaABC))
                    if self.invertb:
                        yb.axpy(0.5*fact, assemble(self.wkformgradbABC))

                    if not self.GN:
                        setfct(self.q, adj[0])
                        if self.inverta:
                            ya.axpy(0.25*fact, assemble(self.wkformhessaABC))
                        if self.invertb:
                            yb.axpy(0.25*fact, assemble(self.wkformhessbABC))

        self.ab.vector().zero()
        if self.inverta:
            assign(self.ab.sub(0), yaF)
        if self.invertb:
            assign(self.ab.sub(1), ybF)
        y.zero()
        y.axpy(1.0/len(self.Bp), self.ab.vector())

        y.axpy(self.alpha_reg, \
        self.regularization.hessianab(self.ahat.vector(), self.bhat.vector()))

    def get_hessiana_full(self):
        if self.GN:
            return assemble(self.wkformhessaGN)
        else:
            return assemble(self.wkformhessa)

    def get_hessiana_lumped(self):
        if self.GN:
            return self.Mprime.get_gradient(self.p.vector(), self.qhat.vector())
        else:
            return self.Mprime.get_gradient(self.phat.vector(), self.q.vector()) +\
            self.Mprime.get_gradient(self.p.vector(), self.qhat.vector())


    def assemble_hessian(self):
        self.regularization.assemble_hessianab(self.PDE.a, self.PDE.b)



    # SETTERS + UPDATE:
    def update_PDE(self, parameters): self.PDE.update(parameters)

    def update_m(self, medparam):
        """ medparam contains both med parameters """
        setfct(self.ab, medparam)
        a, b = self.ab.split(deepcopy=True)
        self.update_PDE({'a':a, 'b':b})

    def backup_m(self): 
        """ back-up current value of med param a and b """
        assign(self.m_bkup.sub(0), self.PDE.a)
        assign(self.m_bkup.sub(1), self.PDE.b)

    def restore_m(self):    
        """ restore backed-up values of a and b """
        a, b = self.m_bkup.split(deepcopy=True)
        self.update_PDE({'a':a, 'b':b})

    def mediummisfit(self, target_medium):
        """
        Compute medium misfit at current position
        """
        assign(self.ab.sub(0), self.PDE.a)
        assign(self.ab.sub(1), self.PDE.b)
        diff = self.ab.vector() - target_medium.vector()
        Md = self.Mass*diff
        self.ab.vector().zero()
        self.ab.vector().axpy(1.0, Md)
        Mda, Mdb = self.ab.split(deepcopy=True)
        self.ab.vector().zero()
        self.ab.vector().axpy(1.0, diff)
        da, db = self.ab.split(deepcopy=True)
        medmisfita = np.sqrt(da.vector().inner(Mda.vector()))
        medmisfitb = np.sqrt(db.vector().inner(Mdb.vector()))
        return medmisfita, medmisfitb 



    # GETTERS:
    def getmbkup(self):         return self.m_bkup.vector()
    def getMG(self):            return self.MGv
    def getprecond(self):       return self.regularization.getprecond()



    # SOLVE INVERSE PROBLEM
    #@profile
    def inversion(self, initial_medium, target_medium, parameters_in=[], \
    boundsLS=None, myplot=None):
        """ 
        Solve inverse problem with that objective function 
        """
        parameters = {}
        parameters['reltolgrad']        = 1e-10
        parameters['abstolgrad']        = 1e-14
        parameters['tolcost']           = 1e-16
        parameters['maxiterNewt']       = 100
        parameters['nbGNsteps']         = 10
        parameters['maxtolcg']          = 0.5
        parameters['inexactCG']         = True
        parameters['isprint']           = False
        parameters.update(parameters_in)
        isprint = parameters['isprint']
        maxiterNewt = parameters['maxiterNewt']
        reltolgrad = parameters['reltolgrad']
        abstolgrad = parameters['abstolgrad']
        tolcost = parameters['tolcost']
        nbGNsteps = parameters['nbGNsteps']
        if parameters['inexactCG']:
            maxtolcg = parameters['maxtolcg']
        else:
            maxtolcg = 1e-12

        if isprint:
            print '\t{:12s} {:10s} {:12s} {:12s} {:12s} {:10s} \t\t\t{:10s} {:12s} {:12s}'.format(\
            'iter', 'cost', 'misfit', 'reg', '|G|', 'medmisf', 'a_ls', 'tol_cg', 'n_cg')

        a0, b0 = initial_medium.split(deepcopy=True)
        self.update_PDE({'a':a0, 'b':b0})
        self._plotab(myplot, 'init')

        Mab = self.Mass*target_medium.vector()
        self.ab.vector().zero()
        self.ab.vector().axpy(1.0, Mab)
        Ma, Mb = self.ab.split(deepcopy=True)
        at, bt = target_medium.split(deepcopy=True)
        atnorm = np.sqrt(at.vector().inner(Ma.vector()))
        btnorm = np.sqrt(bt.vector().inner(Mb.vector()))

        self.solvefwd_cost()
        for it in xrange(maxiterNewt):
            self.solveadj_constructgrad()
            gradnorm = np.sqrt(self.MGv.inner(self.Grad.vector()))
            if it == 0:   gradnorm0 = gradnorm

            medmisfita, medmisfitb = self.mediummisfit(target_medium)

            if isprint:
                print '{:12d} {:12.4e} {:12.2e} {:12.2e} {:11.4e} {:10.2e} ({:4.1f}%) {:10.2e} ({:4.1f}%)'.\
                format(it, self.cost, self.cost_misfit, self.cost_reg, gradnorm,\
                medmisfita, 100.0*medmisfita/atnorm, medmisfitb, 100.0*medmisfitb/btnorm),
            self._plotab(myplot, str(it))
            self._plotgrad(myplot, str(it))

            # Stopping criterion (gradient)
            if gradnorm < gradnorm0*reltolgrad or gradnorm < abstolgrad:
                if isprint:
                    print '\nGradient sufficiently reduced'
                    print 'Optimization converged'
                return

            # Compute search direction and plot
            tolcg = min(maxtolcg, np.sqrt(gradnorm/gradnorm0))
            self.GN = (it < nbGNsteps)
            cgiter, cgres, cgid = compute_searchdirection(self,
            {'method':'Newton', 'tolcg':tolcg}) # most time spent here
            if not self.inverta*self.invertb:
                srcha, srchb = self.srchdir.split(deepcopy=True)
                if not self.inverta:
                    srcha.vector().zero()
                    assign(self.srchdir.sub(0), srcha)
                if not self.invertb:
                    srchb.vector().zero()
                    assign(self.srchdir.sub(1), srchb)
            self._plotsrchdir(myplot, str(it))

            # Backtracking line search
            cost_old = self.cost
            statusLS, LScount, alpha = bcktrcklinesearch(self, parameters, boundsLS)
            cost = self.cost
            if isprint:
                print '{:11.3f} {:12.2e} {:10d}'.format(alpha, tolcg, cgiter)
            # Perform line search for dual variable (TV-PD):
            if self.PD: 
                self.regularization.update_w(self.srchdir.vector(), alpha)

            # Stopping criterion (cost)
            if np.abs(cost-cost_old)/np.abs(cost_old) < tolcost:
                if isprint:
                    print '\nCost function stagnates'
                    print 'Optimization aborted'
                return

        if isprint:
            print '\nMaximum number of Newton iterations reached'
            print 'Optimization aborted'




    # PLOTS:
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



    # SHOULD BE REMOVED:
    def set_abc(self, mesh, class_bc_abc, lumpD):  
        self.PDE.set_abc(mesh, class_bc_abc, lumpD)
    def init_vector(self, x, dim):
        self.Mass.init_vector(x, dim)
    def getmcopyarray(self):    return self.getmcopy().array()
    def getMGarray(self):       return self.MGv.array()
    def setsrcterm(self, ftime):    self.PDE.ftime = ftime

