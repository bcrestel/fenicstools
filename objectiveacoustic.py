import numpy as np
from itertools import izip

from dolfin import LinearOperator, Function, TestFunction, TrialFunction, \
assemble, inner, nabla_grad, dx, sqrt, LUSolver, assign, Constant, \
PETScKrylovSolver, MPI
from miscfenics import setfct, isequal, ZeroRegularization
from linalg.lumpedmatrixsolver import LumpedMassMatrixPrime
from fenicstools.optimsolver import compute_searchdirection, bcktrcklinesearch

from dolfin import FacetFunction, Measure

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
        self.obsop = None   # Observation operator
        self.dd = None  # observations

        self.fwdsource = sources

        Vm = self.PDE.Vm
        V = self.PDE.V
        self.ab = Function(Vm*Vm)
        self.invparam = invparam
        self.MG = Function(Vm*Vm)
        self.MGv = self.MG.vector()
        self.Grad = Function(Vm*Vm)
        self.srchdir = Function(Vm*Vm)
        self.delta_m = Function(Vm*Vm)
        self.m_bkup = Function(Vm*Vm)
        LinearOperator.__init__(self, self.MGv, self.MGv)

        if regularization == None:  
            print '*** Warning: Using zero regularization'
            self.regularization = ZeroRegularization()
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
            self.get_hessiana = self.get_hessiana_full
            self.wkformrhsincra = inner(self.ahat*self.ptrial, self.ptest)*dx
            self.get_incra = self.get_incra_full
        self.wkformgradb = inner(self.mtest*nabla_grad(self.p), nabla_grad(self.q))*dx
        self.wkformgradbout = assemble(self.wkformgradb)
        self.wkformrhsincrb = inner(self.bhat*nabla_grad(self.ptrial), nabla_grad(self.ptest))*dx
        self.wkformhessb = inner(nabla_grad(self.phat)*self.mtest, nabla_grad(self.q))*dx \
        + inner(nabla_grad(self.p)*self.mtest, nabla_grad(self.qhat))*dx

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

        #TODO: use self.factors inside misfit object
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
                self.cost_misfit += self.obsop.costfct(Bp, dd, self.PDE.times)
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
            self.MGv.zero()
            MGa, MGb = self.MG.split(deepcopy=True)
            MGav, MGbv = MGa.vector(), MGb.vector()

            for solfwd, solpfwd, solppfwd, soladj in \
            izip(self.solfwd, self.solpfwd, self.solppfwd, self.soladj):

                for fwd, fwdp, fwdpp, adj, fact in \
                izip(solfwd, solpfwd, solppfwd, reversed(soladj), self.factors):
                    setfct(self.q, adj[0])
                    # gradient a
                    setfct(self.p, fwdpp[0])
                    MGav.axpy(fact, self.get_gradienta()) 
                    # gradient b
                    setfct(self.p, fwd[0])
                    assemble(form=self.wkformgradb, tensor=self.wkformgradbout)
                    MGbv.axpy(fact, self.wkformgradbout)

                    if self.PDE.parameters['abc']:
                        setfct(self.p, fwdp[0])
                        assemble(form=self.wkformgradaABC, tensor=self.wkformgradaABCout)
                        MGav.axpy(0.5*fact, self.wkformgradaABCout)
                        assemble(form=self.wkformgradbABC, tensor=self.wkformgradbABCout)
                        MGbv.axpy(0.5*fact, self.wkformgradbABCout)

            setfct(MGa, MGav/len(self.Bp))
            setfct(MGb, MGbv/len(self.Bp))
            assign(self.MG.sub(0), MGa)
            assign(self.MG.sub(1), MGb)

            self.MGv.axpy(self.alpha_reg, \
            self.regularization.gradab(self.PDE.a, self.PDE.b))

            try:
                self.solverM.solve(self.Grad.vector(), self.MGv)
            except:
                # if |G|<<1, first residuals may diverge
                # caveat: Hope that ALL processes throw an exception
                pseudoGradnorm = np.sqrt(self.MGv.inner(self.MGv))
                if pseudoGradnorm < 1e-8:
                    print '*** Warning: Increasing divergence_limit for Mass matrix solver'
                    self.solverM.parameters["divergence_limit"] = 1e6
                    self.solverM.solve(self.Grad.vector(), self.MGv)
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

        # bhat: bhat*grad(ptilde).grad(v)
        assert isequal(tt, self.soladji[indexa][1], 1e-16)
        setfct(self.q, self.soladji[indexa][0])
        self.qhat.vector().zero()
        self.qhat.vector().axpy(1.0, self.C*self.q.vector())

        # ahat: ahat*ptilde*q'':
        setfct(self.q, self.solppadji[indexa][0])
        self.qhat.vector().axpy(1.0, self.get_incra(self.q.vector()))

        # B* B phat
        assert isequal(tt, self.solincrfwd[indexf][1], 1e-16)
        setfct(self.phat, self.solincrfwd[indexf][0])
        self.qhat.vector().axpy(1.0, self.obsop.incradj(self.phat, tt))

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

                # Hessian b
                setfct(self.p, fwd[0])
                setfct(self.q, adj[0])
                setfct(self.phat, incrfwd[0])
                setfct(self.qhat, incradj[0])
                yb.axpy(fact, assemble(self.wkformhessb))

                # Hessian a
                setfct(self.p, fwdpp[0])
                setfct(self.phat, incrfwdpp[0])
                ya.axpy(fact, self.get_hessiana())

                if self.PDE.parameters['abc']:
                    setfct(self.p, incrfwdp[0])
                    ya.axpy(0.5*fact, assemble(self.wkformgradaABC))
                    yb.axpy(0.5*fact, assemble(self.wkformgradbABC))

                    setfct(self.p, fwdp[0])
                    setfct(self.q, incradj[0])
                    ya.axpy(0.5*fact, assemble(self.wkformgradaABC))
                    yb.axpy(0.5*fact, assemble(self.wkformgradbABC))

                    setfct(self.q, adj[0])
                    ya.axpy(0.25*fact, assemble(self.wkformhessaABC))
                    yb.axpy(0.25*fact, assemble(self.wkformhessbABC))

        y.zero()
        assign(self.ab.sub(0), yaF)
        assign(self.ab.sub(1), ybF)
        y.axpy(1.0/len(self.Bp), self.ab.vector())

        y.axpy(self.alpha_reg, \
        self.regularization.hessianab(self.ahat.vector(), self.bhat.vector()))

    def get_hessiana_full(self):
        return assemble(self.wkformhessa)

    def get_hessiana_lumped(self):
        return self.Mprime.get_gradient(self.phat.vector(), self.q.vector()) +\
        self.Mprime.get_gradient(self.p.vector(), self.qhat.vector())



    # SETTERS + UPDATE:
    def init_vector(self, x, dim):
        self.Mass.init_vector(x, dim)

    def update_PDE(self, parameters): self.PDE.update(parameters)

    def update_m(self, medparam):
        """ medparam contains both med parameters """
        setfct(self.ab, medparam)
        a, b = self.ab.split(deepcopy=True)
        self.update_PDE({'a':a, 'b':b})

    def set_abc(self, mesh, class_bc_abc, lumpD):  
        self.PDE.set_abc(mesh, class_bc_abc, lumpD)

    def backup_m(self): 
        """ back-up current value of med param a and b """
        assign(self.m_bkup.sub(0), self.PDE.a)
        assign(self.m_bkup.sub(1), self.PDE.b)

    def restore_m(self):    
        """ restore backed-up values of a and b """
        a, b = self.m_bkup.split(deepcopy=True)
        self.update_PDE({'a':a, 'b':b})

    def setsrcterm(self, ftime):    self.PDE.ftime = ftime



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
            self.regularization.assemble_hessianab(self.PDE.a, self.PDE.b)
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
