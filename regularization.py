import sys
import numpy as np

from dolfin import sqrt, inner, nabla_grad, grad, dx, \
Function, TestFunction, TrialFunction, Vector, assemble, solve, \
Constant, plot, interactive, assign, FunctionSpace, interpolate, Expression, \
PETScKrylovSolver, PETScLUSolver, LUSolver, mpi_comm_world, PETScMatrix, \
as_backend_type, norm
from miscfenics import isVector, setfct
from linalg.miscroutines import get_diagonal, setupPETScmatrix, compute_eigfenics
from hippylib.linalg import MatMatMult, Transpose, pointwiseMaxCount

import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc


#----------------------------------------------------------------------
#----------------------------------------------------------------------
class TV():
    """
    Define Total Variation regularization
    """

    def __init__(self, parameters=[]):
        """
        TV regularization in primal form:
                |f|_TV = int k(x) sqrt{|grad f|^2 + eps} dx 
        Input parameters:
            * k = regularization parameter
            * eps = regularization constant (see above)
            * GNhessian = use GN format (aka, lagged diffusivity) (bool)
            * PCGN = use GN Hessian to precondition (bool); only used if 'GNhessian = False'
            * print (bool)
        """

        self.parameters = {}
        self.parameters['k']            = 1.0
        self.parameters['eps']          = 1e-2
        self.parameters['GNhessian']    = False
        self.parameters['PCGN']         = False
        self.parameters['print']        = False

        assert parameters.has_key('Vm')
        self.parameters.update(parameters)
        GN = self.parameters['GNhessian']
        self.Vm = self.parameters['Vm']
        eps = self.parameters['eps']
        k = self.parameters['k']
        isprint = self.parameters['print']

        self.m = Function(self.Vm)
        test, trial = TestFunction(self.Vm), TrialFunction(self.Vm)
        factM = 1e-2*k
        M = assemble(inner(test, trial)*dx)
        self.sMass = M*factM

        self.Msolver = PETScKrylovSolver('cg', 'jacobi')
        self.Msolver.parameters["maximum_iterations"] = 2000
        self.Msolver.parameters["relative_tolerance"] = 1e-24
        self.Msolver.parameters["absolute_tolerance"] = 1e-24
        self.Msolver.parameters["error_on_nonconvergence"] = True 
        self.Msolver.parameters["nonzero_initial_guess"] = False 
        self.Msolver.set_operator(M)

        self.fTV = inner(nabla_grad(self.m), nabla_grad(self.m)) + Constant(eps)
        self.kovsq = Constant(k) / sqrt(self.fTV)

        self.wkformcost = Constant(k) * sqrt(self.fTV) * dx

        self.wkformgrad = self.kovsq*inner(nabla_grad(self.m), nabla_grad(test))*dx

        self.wkformGNhess = self.kovsq*inner(nabla_grad(trial), nabla_grad(test))*dx
        self.wkformFhess = self.kovsq*( \
        inner(nabla_grad(trial), nabla_grad(test)) - \
        inner(nabla_grad(self.m), nabla_grad(test))*\
        inner(nabla_grad(trial), nabla_grad(self.m))/self.fTV )*dx
        if isprint: print 'TV regularization',
        if GN: 
            self.wkformhess = self.wkformGNhess
            if isprint: print ' -- GN Hessian',
        else:   
            self.wkformhess = self.wkformFhess
            if isprint: print ' -- full Hessian',
        if isprint: 
            if self.parameters['PCGN']:
                print ' -- PCGN',
            print ' -- k={}, eps={}'.format(self.parameters['k'], self.parameters['eps'])

        try:
            solver = PETScKrylovSolver('cg', 'hypre_amg')
            self.amgprecond = 'hypre_amg'
        except:
            self.amgprecond = 'petsc_amg'


    def isTV(self): return True
    def isPD(self): return False


    def cost(self, m_in):
        """ returns the cost functional for self.m=m_in """
        setfct(self.m, m_in)
        return assemble(self.wkformcost)

    def costvect(self, m_in):   return self.cost(m_in)


    def grad(self, m_in):
        """ returns the gradient (in vector format) evaluated at self.m = m_in """
        setfct(self.m, m_in)
        return assemble(self.wkformgrad)

    def gradvect(self, m_in):   return self.grad(m_in)


    def assemble_hessian(self, m_in):
        """ Assemble the Hessian of TV at m_in """
        setfct(self.m, m_in)
        self.H = assemble(self.wkformhess)
        PCGN = self.parameters['PCGN']
        if PCGN:
            HGN = assemble(self.wkformGNhess)
            self.precond = HGN + self.sMass
        else:
            self.precond = self.H + self.sMass

    def assemble_GNhessian(self, m_in):
        """ Assemble the Gauss-Newton Hessian at m_in 
        Not used anymore (wkformhess selects GN Hessian if needed)
        Left here for back-compatibility """
        setfct(self.m, m_in)
        self.H = assemble(self.wkformGNhess)
        self.precond = self.H + self.sMass

    def hessian(self, mhat):
        """ returns the Hessian applied along a direction mhat """
        isVector(mhat)
        return self.H * mhat


    def getprecond(self):
        """ Precondition by inverting the TV Hessian """

        solver = PETScKrylovSolver('cg', self.amgprecond)
        solver.parameters["maximum_iterations"] = 2000
        solver.parameters["relative_tolerance"] = 1e-24
        solver.parameters["absolute_tolerance"] = 1e-24
        solver.parameters["error_on_nonconvergence"] = True 
        solver.parameters["nonzero_initial_guess"] = False 

        # used to compare iterative application of preconditioner 
        # with exact application of preconditioner:
        #solver = PETScLUSolver("petsc")
        #solver.parameters['symmetric'] = True
        #solver.parameters['reuse_factorization'] = True

        solver.set_operator(self.precond)
        return solver


    def init_vector(self, u, dim):
        self.sMass.init_vector(u, dim)



#----------------------------------------------------------------------
#----------------------------------------------------------------------
class TVPD():
    """ Total variation using primal-dual Newton """

    def __init__(self, parameters, mpicomm=PETSc.COMM_WORLD):
        """ 
        TV regularization in primal-dual format
        Input parameters:
            * k = regularization parameter
            * eps = regularization constant (see above)
            * rescaledradiusdual = radius of dual set
            * exact = use full TV (bool)
            * PCGN = use GN Hessian to precondition (bool); 
            only used if 'GNhessian = False';
            not recommended for performance but can help avoid num.instability
            * print (bool)
        """

        self.parameters = {}
        self.parameters['k']                    = 1.0
        self.parameters['eps']                  = 1e-2
        self.parameters['rescaledradiusdual']   = 1.0
        self.parameters['exact']                = False
        self.parameters['PCGN']                 = False
        self.parameters['print']                = False

        assert parameters.has_key('Vm')
        self.parameters.update(parameters)
        self.Vm = self.parameters['Vm'] 
        k = self.parameters['k']
        eps = self.parameters['eps']
        exact = self.parameters['exact']

        self.m = Function(self.Vm)
        testm = TestFunction(self.Vm)
        trialm = TrialFunction(self.Vm)

        # WARNING: should not be changed.
        # As it is, code only works with DG0
        if self.parameters.has_key('Vw'):
            Vw = self.parameters['Vw']
        else:
            Vw = FunctionSpace(self.Vm.mesh(), 'DG', 0)
        self.wx = Function(Vw)
        self.wxrs = Function(Vw)   # re-scaled dual variable
        self.wxhat = Function(Vw)
        self.gwx = Function(Vw)
        self.wy = Function(Vw)
        self.wyrs = Function(Vw)   # re-scaled dual variable
        self.wyhat = Function(Vw)
        self.gwy = Function(Vw)
        self.wxsq = Vector()
        self.wysq = Vector()
        self.normw = Vector()
        self.factorw = Vector()
        testw = TestFunction(Vw)
        trialw = TrialFunction(Vw)

        normm = inner(nabla_grad(self.m), nabla_grad(self.m))
        TVnormsq = normm + Constant(eps)
        TVnorm = sqrt(TVnormsq)
        self.wkformcost = Constant(k)*TVnorm*dx
        if exact:
            sys.exit(1)
#            self.w = nabla_grad(self.m)/TVnorm # full Hessian
#            self.Htvw = inner(Constant(k) * nabla_grad(testm), self.w) * dx

        self.misfitwx = inner(testw, self.wx*TVnorm - self.m.dx(0))*dx
        self.misfitwy = inner(testw, self.wy*TVnorm - self.m.dx(1))*dx

        self.Htvx = assemble(inner(Constant(k)*testm.dx(0), trialw)*dx)
        self.Htvy = assemble(inner(Constant(k)*testm.dx(1), trialw)*dx)

        self.massw = inner(TVnorm*testw, trialw)*dx

        invMwMat, VDM, VDM = setupPETScmatrix(Vw, Vw, 'aij', mpicomm)
        for ii in VDM.dofs():
            invMwMat[ii, ii] = 1.0
        invMwMat.assemblyBegin()
        invMwMat.assemblyEnd()
        self.invMwMat = PETScMatrix(invMwMat)
        self.invMwd = Vector()
        self.invMwMat.init_vector(self.invMwd, 0)
        self.invMwMat.init_vector(self.wxsq, 0)
        self.invMwMat.init_vector(self.wysq, 0)
        self.invMwMat.init_vector(self.normw, 0)
        self.invMwMat.init_vector(self.factorw, 0)

        u = Function(Vw)
        if u.rank() == 0:
            ones = ("1.0")
        elif u.rank() == 1:
            ones = (("1.0", "1.0"))
        else:
            sys.exit(1)
        u = interpolate(Expression(ones), Vw)
        self.one = u.vector()

        self.wkformAx = inner(testw, trialm.dx(0) - \
        self.wx * inner(nabla_grad(self.m), nabla_grad(trialm)) / TVnorm) * dx
        self.wkformAxrs = inner(testw, trialm.dx(0) - \
        self.wxrs * inner(nabla_grad(self.m), nabla_grad(trialm)) / TVnorm) * dx
        self.wkformAy = inner(testw, trialm.dx(1) - \
        self.wy * inner(nabla_grad(self.m), nabla_grad(trialm)) / TVnorm) * dx
        self.wkformAyrs = inner(testw, trialm.dx(1) - \
        self.wyrs * inner(nabla_grad(self.m), nabla_grad(trialm)) / TVnorm) * dx

        kovsq = Constant(k) / TVnorm
        self.wkformGNhess = kovsq*inner(nabla_grad(trialm), nabla_grad(testm))*dx
        factM = 1e-2*k
        M = assemble(inner(testm, trialm)*dx)
        self.sMass = M*factM

        self.Msolver = PETScKrylovSolver('cg', 'jacobi')
        self.Msolver.parameters["maximum_iterations"] = 2000
        self.Msolver.parameters["relative_tolerance"] = 1e-24
        self.Msolver.parameters["absolute_tolerance"] = 1e-24
        self.Msolver.parameters["error_on_nonconvergence"] = True 
        self.Msolver.parameters["nonzero_initial_guess"] = False 
        self.Msolver.set_operator(M)

        if self.parameters['print']:
            print 'TV regularization -- primal-dual method',
            if self.parameters['PCGN']:
                print ' -- PCGN',
            print ' -- k={}, eps={}'.format(self.parameters['k'], self.parameters['eps'])
        try:
            solver = PETScKrylovSolver('cg', 'hypre_amg')
            self.amgprecond = 'hypre_amg'
        except:
            self.amgprecond = 'petsc_amg'


    def isTV(self): return True
    def isPD(self): return True

    
    def cost(self, m):
        """ evaluate the cost functional at m """
        setfct(self.m, m)
        return assemble(self.wkformcost)

    def costvect(self, m_in):   return self.cost(m_in)


    def _assemble_invMw(self):
        """ Assemble inverse of matrix Mw,
        weighted mass matrix in dual space """
        # WARNING: only works if Mw is diagonal (e.g, DG0)
        Mw = assemble(self.massw)
        Mwd = get_diagonal(Mw)
        as_backend_type(self.invMwd).vec().pointwiseDivide(\
            as_backend_type(self.one).vec(),\
            as_backend_type(Mwd).vec())
        self.invMwMat.set_diagonal(self.invMwd)


    def grad(self, m):
        """ compute the gradient at m """
        setfct(self.m, m)
        self._assemble_invMw()

        self.gwx.vector().zero()
        self.gwx.vector().axpy(1.0, assemble(self.misfitwx))
        normgwx = norm(self.gwx.vector())

        self.gwy.vector().zero()
        self.gwy.vector().axpy(1.0, assemble(self.misfitwy))
        normgwy = norm(self.gwy.vector())

        if self.parameters['print']:
            print '|gw|={}'.format(np.sqrt(normgwx**2 + normgwy**2))

        return self.Htvx*(self.wx.vector() - self.invMwd*self.gwx.vector()) \
        + self.Htvy*(self.wy.vector() - self.invMwd*self.gwy.vector())
        #return assemble(self.Htvw) - self.Htv*(self.invMwd*self.gw.vector())

    def gradvect(self, m_in):   return self.grad(m_in)


    def assemble_hessian(self, m):
        """ build Hessian matrix at given point m """
        setfct(self.m, m)
        self._assemble_invMw()

        self.Ax = assemble(self.wkformAx)
        Hxasym = MatMatMult(self.Htvx, MatMatMult(self.invMwMat, self.Ax))
        Hx = (Hxasym + Transpose(Hxasym)) * 0.5
        Axrs = assemble(self.wkformAxrs)
        Hxrsasym = MatMatMult(self.Htvx, MatMatMult(self.invMwMat, Axrs))
        Hxrs = (Hxrsasym + Transpose(Hxrsasym)) * 0.5

        self.Ay = assemble(self.wkformAy)
        Hyasym = MatMatMult(self.Htvy, MatMatMult(self.invMwMat, self.Ay))
        Hy = (Hyasym + Transpose(Hyasym)) * 0.5
        Ayrs = assemble(self.wkformAyrs)
        Hyrsasym = MatMatMult(self.Htvy, MatMatMult(self.invMwMat, Ayrs))
        Hyrs = (Hyrsasym + Transpose(Hyrsasym)) * 0.5

        self.H = Hx + Hy
        self.Hrs = Hxrs + Hyrs

        PCGN = self.parameters['PCGN']
        if PCGN:
            HGN = assemble(self.wkformGNhess)
            self.precond = HGN + self.sMass
        else:
            self.precond = self.Hrs + self.sMass


    def hessian(self, mhat):
        return self.Hrs * mhat


    def compute_what(self, mhat):
        """ Compute update direction for what, given mhat """
        self.wxhat.vector().zero()
        self.wxhat.vector().axpy(1.0, self.invMwd*(self.Ax*mhat - self.gwx.vector()))
        normwxhat = norm(self.wxhat.vector())

        self.wyhat.vector().zero()
        self.wyhat.vector().axpy(1.0, self.invMwd*(self.Ay*mhat - self.gwy.vector()))
        normwyhat = norm(self.wyhat.vector())

        if self.parameters['print']:
            print '|what|={}'.format(np.sqrt(normwxhat**2 + normwyhat**2))


    def update_w(self, mhat, alphaLS, compute_what=True):
        """ update dual variable in direction what 
        and update re-scaled version """
        if compute_what:    self.compute_what(mhat)
        self.wx.vector().axpy(alphaLS, self.wxhat.vector())
        self.wy.vector().axpy(alphaLS, self.wyhat.vector())

        # rescaledradiusdual=1.0: checked empirically to be max radius acceptable
        rescaledradiusdual = self.parameters['rescaledradiusdual']    
        # wx**2
        as_backend_type(self.wxsq).vec().pointwiseMult(\
            as_backend_type(self.wx.vector()).vec(),\
            as_backend_type(self.wx.vector()).vec())
        # wy**2
        as_backend_type(self.wysq).vec().pointwiseMult(\
            as_backend_type(self.wy.vector()).vec(),\
            as_backend_type(self.wy.vector()).vec())
        # |w|
        self.normw = self.wxsq + self.wysq
        as_backend_type(self.normw).vec().sqrtabs()
        # |w|/r
        as_backend_type(self.normw).vec().pointwiseDivide(\
            as_backend_type(self.normw).vec(),\
            as_backend_type(self.one*rescaledradiusdual).vec())
        # max(1.0, |w|/r)
#        as_backend_type(self.factorw).vec().pointwiseMax(\
#            as_backend_type(self.one).vec(),\
#            as_backend_type(self.normw).vec())
        count = pointwiseMaxCount(self.factorw, self.normw, 1.0)
        # rescale wx and wy
        as_backend_type(self.wxrs.vector()).vec().pointwiseDivide(\
            as_backend_type(self.wx.vector()).vec(),\
            as_backend_type(self.factorw).vec())
        as_backend_type(self.wyrs.vector()).vec().pointwiseDivide(\
            as_backend_type(self.wy.vector()).vec(),\
            as_backend_type(self.factorw).vec())

        minf = self.factorw.min()
        maxf = self.factorw.max()
        if self.parameters['print']:
#            print 'min(factorw)={}, max(factorw)={}'.format(minf, maxf)
            print 'perc. dual entries rescaled={:.2f} %, min(factorw)={}, max(factorw)={}'.format(\
            100.*float(count)/self.factorw.size(), minf, maxf)


    def getprecond(self):
        """ Precondition by inverting the TV Hessian """

        solver = PETScKrylovSolver('cg', self.amgprecond)
        solver.parameters["maximum_iterations"] = 2000
        solver.parameters["relative_tolerance"] = 1e-24
        solver.parameters["absolute_tolerance"] = 1e-24
        solver.parameters["error_on_nonconvergence"] = True 
        solver.parameters["nonzero_initial_guess"] = False 

        # used to compare iterative application of preconditioner 
        # with exact application of preconditioner:
        #solver = PETScLUSolver("petsc")
        #solver.parameters['symmetric'] = True
        #solver.parameters['reuse_factorization'] = True

        solver.set_operator(self.precond)

        return solver


    def init_vector(self, u, dim):
        self.sMass.init_vector(u, dim)










#### MATRIX-FREE VERSION -- NOT SUITABLE FOR FENICS PRECONDITIONERS
#    def grad(self, m):
#        """ compute the gradient at m (and what2) """
#
#        setfct(self.m, m)
#        self._assemble_invMw()
#
#        self.what2.vector().zero()
#        self.what2.vector().axpy(-1.0, self.invMwd * assemble(self.misfitw))
#        #print '|what2|={}'.format(np.linalg.norm(self.what2.vector().array()))
#
#        return assemble(self.wkformgrad) + self.Htv*self.what2.vector()
#        #return assemble(self.wkformgrad)
#
#
#    def assemble_hessian(self, m):
#        """ For PD-TV, we do not really assemble the Hessian,
#        but instead assemble the matrices that will be inverted
#        in the evaluation of what """
#
#        setfct(self.m, m)
#        self._assemble_invMw()
#
#        self.Ars = assemble(self.wkformArs)
#        #self.A = assemble(self.wkformA)
#        self.A = self.Ars   # using re-scaled dual variable for Hessian
#        self.yA, self.xA = Vector(), Vector()
#        self.A.init_vector(self.yA, 1)
#        self.A.init_vector(self.xA, 0)
#
#
#    def hessian(self, mhat):
#        """ evaluate H*mhat, with H symmetric
#        mhat must be a dolfin vector """
#
#        rhswhat1 = self.A * mhat
#        self.what1.vector().zero()
#        self.what1.vector().axpy(1.0, self.invMwd * rhswhat1)
#
#        self.xH.zero()
#        self.xH.axpy(1.0, mhat)
#        self.Htv.transpmult(self.xH , self.rhswhat1s)
#        self.what1s.vector().zero()
#        self.what1s.vector().axpy(1.0, self.invMwd * self.rhswhat1s)
#
#        self.what.vector().zero()
#        self.what.vector().axpy(1.0, self.what1.vector())
#        self.what.vector().axpy(1.0, self.what2.vector())
#
#        self.xA.zero()
#        self.xA.axpy(1.0, self.what1s.vector())
#        self.A.transpmult(self.xA, self.yA)
#        return 0.5*(self.Htv * self.what.vector() + self.yA)
#        #return self.Htv * self.what.vector()
#
#
#    def hessianrs(self, mhat):
#        """ evaluate H*mhat, with H symmetric and positive definite
#        mhat must be a dolfin vector """
#
#        rhswhat1 = self.Ars * mhat
#
#        self.xH.zero()
#        self.xH.axpy(1.0, mhat)
#        self.Htv.transpmult(self.xH , self.rhswhat1s)
#
#        self.wtmp.vector().zero()
#        self.wtmp.vector().axpy(1.0, self.invMwd * rhswhat1)
#        self.wtmp.vector().axpy(1.0, self.what2.vector())
#
#        self.xA.zero()
#        self.xA.axpy(1.0, self.invMwd * self.rhswhat1s)
#        self.Ars.transpmult(self.xA, self.yA)
#
#        return 0.5*(self.Htv * self.wtmp.vector() + self.yA)

