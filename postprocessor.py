import numpy as np

from dolfin import norm, Function, interpolate, MPI, mpi_comm_world

class PostProcessor():
    """Handles printing of results
    and stopping criteria for optimization"""

    # Instantiation
    def __init__(self, meth, Vm, mtrue, maxnbLS=15, mycomm=mpi_comm_world()):
        self.meth = meth
        if self.meth == 'Newt': self.Newt = True
        else:   self.Newt = False
        self.Vm = Vm
        self.mtrue = interpolate(mtrue, self.Vm)
        self.diff = Function(self.Vm)
        self.normmtrue = norm(self.mtrue)   # this is a global result (somehow)
        self.maxnbLS = maxnbLS
        self._createoutputlines()
        self.index = 0
        self.gradnorminit = None
        # MPI:
        self.mycomm = mycomm
        self.myrank = MPI.rank(self.mycomm)

    def setnormmtrue(self, normtrue):  self.normmtrue = normmtrue

    def _createoutputlines(self):
        self.titleline = ('{:2s} '+\
        '{:12s} {:12s} {:12s} \t'+\
        '{:10s} {:6s} \t'+\
        '{:12s} {:8s} {:6s} \t'+\
        '{:10s} {:3s} \t').format('it', \
        '    cost', '    datamis', '    regul', \
        '  Mmis-abs', '  rel', \
        '||grad||-abs', '   rel', '  G.p', \
        'LS-length', 'iter')
        if self.Newt:
            self.titleline = self.titleline + \
            '{:10s} {:5s} {:10s}'.format('  CG-tol', 'iter', 'finalres')

        self.dataline0 = '{:2d} {:12.5e} {:12.5e} {:12.5e} \t{:10.2e} {:6.3f}'
        self.dataline = '{:2d} '+\
        '{:12.5e} {:12.5e} {:12.5e} \t'+\
        '{:10.2e} {:6.3f} \t'+\
        '{:12.5e} {:8.2e} {:6.2f} \t'+\
        '{:10.3e} {:3d} \t\t'
        if self.Newt:
            self.dataline = self.dataline + '{:10.1e} {:3d} {:10.1e}'

    def errornorm(self, MM, m):
        self.diff.vector()[:] = (m.vector() - self.mtrue.vector()).array()
        return np.sqrt(np.dot(self.diff.vector().array(), \
        (MM * self.diff.vector()).array()))

    def getResults0(self, Obj):
        """Get results before first step of iteration"""
        self.index = 0
        # Cost
        self.costloc, self.misfitloc, self.regulloc = Obj.getcostloc()
        self.misfit = MPI.sum(self.mycomm, self.misfitloc)
        self.regul = MPI.sum(self.mycomm, self.regulloc)
        self.cost = MPI.sum(self.mycomm, self.costloc)
        # Med Misfit
        self.medmisfitloc = self.errornorm(Obj.getMass(), Obj.getm())
        self.medmisfit = np.sqrt(MPI.sum(self.mycomm, self.medmisfitloc**2))
        self.medmisfitrel = self.medmisfit/self.normmtrue
        # Grad
        self.gradnorm = 0.0
        self.gradnorminit = None

    def getResults(self, Obj, LSresults, CGresults, index=None):
        """LSresults = [LSsuccess, LScount, ratio]
        CGresults = [tolcg]"""
        if index == None:   self.index += 1
        else:   self.index = index

        if self.Newt and CGresults == None and self.index > 0:
            raise ValueError(\
            "CGresults must be provided when using Newton method")
        # Cost
        self.costloc, self.misfitloc, self.regulloc = Obj.getcostloc()
        self.misfit = MPI.sum(self.mycomm, self.misfitloc)
        self.regul = MPI.sum(self.mycomm, self.regulloc)
        self.cost = MPI.sum(self.mycomm, self.costloc)
        # Med Misfit
        self.medmisfitloc = self.errornorm(Obj.getMass(), Obj.getm())
        self.medmisfit = np.sqrt(MPI.sum(self.mycomm, self.medmisfitloc**2))
        self.medmisfitrel = self.medmisfit/self.normmtrue
        # Grad
        self.gradnorm = Obj.getGradnorm()
        if (self.gradnorminit == None) and (self.gradnorm > 0.0):
            self.gradnorminit = self.gradnorm
        if not (self.gradnorminit == None):
            self.gradnormrel = self.gradnorm/self.gradnorminit
        if self.gradnorm > 1e-16:
            srchdirnorm = np.sqrt(MPI.sum(self.mycomm, Obj.getsrchdirnorm()**2))
            if srchdirnorm > 1e-16:
                self.Gpangle = Obj.getgradxdir()/(self.gradnorm*srchdirnorm)
            else:   self.Gpangle = np.inf
        else:   self.Gpangle = np.inf
        # Line Search
        self.LSsuccess = LSresults[0]
        self.LScount = LSresults[1]
        self.LSratio = LSresults[2]
        # CG
        if self.Newt:
            self.CGiter = CGresults[0]
            self.finalnormCG = CGresults[1]
            self.tolCG = CGresults[3]

    def printResults(self, rank=0):
        """Print results on process rank"""
        if self.myrank == rank:
            if self.index == 0: 
                print self.titleline
                print self.dataline0.format(self.index, \
            self.cost, self.misfit, self.regul, \
            self.medmisfit, self.medmisfitrel)
            elif self.Newt:
                print self.dataline.format(self.index, \
            self.cost, self.misfit, self.regul, \
            self.medmisfit, self.medmisfitrel, \
            self.gradnorm, self.gradnormrel, self.Gpangle, \
            self.LSratio, self.LScount, \
            self.tolCG, self.CGiter, self.finalnormCG)
            else:
                print self.dataline.format(self.index, \
            self.cost, self.misfit, self.regul, \
            self.medmisfit, self.medmisfitrel, \
            self.gradnorm, self.gradnormrel, self.Gpangle, \
            self.LSratio, self.LScount)

    def Stop(self, rank=0):
        """Compute stopping criterion and 
        return True if iteration must stop"""
        if not self.LSsuccess:
            if self.myrank == rank:
                print 'Line Search failed after {0} counts'.format(self.LScount)
            return True
        elif self.gradnormrel < 1e-10:
            if self.myrank == rank: print 'Optimization converged!'
            return True
        else:   return False

    def alpha_init(self):
        """Evaluate initial length for line search in next iteration"""
        if self.Newt:   return 1.0
        else:
            alpha = self.LSratio
            if self.LScount == 1:    return 10.*alpha
            elif self.LScount < 5:   return 4.*alpha
            else:   return alpha
